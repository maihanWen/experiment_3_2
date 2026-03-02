#!/usr/bin/env python3
"""
Build Scalpel CFG + find paths to torch boundary + (NEW) record source locations.

Updates vs your original:
  - BlockInfo now includes: filename, line_start, line_end
  - We track which file each discovered scope came from (scope_to_file)
  - We compute line spans from block.statements (AST lineno/end_lineno)
  - NEW optional JSON export of torch-boundary blocks:
        --export-boundaries-json boundaries.json

This script still prints the same text report, but now each block (especially torch-boundary)
has a stable source anchor you can use to instrument code later.
"""

import ast
import sys
import json
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from collections import deque, defaultdict

from scalpel.cfg import CFGBuilder

try:
    import astor
except ImportError:
    astor = None


class ReportWriter:
    """
    Write to both stdout and an optional output file.

    Usage:
        writer = ReportWriter("out.txt", debug=True)
        writer.write("hello")
        writer.debug("only if debug enabled")
        writer.close()
    """

    def __init__(self, filepath: str | None = None, debug: bool = False):
        self.filepath = filepath
        self.debug_enabled = debug
        self.fp = open(filepath, "w", encoding="utf-8") if filepath else None

    def write(self, text: str = ""):
        print(text)
        if self.fp:
            self.fp.write(text + "\n")

    def debug(self, text: str = ""):
        if self.debug_enabled:
            self.write(f"[DEBUG] {text}")

    def close(self):
        if self.fp:
            self.fp.close()


# -------------------------
# Utilities
# -------------------------

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def safe_to_source(node: ast.AST) -> str:
    if astor is None:
        return ""
    try:
        return astor.to_source(node).rstrip()
    except Exception:
        return ""


def get_full_attr_name(node: ast.AST) -> str | None:
    """
    Reconstruct full dotted attribute name from AST.
    Examples:
        torch.utils.data.DataLoader
        nn.Conv2d
        F.pad
    Returns None if it can't be reconstructed to a Name/Attribute chain.
    """
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return ".".join(reversed(parts))
    return None


def block_source_lines(block) -> list[str]:
    # Prefer AST pretty print if available for full statements
    if astor is not None:
        parts = []
        for stmt in getattr(block, "statements", []) or []:
            s = safe_to_source(stmt)
            if s:
                parts.append(s)
        if parts:
            return [ln for ln in "\n".join(parts).splitlines() if ln.strip()]

    # fallback
    try:
        src = block.get_source() or ""
    except Exception:
        src = ""
    return [ln for ln in src.splitlines() if ln.strip()]


def get_block_line_span(block) -> tuple[int | None, int | None]:
    """
    Compute (min lineno, max end_lineno) from block.statements.
    Works best on Python 3.8+ where end_lineno exists.
    """
    l0: int | None = None
    l1: int | None = None
    for stmt in getattr(block, "statements", []) or []:
        ln = getattr(stmt, "lineno", None)
        en = getattr(stmt, "end_lineno", None) or ln
        if ln is None:
            continue
        l0 = ln if l0 is None else min(l0, ln)
        l1 = en if l1 is None else max(l1, en)
    return l0, l1


def get_block_call_names(block) -> list[str]:
    """
    Extract call names from a scalpel block.

    Priority:
      1) scalpel's block.func_calls if present
      2) AST walk through block.statements (if present) to find ast.Call nodes
         and reconstruct full dotted function names (torch.utils.data.DataLoader).
    """
    names: list[str] = []

    # 1) scalpel's func_calls
    for entry in getattr(block, "func_calls", []) or []:
        if isinstance(entry, dict):
            n = entry.get("name", "")
            if n:
                names.append(n)
        elif isinstance(entry, (list, tuple)) and entry:
            names.append(str(entry[0]))
        else:
            names.append(str(entry))

    # 2) AST walk (works even if astor isn't installed)
    for stmt in getattr(block, "statements", []) or []:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Call):
                fn = node.func
                if isinstance(fn, ast.Name):
                    names.append(fn.id)
                elif isinstance(fn, ast.Attribute):
                    full = get_full_attr_name(fn)
                    if full:
                        names.append(full)
                    else:
                        names.append(fn.attr)

    # de-dup preserving order
    out: list[str] = []
    seen: set[str] = set()
    for n in names:
        n = (n or "").strip()
        if not n or n in seen:
            continue
        seen.add(n)
        out.append(n)
    return out


def module_name_from_path(pyfile: Path) -> str:
    # best-effort: file stem as module
    return pyfile.stem


def scope_key(scope: str, block_id: int) -> str:
    return f"{scope}.Block{block_id}"


# -------------------------
# Import resolution (for torch boundary normalization)
# -------------------------

@dataclass
class ImportIndex:
    # alias -> fully qualified module
    alias_to_module: dict[str, str] = field(default_factory=dict)
    # imported symbol -> fully qualified name
    symbol_to_qual: dict[str, str] = field(default_factory=dict)

    def normalize_call(self, call: str) -> str:
        """
        Normalize call like:
          - nn.Conv2d -> torch.nn.Conv2d  (alias)
          - F.pad -> torch.nn.functional.pad
          - pad -> torch.nn.functional.pad  (from-import symbol)
          - torch.xxx stays
        """
        if not call:
            return call

        # direct symbol import
        if call in self.symbol_to_qual:
            return self.symbol_to_qual[call]

        if "." not in call:
            return call

        head, tail = call.split(".", 1)
        if head in self.alias_to_module:
            return f"{self.alias_to_module[head]}.{tail}"
        return call


def build_import_index(py_src: str) -> ImportIndex:
    idx = ImportIndex()
    try:
        tree = ast.parse(py_src)
    except Exception:
        return idx

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                name = a.name
                asname = a.asname or name.split(".")[0]
                idx.alias_to_module[asname] = name

        elif isinstance(node, ast.ImportFrom):
            if not node.module:
                continue
            mod = node.module
            for a in node.names:
                if a.name == "*":
                    continue
                asname = a.asname or a.name
                full = f"{mod}.{a.name}"
                idx.alias_to_module[asname] = full
                idx.symbol_to_qual[asname] = full

    return idx


# -------------------------
# Graph data structures
# -------------------------

@dataclass
class BlockInfo:
    node_id: int
    key: str
    scope: str
    block_id: int
    succ_keys: list[str] = field(default_factory=list)

    calls_raw: list[str] = field(default_factory=list)
    calls_norm: list[str] = field(default_factory=list)
    torch_calls: list[str] = field(default_factory=list)

    source_lines: list[str] = field(default_factory=list)
    is_entry: bool = False

    # NEW: source anchor (needed for instrumentation)
    filename: str | None = None
    line_start: int | None = None
    line_end: int | None = None

    # loop tags
    is_loop_header: bool = False
    is_loop_backedge: bool = False


@dataclass
class MethodInfo:
    scope: str
    module: str
    entry_block_id: int | None
    block_keys: list[str] = field(default_factory=list)


@dataclass
class ProgramIndex:
    methods: dict[str, MethodInfo] = field(default_factory=dict)
    blocks: dict[str, BlockInfo] = field(default_factory=dict)

    succ_map: dict[str, list[str]] = field(default_factory=dict)
    pred_map: dict[str, list[str]] = field(default_factory=dict)

    # Interprocedural: caller block -> callee entry block
    block_call_edges: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))

    import_index_by_module: dict[str, ImportIndex] = field(default_factory=dict)

    torch_roots: set[str] = field(default_factory=lambda: {"torch"})
    torch_boundary_blocks: set[str] = field(default_factory=set)


# -------------------------
# CFG build + scope discovery
# -------------------------

def list_all_blocks(cfg) -> list:
    try:
        return cfg.get_all_blocks()
    except Exception:
        out = []
        q = deque([cfg.entryblock] if getattr(cfg, "entryblock", None) else [])
        seen = set()
        while q:
            b = q.popleft()
            if b is None or b.id in seen:
                continue
            seen.add(b.id)
            out.append(b)
            for link in getattr(b, "exits", []) or []:
                q.append(link.target)
        return out


def build_scope_cfg_map(cfg, scope: str) -> dict[str, tuple[object, dict[int, object]]]:
    """
    scope -> (cfg, {block_id: block})
    include nested functioncfgs and class methods
    """
    result: dict[str, tuple[object, dict[int, object]]] = {}

    blocks = list_all_blocks(cfg)
    id_to_block = {b.id: b for b in blocks}
    result[scope] = (cfg, id_to_block)

    for (_, fun_name), fun_cfg in getattr(cfg, "functioncfgs", {}).items():
        result.update(build_scope_cfg_map(fun_cfg, f"{scope}.{fun_name}"))

    for cls_name, cls_cfg in getattr(cfg, "class_cfgs", {}).items():
        for (_, method_name), method_cfg in getattr(cls_cfg, "functioncfgs", {}).items():
            result.update(build_scope_cfg_map(method_cfg, f"{scope}.{cls_name}.{method_name}"))

    return result


def resolve_callee_scope(
    call_name: str,
    current_scope: str,
    scope_to_cfg: dict,
    imp: ImportIndex | None = None,
) -> str | None:
    """
    Best-effort resolution from a call string to a known scalpel scope in scope_to_cfg.
    Supports:
      - foo() -> module.foo or parent.foo
      - module.foo() (qualified)
      - alias.foo() via ImportIndex.normalize_call
      - ClassName(...) -> ...ClassName.__init__  (constructor heuristic)
    """
    if not call_name:
        return None

    module = current_scope.split(".", 1)[0] if "." in current_scope else current_scope

    candidates_to_try: list[str] = [call_name]
    if imp is not None:
        norm = imp.normalize_call(call_name)
        if norm and norm != call_name:
            candidates_to_try.append(norm)

    def try_match(scope_candidate: str) -> str | None:
        if scope_candidate in scope_to_cfg:
            return scope_candidate

        ctor_suffix = f".{scope_candidate}.__init__" if "." not in scope_candidate else f"{scope_candidate}.__init__"
        ctor_matches = [s for s in scope_to_cfg.keys() if s.endswith(ctor_suffix)]
        if ctor_matches:
            mod_matches = [s for s in ctor_matches if s.startswith(module + ".")]
            if len(mod_matches) == 1:
                return mod_matches[0]
            if len(ctor_matches) == 1:
                return ctor_matches[0]

        suffix = "." + scope_candidate
        suffix_matches = [s for s in scope_to_cfg.keys() if s.endswith(suffix)]
        mod_matches = [s for s in suffix_matches if s.startswith(module + ".")]
        if len(mod_matches) == 1:
            return mod_matches[0]
        if len(suffix_matches) == 1:
            return suffix_matches[0]

        return None

    for name in candidates_to_try:
        if not name:
            continue

        if "." in name:
            m = try_match(name)
            if m:
                return m

            parts = name.split(".")
            for i in range(1, len(parts)):
                suffix_name = ".".join(parts[i:])
                m = try_match(suffix_name)
                if m:
                    return m
            continue

        unq = name
        local_candidates = []
        if "." in current_scope:
            parent = current_scope.rsplit(".", 1)[0]
            local_candidates.append(f"{parent}.{unq}")
        local_candidates.append(f"{module}.{unq}")
        local_candidates.append(unq)

        for c in local_candidates:
            m = try_match(c)
            if m:
                return m

        m = try_match(unq)
        if m:
            return m

    return None


# -------------------------
# Entry detection (possible main)
# -------------------------

def detect_module_has_dunder_main(py_src: str) -> bool:
    try:
        t = ast.parse(py_src)
    except Exception:
        return False

    for node in t.body:
        if isinstance(node, ast.If):
            test = node.test
            if isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq):
                left = test.left
                rights = test.comparators
                if (
                    isinstance(left, ast.Name)
                    and left.id == "__name__"
                    and len(rights) == 1
                    and isinstance(rights[0], ast.Constant)
                    and rights[0].value == "__main__"
                ):
                    return True
    return False


def choose_entry_scope(scope_to_cfg: dict, module_dunder_main: set[str]) -> tuple[str | None, int | None]:
    main_scopes = [s for s in scope_to_cfg.keys() if s.endswith(".main")]

    def score(s: str) -> tuple[int, str]:
        mod = s.split(".", 1)[0]
        return (0 if mod in module_dunder_main else 1, s)

    for s in sorted(main_scopes, key=score):
        cfg, _ = scope_to_cfg[s]
        if getattr(cfg, "entryblock", None):
            return s, cfg.entryblock.id

    for mod in sorted(module_dunder_main):
        if mod in scope_to_cfg:
            cfg, _ = scope_to_cfg[mod]
            if getattr(cfg, "entryblock", None):
                return mod, cfg.entryblock.id

    for s in sorted(scope_to_cfg.keys()):
        cfg, _ = scope_to_cfg[s]
        if getattr(cfg, "entryblock", None):
            return s, cfg.entryblock.id

    return None, None


# -------------------------
# Torch boundary detection
# -------------------------

def is_torch_call(call_norm: str, torch_roots: set[str]) -> bool:
    if not call_norm:
        return False
    for r in torch_roots:
        if call_norm == r or call_norm.startswith(r + "."):
            return True
    return False


# -------------------------
# Loop detection
# -------------------------

def detect_loops(index: ProgramIndex) -> None:
    visited: set[str] = set()
    stack: set[str] = set()

    def dfs(node: str):
        visited.add(node)
        stack.add(node)

        for nxt in index.succ_map.get(node, []):
            if nxt not in visited:
                dfs(nxt)
            elif nxt in stack:
                b_tgt = index.blocks.get(nxt)
                if b_tgt:
                    b_tgt.is_loop_header = True
                b_src = index.blocks.get(node)
                if b_src:
                    b_src.is_loop_backedge = True

        stack.remove(node)

    for node in index.blocks.keys():
        if node not in visited:
            dfs(node)


# -------------------------
# Path search: ENTRY -> torch boundary (collect all)
# -------------------------

def find_paths_to_torch_all(
    index: ProgramIndex,
    entry_block_key: str,
    max_paths: int = 500,
    max_depth: int = 120,
    allow_revisit: bool = False,
) -> list[list[str]]:
    results: list[list[str]] = []
    q = deque([[entry_block_key]])
    seen_expand: set[tuple[str, int]] = set()

    while q and len(results) < max_paths:
        path = q.popleft()
        cur = path[-1]
        depth = len(path)

        if depth > max_depth:
            continue

        if cur in index.torch_boundary_blocks:
            results.append(path)
            if len(results) >= max_paths:
                break

        sig = (cur, depth)
        if sig in seen_expand:
            continue
        seen_expand.add(sig)

        nexts: list[str] = []
        nexts.extend(index.succ_map.get(cur, []))
        nexts.extend(list(index.block_call_edges.get(cur, set())))

        for nxt in nexts:
            if not allow_revisit and nxt in path:
                continue
            q.append(path + [nxt])

    return results


def find_paths_from_node_all(
    index: ProgramIndex,
    start_block_key: str,
    max_paths: int = 200,
    max_depth: int = 80,
    allow_revisit: bool = False,
    stop_at_deadend: bool = True,
) -> list[list[str]]:
    results: list[list[str]] = []
    q = deque([[start_block_key]])
    seen_expand: set[tuple[str, int]] = set()

    while q and len(results) < max_paths:
        path = q.popleft()
        cur = path[-1]
        depth = len(path)

        if depth > max_depth:
            results.append(path)
            continue

        nexts: list[str] = []
        nexts.extend(index.succ_map.get(cur, []))
        nexts.extend(list(index.block_call_edges.get(cur, set())))

        if stop_at_deadend and not nexts:
            results.append(path)
            continue

        sig = (cur, depth)
        if sig in seen_expand:
            continue
        seen_expand.add(sig)

        for nxt in nexts:
            if not allow_revisit and nxt in path:
                continue
            q.append(path + [nxt])

    return results


# -------------------------
# Debug: torch boundary dump
# -------------------------

def debug_print_torch_boundaries(index: ProgramIndex, writer: ReportWriter, limit: int = 200) -> None:
    writer.debug("=== TORCH BOUNDARY DEBUG DUMP ===")
    writer.debug(f"torch_roots={sorted(index.torch_roots)}")
    writer.debug(f"torch_boundary_blocks={len(index.torch_boundary_blocks)}")

    if not index.torch_boundary_blocks:
        writer.debug("No torch boundary blocks detected.")
        return

    def sort_key(bkey: str) -> tuple[str, int]:
        b = index.blocks.get(bkey)
        if not b:
            return (bkey, 0)
        return (b.scope, b.block_id)

    for i, bkey in enumerate(sorted(index.torch_boundary_blocks, key=sort_key), start=1):
        if i > limit:
            writer.debug(f"... truncated after {limit} boundary blocks ...")
            break

        b = index.blocks.get(bkey)
        if not b:
            writer.debug(f"[{i}] {bkey} (missing BlockInfo)")
            continue

        loop_tag = ""
        if b.is_loop_header:
            loop_tag += " [LOOP_HEADER]"
        if b.is_loop_backedge:
            loop_tag += " [LOOP_BACKEDGE]"

        loc = ""
        if b.filename and b.line_start:
            loc = f" @ {b.filename}:{b.line_start}-{b.line_end or b.line_start}"

        writer.debug(f"[{i}] {b.key}{loop_tag}{loc}")
        writer.debug(f"     torch_calls={b.torch_calls}")

        if b.source_lines:
            max_lines = 8
            for ln in b.source_lines[:max_lines]:
                writer.debug(f"     | {ln}")
            if len(b.source_lines) > max_lines:
                writer.debug(f"     | ... (+{len(b.source_lines) - max_lines} more lines)")
        else:
            writer.debug("     | (no source lines)")


# -------------------------
# Build ProgramIndex from files
# -------------------------

def build_program_index(
    py_files: list[Path],
    torch_roots: set[str] | None = None,
    writer: ReportWriter | None = None,
) -> tuple[ProgramIndex, str | None, int | None]:
    index = ProgramIndex()
    if torch_roots:
        index.torch_roots = set(torch_roots)

    scope_to_cfg: dict[str, tuple[object, dict[int, object]]] = {}
    scope_to_file: dict[str, str] = {}  # NEW: scope -> source file path
    module_dunder_main: set[str] = set()

    for f in py_files:
        if writer:
            writer.debug(f"Processing file: {f}")

        if not f.exists() or f.suffix != ".py":
            if writer:
                writer.debug(f"Skipping (not an existing .py): {f}")
            continue

        mod = module_name_from_path(f)
        src = read_text(f)
        index.import_index_by_module[mod] = build_import_index(src)
        if detect_module_has_dunder_main(src):
            module_dunder_main.add(mod)
            if writer:
                writer.debug(f"Detected __main__ guard in module: {mod}")

        cfg = CFGBuilder(separate=True).build_from_file(mod, str(f))
        scoped = build_scope_cfg_map(cfg, mod)
        scope_to_cfg.update(scoped)
        # record file for every discovered scope originating from this file
        for s in scoped.keys():
            scope_to_file.setdefault(s, str(f))

    entry_scope, entry_block_id = choose_entry_scope(scope_to_cfg, module_dunder_main)

    # Method table
    for scope, (cfg, _id_to_block) in scope_to_cfg.items():
        module = scope.split(".", 1)[0] if "." in scope else scope
        entry_id = cfg.entryblock.id if getattr(cfg, "entryblock", None) else None
        index.methods[scope] = MethodInfo(scope=scope, module=module, entry_block_id=entry_id)

    # Block table + CFG edges
    node_id_counter = 0
    for scope in sorted(scope_to_cfg.keys()):
        _cfg, id_to_block = scope_to_cfg[scope]
        module = scope.split(".", 1)[0] if "." in scope else scope
        imp = index.import_index_by_module.get(module, ImportIndex())
        filename = scope_to_file.get(scope)

        method = index.methods[scope]
        for bid in sorted(id_to_block.keys()):
            block = id_to_block[bid]
            node_id_counter += 1
            key = scope_key(scope, bid)

            raw_calls = get_block_call_names(block)
            norm_calls = [imp.normalize_call(c) for c in raw_calls]
            torch_calls = [c for c in norm_calls if is_torch_call(c, index.torch_roots)]

            succ_keys = []
            for link in getattr(block, "exits", []) or []:
                if link.target is None:
                    continue
                succ_keys.append(scope_key(scope, link.target.id))

            line_start, line_end = get_block_line_span(block)

            binfo = BlockInfo(
                node_id=node_id_counter,
                key=key,
                scope=scope,
                block_id=bid,
                succ_keys=succ_keys,
                calls_raw=raw_calls,
                calls_norm=norm_calls,
                torch_calls=torch_calls,
                source_lines=block_source_lines(block),
                is_entry=(method.entry_block_id == bid),
                filename=filename,
                line_start=line_start,
                line_end=line_end,
            )
            index.blocks[key] = binfo
            method.block_keys.append(key)

            index.succ_map[key] = list(succ_keys)
            for sk in succ_keys:
                index.pred_map.setdefault(sk, []).append(key)

            if torch_calls:
                index.torch_boundary_blocks.add(key)

    # Interprocedural call edges
    for bkey, binfo in index.blocks.items():
        scope = binfo.scope
        module = scope.split(".", 1)[0] if "." in scope else scope
        imp = index.import_index_by_module.get(module, ImportIndex())

        for raw_call in binfo.calls_raw:
            callee_scope = resolve_callee_scope(raw_call, scope, scope_to_cfg, imp=imp)
            if not callee_scope:
                continue
            callee_cfg, _ = scope_to_cfg[callee_scope]
            if not getattr(callee_cfg, "entryblock", None):
                continue
            callee_entry_key = scope_key(callee_scope, callee_cfg.entryblock.id)

            # avoid self call-edges unless explicitly desired
            if callee_entry_key == bkey:
                continue

            index.block_call_edges[bkey].add(callee_entry_key)

    if writer:
        writer.debug(f"Total scopes discovered: {len(scope_to_cfg)}")
        writer.debug(f"Total blocks indexed: {len(index.blocks)}")
        writer.debug(f"Torch-boundary blocks: {len(index.torch_boundary_blocks)}")
        if entry_scope is not None and entry_block_id is not None:
            writer.debug(f"Chosen entry scope: {entry_scope}, entry block id: {entry_block_id}")
        else:
            writer.debug("No entry scope/block chosen.")

    return index, entry_scope, entry_block_id


# -------------------------
# Reporting (writer-based)
# -------------------------

def print_blocks_by_method(index: ProgramIndex, writer: ReportWriter) -> None:
    writer.write("\n=== BLOCKS BY METHOD (node ids) ===\n")
    for scope in sorted(index.methods.keys()):
        m = index.methods[scope]
        writer.write(f"-- {scope} --")
        for bkey in m.block_keys:
            b = index.blocks[bkey]
            succ_pretty = [s for s in b.succ_keys]

            loop_tag = ""
            if b.is_loop_header:
                loop_tag += " [LOOP]"
            if b.is_loop_backedge:
                loop_tag += " [BACKEDGE]"

            torch_pretty = ", ".join(b.torch_calls) if b.torch_calls else "(none)"
            loc = ""
            if b.filename and b.line_start:
                loc = f" @ {b.filename}:{b.line_start}-{b.line_end or b.line_start}"

            writer.write(f"  NodeID: {b.node_id}")
            writer.write(f"  Key: {b.key}{loop_tag}{loc}")
            writer.write(f"  Successors: [{', '.join(succ_pretty)}]")
            writer.write(f"  TorchBoundaryCalls: {torch_pretty}")
            writer.write("  Code:")
            if b.source_lines:
                for ln in b.source_lines:
                    writer.write(f"    | {ln}")
            else:
                writer.write("    | (empty)")
            writer.write("")


def print_cfg_edges(
    index: ProgramIndex,
    writer: ReportWriter,
    *,
    filter_self_loops: bool = True,
    keep_loop_self_edges: bool = True,
) -> None:
    """
    Filters CFG self-loops (A -> A) by default.
    If keep_loop_self_edges=True, we keep A -> A only when the block is tagged as a loop backedge/header.
    """
    writer.write("\n=== CFG EDGES (method.BlockId) ===\n")
    edges = []
    for src, succs in index.succ_map.items():
        for dst in succs:
            if filter_self_loops and src == dst:
                if keep_loop_self_edges:
                    b = index.blocks.get(src)
                    if b and (b.is_loop_backedge or b.is_loop_header):
                        edges.append((src, dst))
                continue
            edges.append((src, dst))

    for src, dst in sorted(edges):
        writer.write(f"{src} -> {dst}")


def print_call_edges(index: ProgramIndex, writer: ReportWriter) -> None:
    writer.write("\n=== CALL EDGES (caller => callee_entry) ===\n")
    edges = []
    for src, tgts in index.block_call_edges.items():
        for dst in tgts:
            edges.append((src, dst))
    if not edges:
        writer.write("(no call edges found)")
        return
    for src, dst in sorted(edges):
        writer.write(f"{src} => {dst}")


def print_paths_out_from_torch(
    index: ProgramIndex,
    writer: ReportWriter,
    max_boundaries: int = 50,
    max_paths_per_boundary: int = 50,
    max_depth: int = 80,
) -> None:
    writer.write("\n=== PATHS: TORCH BOUNDARY -> OUT (forward) ===\n")

    boundaries = sorted(index.torch_boundary_blocks)
    if not boundaries:
        writer.write("(no torch boundary blocks)")
        return

    if len(boundaries) > max_boundaries:
        writer.write(f"(showing first {max_boundaries} of {len(boundaries)} boundary blocks)")
        boundaries = boundaries[:max_boundaries]

    for bkey in boundaries:
        b = index.blocks.get(bkey)
        torch_calls = ", ".join(getattr(b, "torch_calls", [])) if b else ""
        loc = ""
        if b and b.filename and b.line_start:
            loc = f" @ {b.filename}:{b.line_start}-{b.line_end or b.line_start}"

        writer.write(f"\n-- START TORCH BOUNDARY: {bkey}{loc} --")
        if torch_calls:
            writer.write(f"   TorchCalls: {torch_calls}")

        paths = find_paths_from_node_all(
            index,
            bkey,
            max_paths=max_paths_per_boundary,
            max_depth=max_depth,
            allow_revisit=False,
            stop_at_deadend=True,
        )

        if not paths:
            writer.write("  (no outgoing paths)")
            continue

        for i, p in enumerate(paths, start=1):
            writer.write(f"  PathOut {i}:")
            for node in p:
                bn = index.blocks.get(node)
                tag = ""
                if bn:
                    if bn.is_loop_header:
                        tag += " [LOOP]"
                    if bn.is_loop_backedge:
                        tag += " [BACKEDGE]"
                writer.write(f"    {node}{tag}")


def print_paths(entry_key: str, index: ProgramIndex, writer: ReportWriter, max_paths: int = 50) -> None:
    writer.write("\n=== PATHS: ENTRY -> TORCH BOUNDARY (ALL) ===\n")
    paths = find_paths_to_torch_all(
        index,
        entry_key,
        max_paths=max_paths,
        max_depth=120,
        allow_revisit=False,
    )
    if not paths:
        writer.write("(no paths found)")
        return

    for i, path in enumerate(paths, start=1):
        last = path[-1]
        torch_calls = ", ".join(index.blocks[last].torch_calls) if last in index.blocks else ""
        writer.write(f"Path {i}:")
        for j, node in enumerate(path):
            b = index.blocks.get(node)
            tag = ""
            if b:
                if b.is_loop_header:
                    tag += " [LOOP]"
                if b.is_loop_backedge:
                    tag += " [BACKEDGE]"

            if j == len(path) - 1 and node in index.torch_boundary_blocks:
                writer.write(f"  {node}{tag}  <-- TORCH: {torch_calls}")
            else:
                writer.write(f"  {node}{tag}")
        writer.write("")


# -------------------------
# NEW: Export torch boundary blocks as JSON plan
# -------------------------

def export_torch_boundaries_json(index: ProgramIndex, out_path: Path, writer: ReportWriter | None = None) -> None:
    """
    Export an instrumentation plan:
      - Only torch boundary blocks
      - Includes file + line range + torch_calls
    """
    items = []
    for bkey in sorted(index.torch_boundary_blocks):
        b = index.blocks.get(bkey)
        if not b:
            continue
        items.append({
            "block_key": b.key,
            "scope": b.scope,
            "block_id": b.block_id,
            "file": b.filename,
            "line_start": b.line_start,
            "line_end": b.line_end,
            "torch_calls": list(b.torch_calls),
            "is_loop_header": bool(b.is_loop_header),
            "is_loop_backedge": bool(b.is_loop_backedge),
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(items, indent=2), encoding="utf-8")
    if writer:
        writer.write(f"Exported torch boundary plan JSON: {out_path} ({len(items)} blocks)")


# -------------------------
# CLI entry
# -------------------------

def collect_py_files(inputs: list[Path], writer: ReportWriter | None = None) -> list[Path]:
    py_files: list[Path] = []
    if writer:
        writer.debug("Collecting Python files...")

    for p in inputs:
        if writer:
            writer.debug(f"Checking input: {p}")

        if p.is_dir():
            if writer:
                writer.debug(f"Scanning directory recursively: {p}")
            found = sorted(p.rglob("*.py"))
            if writer:
                writer.debug(f"Found {len(found)} .py file(s) under: {p}")
                for f in found:
                    writer.debug(f"  Found: {f}")
            py_files.extend(found)

        elif p.is_file() and p.suffix == ".py":
            if writer:
                writer.debug(f"Single Python file: {p}")
            py_files.append(p)
        else:
            if writer:
                writer.debug(f"Skipping input (not dir or .py file): {p}")

    seen = set()
    out = []
    for f in py_files:
        if f not in seen:
            seen.add(f)
            out.append(f)

    if writer:
        writer.debug(f"Total unique Python files collected: {len(out)}")

    return out


def main():
    parser = argparse.ArgumentParser(description="Build Scalpel CFG + find paths to torch boundary.")
    parser.add_argument("inputs", nargs="+", help="Python files or directories")
    parser.add_argument("--out", type=str, default="cfg_torch_paths.txt", help="Output text file path")
    parser.add_argument("--max-paths", type=int, default=200, help="Max paths to print (entry->torch)")
    parser.add_argument("--debug", action="store_true", help="Enable debug output (also written to --out)")

    # Output filtering controls
    parser.add_argument("--no-filter-self-loops", action="store_true",
                        help="Do not filter CFG self-loops (A -> A) in printed CFG edges")
    parser.add_argument("--drop-loop-self-edges", action="store_true",
                        help="If filtering self-loops, also drop loop-marked self-edges")

    # NEW: export torch boundary instrumentation plan
    parser.add_argument("--export-boundaries-json", type=str, default="",
                        help="If set, export torch-boundary blocks with file/line ranges to this JSON file")

    args = parser.parse_args()

    writer = ReportWriter(args.out, debug=args.debug)

    try:
        inputs = [Path(x) for x in args.inputs]
        py_files = collect_py_files(inputs, writer)

        torch_roots = {
            "torch",
            "torch.nn",
            "torch.nn.functional",
            "torch.optim",
            "torch.utils",
        }

        index, entry_scope, entry_block_id = build_program_index(
            py_files,
            torch_roots=torch_roots,
            writer=writer,
        )

        detect_loops(index)
        debug_print_torch_boundaries(index, writer, limit=200)

        if args.export_boundaries_json:
            export_torch_boundaries_json(index, Path(args.export_boundaries_json), writer=writer)

        if entry_scope is None or entry_block_id is None:
            writer.write("No entry found.")
            sys.exit(2)

        entry_key = scope_key(entry_scope, entry_block_id)

        writer.write(f"ENTRY: {entry_key}")
        writer.write(f"TORCH BOUNDARY ROOTS: {sorted(list(index.torch_roots))}")
        writer.write(f"Detected {len(index.torch_boundary_blocks)} torch-boundary block(s).")

        print_blocks_by_method(index, writer)

        print_cfg_edges(
            index,
            writer,
            filter_self_loops=not args.no_filter_self_loops,
            keep_loop_self_edges=not args.drop_loop_self_edges,
        )
        print_call_edges(index, writer)

        print_paths(entry_key, index, writer, max_paths=args.max_paths)
        print_paths_out_from_torch(index, writer, max_boundaries=50, max_paths_per_boundary=25, max_depth=80)

        writer.write(f"\nSaved to: {args.out}")

    finally:
        writer.close()


if __name__ == "__main__":
    main()