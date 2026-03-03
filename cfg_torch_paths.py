#!/usr/bin/env python3
"""
cfg_torch_paths.py (CALL-BASED PLAN + FORWARD SUPPORT)

What changed vs block-based:
- We STILL build CFGs and print the same text report (optional).
- But the exported plan JSON is now CALL-BASED:
    Each plan item corresponds to a statement span (lineno..end_lineno)
    that contains a torch-relevant call.
- We KEEP module-level CFG in the analysis graph, but we DO NOT export
  module-scope call sites (scope has no dot), because instrumentation is
  intended for runtime regions inside functions/methods.

Forward support:
- In any scope whose name ends with ".forward", we also treat calls of the form:
      self.xxx(...)
  as call sites to instrument (even though they are not syntactically torch.*).
  This is the pragmatic fix to "forward has no torch calls" in static detection.

Export:
  --export-boundaries-json torch_boundaries.json

The plan JSON entries look like:
{
  "kind": "call",
  "scope": "train.main",
  "file": ".../train.py",
  "line_start": 42,
  "line_end": 42,
  "call": "torch.randn",
  "label": "TB:train.main.call.L42 torch=torch.randn"
}

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


# -------------------------
# Writer
# -------------------------

class ReportWriter:
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


def module_name_from_path(pyfile: Path) -> str:
    return pyfile.stem


def scope_key(scope: str, block_id: int) -> str:
    return f"{scope}.Block{block_id}"


def get_full_attr_name(node: ast.AST) -> str | None:
    """
    Reconstruct dotted attribute from AST.Attribute chain:
      torch.nn.Linear
      nn.Linear
      self.fc1
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


def block_source_lines(block) -> list[str]:
    if astor is not None:
        parts = []
        for stmt in getattr(block, "statements", []) or []:
            s = safe_to_source(stmt)
            if s:
                parts.append(s)
        if parts:
            return [ln for ln in "\n".join(parts).splitlines() if ln.strip()]

    try:
        src = block.get_source() or ""
    except Exception:
        src = ""
    return [ln for ln in src.splitlines() if ln.strip()]


def get_block_line_span(block) -> tuple[int | None, int | None]:
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


# -------------------------
# Import normalization
# -------------------------

@dataclass
class ImportIndex:
    alias_to_module: dict[str, str] = field(default_factory=dict)
    symbol_to_qual: dict[str, str] = field(default_factory=dict)

    def normalize_call(self, call: str) -> str:
        """
        Normalize:
          nn.Linear -> torch.nn.Linear
          F.pad -> torch.nn.functional.pad
          Adam -> torch.optim.Adam (if from-import)
        """
        if not call:
            return call
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
# Torch call predicate
# -------------------------

def is_torch_call(call_norm: str, torch_roots: set[str]) -> bool:
    if not call_norm:
        return False
    for r in torch_roots:
        if call_norm == r or call_norm.startswith(r + "."):
            return True
    return False


# -------------------------
# Program index (for reporting)
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

    filename: str | None = None
    line_start: int | None = None
    line_end: int | None = None

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

    torch_roots: set[str] = field(default_factory=lambda: {"torch"})


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
# Call-site extraction (NEW)
# -------------------------

def _stmt_spans(stmt: ast.AST) -> tuple[int | None, int | None]:
    ln = getattr(stmt, "lineno", None)
    en = getattr(stmt, "end_lineno", None) or ln
    return ln, en


def _collect_call_sites_from_statements(
    stmts: list[ast.AST],
    *,
    scope: str,
    filename: str,
    imp: ImportIndex,
    torch_roots: set[str],
) -> list[dict]:
    """
    Return a list of call-site dicts:
      {scope,file,line_start,line_end,call,call_norm,label}
    - If scope endswith ".forward": include self.xxx(...) calls too.
    """
    out: list[dict] = []
    seen: set[tuple[str, str, int, int]] = set()

    in_forward = scope.endswith(".forward")

    for stmt in stmts or []:
        ls, le = _stmt_spans(stmt)
        if not isinstance(ls, int) or ls <= 0:
            continue
        if not isinstance(le, int) or le <= 0:
            le = ls

        # gather calls inside this statement
        calls_here: list[str] = []
        for node in ast.walk(stmt):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func

            # torch-ish calls: Name or Attribute chain
            if isinstance(fn, ast.Name):
                calls_here.append(fn.id)
            elif isinstance(fn, ast.Attribute):
                full = get_full_attr_name(fn)
                calls_here.append(full if full else fn.attr)

        # normalize + filter
        for call_raw in calls_here:
            if not call_raw:
                continue

            # Forward heuristic: include self.xxx(...) as call sites
            if in_forward and call_raw.startswith("self.") and len(call_raw.split(".")) == 2:
                call_norm = call_raw  # keep as-is
                label = f"TB:{scope}.call.L{ls} torch={call_norm}"
                sig = (filename, call_norm, ls, le)
                if sig in seen:
                    continue
                seen.add(sig)
                out.append({
                    "kind": "call",
                    "scope": scope,
                    "file": filename,
                    "line_start": ls,
                    "line_end": le,
                    "call": call_raw,
                    "call_norm": call_norm,
                    "label": label,
                })
                continue

            call_norm = imp.normalize_call(call_raw)

            # only include real torch roots
            if not is_torch_call(call_norm, torch_roots):
                continue

            label = f"TB:{scope}.call.L{ls} torch={call_norm}"
            sig = (filename, call_norm, ls, le)
            if sig in seen:
                continue
            seen.add(sig)
            out.append({
                "kind": "call",
                "scope": scope,
                "file": filename,
                "line_start": ls,
                "line_end": le,
                "call": call_raw,
                "call_norm": call_norm,
                "label": label,
            })

    return out


# -------------------------
# Build index + plan
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


def collect_py_files(inputs: list[Path], writer: ReportWriter | None = None) -> list[Path]:
    py_files: list[Path] = []
    for p in inputs:
        if p.is_dir():
            py_files.extend(sorted(p.rglob("*.py")))
        elif p.is_file() and p.suffix == ".py":
            py_files.append(p)
    # dedup
    seen = set()
    out = []
    for f in py_files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def build_program_index_and_call_plan(
    py_files: list[Path],
    torch_roots: set[str],
    writer: ReportWriter | None = None,
) -> tuple[ProgramIndex, list[dict], str | None, int | None]:
    """
    Returns:
      index (for printing),
      call_plan (list of call-site items),
      entry_scope, entry_block_id
    """
    index = ProgramIndex()
    index.torch_roots = set(torch_roots)

    scope_to_cfg: dict[str, tuple[object, dict[int, object]]] = {}
    scope_to_file: dict[str, str] = {}
    module_dunder_main: set[str] = set()

    import_index_by_module: dict[str, ImportIndex] = {}

    # 1) build CFGs + scope map
    for f in py_files:
        mod = module_name_from_path(f)
        src = read_text(f)
        import_index_by_module[mod] = build_import_index(src)

        if detect_module_has_dunder_main(src):
            module_dunder_main.add(mod)

        cfg = CFGBuilder(separate=True).build_from_file(mod, str(f))
        scoped = build_scope_cfg_map(cfg, mod)
        scope_to_cfg.update(scoped)
        for s in scoped.keys():
            scope_to_file.setdefault(s, str(f))

    entry_scope, entry_block_id = choose_entry_scope(scope_to_cfg, module_dunder_main)

    # 2) index blocks for reporting (unchanged idea)
    node_id_counter = 0
    for scope in sorted(scope_to_cfg.keys()):
        cfg, id_to_block = scope_to_cfg[scope]
        module = scope.split(".", 1)[0] if "." in scope else scope
        imp = import_index_by_module.get(module, ImportIndex())
        filename = scope_to_file.get(scope)

        entry_id = cfg.entryblock.id if getattr(cfg, "entryblock", None) else None
        index.methods[scope] = MethodInfo(scope=scope, module=module, entry_block_id=entry_id)

        m = index.methods[scope]
        for bid in sorted(id_to_block.keys()):
            block = id_to_block[bid]
            node_id_counter += 1
            key = scope_key(scope, bid)

            # calls_raw / calls_norm for printing only (keep simple)
            calls_raw: list[str] = []
            for stmt in getattr(block, "statements", []) or []:
                for node in ast.walk(stmt):
                    if isinstance(node, ast.Call):
                        fn = node.func
                        if isinstance(fn, ast.Name):
                            calls_raw.append(fn.id)
                        elif isinstance(fn, ast.Attribute):
                            full = get_full_attr_name(fn)
                            calls_raw.append(full if full else fn.attr)
            # dedup
            tmp = []
            seen = set()
            for c in calls_raw:
                if c and c not in seen:
                    seen.add(c)
                    tmp.append(c)
            calls_raw = tmp

            calls_norm = [imp.normalize_call(c) for c in calls_raw]
            torch_calls = [c for c in calls_norm if is_torch_call(c, index.torch_roots)]

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
                calls_raw=calls_raw,
                calls_norm=calls_norm,
                torch_calls=torch_calls,
                source_lines=block_source_lines(block),
                is_entry=(entry_id == bid),
                filename=filename,
                line_start=line_start,
                line_end=line_end,
            )
            index.blocks[key] = binfo
            m.block_keys.append(key)

            index.succ_map[key] = list(succ_keys)
            for sk in succ_keys:
                index.pred_map.setdefault(sk, []).append(key)

    detect_loops(index)

    # 3) build CALL-BASED plan
    call_plan: list[dict] = []
    for scope in sorted(scope_to_cfg.keys()):
        # skip module-scope call sites for plan export
        if "." not in scope:
            continue
        module = scope.split(".", 1)[0]
        imp = import_index_by_module.get(module, ImportIndex())
        filename = scope_to_file.get(scope)
        if not filename:
            continue

        _cfg, id_to_block = scope_to_cfg[scope]

        # gather statements across all blocks in this scope
        stmts: list[ast.AST] = []
        for bid in sorted(id_to_block.keys()):
            b = id_to_block[bid]
            stmts.extend(getattr(b, "statements", []) or [])

        call_plan.extend(
            _collect_call_sites_from_statements(
                stmts,
                scope=scope,
                filename=filename,
                imp=imp,
                torch_roots=index.torch_roots,
            )
        )

    # de-dup plan globally (same file/lines/call_norm)
    uniq = {}
    for it in call_plan:
        k = (it.get("file"), it.get("line_start"), it.get("line_end"), it.get("call_norm"), it.get("scope"))
        uniq[k] = it
    call_plan = list(uniq.values())
    call_plan.sort(key=lambda x: (x["file"], x["line_start"], x["line_end"], x["scope"], x["call_norm"]))

    return index, call_plan, entry_scope, entry_block_id


# -------------------------
# Reporting
# -------------------------

def print_blocks_by_method(index: ProgramIndex, writer: ReportWriter) -> None:
    writer.write("\n=== BLOCKS BY METHOD (node ids) ===\n")
    for scope in sorted(index.methods.keys()):
        m = index.methods[scope]
        writer.write(f"-- {scope} --")
        for bkey in m.block_keys:
            b = index.blocks[bkey]
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
            writer.write(f"  Successors: [{', '.join(b.succ_keys)}]")
            writer.write(f"  TorchBoundaryCalls: {torch_pretty}")
            writer.write("  Code:")
            if b.source_lines:
                for ln in b.source_lines:
                    writer.write(f"    | {ln}")
            else:
                writer.write("    | (empty)")
            writer.write("")


def export_call_plan_json(call_plan: list[dict], out_path: Path, writer: ReportWriter | None = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(call_plan, indent=2), encoding="utf-8")
    if writer:
        writer.write(f"Exported call-based plan JSON: {out_path} ({len(call_plan)} call-sites)")


# -------------------------
# CLI
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="Build Scalpel CFG + export call-based torch instrumentation plan.")
    parser.add_argument("inputs", nargs="+", help="Python files or directories")
    parser.add_argument("--out", type=str, default="cfg_torch_paths.txt", help="Output text file path")
    parser.add_argument("--debug", action="store_true", help="Enable debug output (also written to --out)")
    parser.add_argument("--export-boundaries-json", type=str, default="",
                        help="Export CALL-based plan JSON to this path")

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

        index, call_plan, entry_scope, entry_block_id = build_program_index_and_call_plan(
            py_files, torch_roots=torch_roots, writer=writer
        )

        entry_key = None
        if entry_scope is not None and entry_block_id is not None:
            entry_key = scope_key(entry_scope, entry_block_id)

        writer.write(f"ENTRY: {entry_key or '(none)'}")
        writer.write(f"TORCH ROOTS: {sorted(list(index.torch_roots))}")
        writer.write(f"Exportable call-sites: {len(call_plan)}")
        writer.write("")

        print_blocks_by_method(index, writer)

        if args.export_boundaries_json:
            export_call_plan_json(call_plan, Path(args.export_boundaries_json), writer=writer)

        writer.write(f"\nSaved to: {args.out}")

    finally:
        writer.close()


if __name__ == "__main__":
    main()