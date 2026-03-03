#!/usr/bin/env python3
"""
instrument.py (CALL-SITE INSTRUMENTATION + LOOP PRUNE)

Fix:
- If two callsites overlap and one strictly contains another, drop the larger one.
  This specifically removes loop-header spans (e.g. for-loop node) when inner
  statements exist, preventing double TB markers for the same loop body.

Plan format expected:
[
  {
    "kind": "call",
    "scope": "...",
    "file": "...",
    "line_start": 40,
    "line_end": 54,
    "call_norm": "torch.randn",
    "label": "TB:..."
  },
  ...
]
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict


PROFILE_RUNNER_NAME = "run_profile.py"


@dataclass(frozen=True)
class CallSite:
    file: str
    scope: str
    line_start: int
    line_end: int
    label: str
    call_norm: str  # used for smarter pruning


def _load_plan(plan_path: Path) -> list[CallSite]:
    data = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Plan JSON must be a list of objects.")

    out: list[CallSite] = []
    for obj in data:
        if not isinstance(obj, dict):
            continue
        if obj.get("kind") != "call":
            continue

        f = obj.get("file")
        scope = obj.get("scope") or ""
        ls = obj.get("line_start")
        le = obj.get("line_end")
        label = obj.get("label") or ""
        call_norm = obj.get("call_norm") or ""

        if not f or not isinstance(ls, int) or ls <= 0:
            continue
        if not isinstance(le, int) or le <= 0:
            le = ls
        if le < ls:
            ls, le = le, ls
        if not label:
            label = f"TB:{scope}.call.L{ls}"

        out.append(
            CallSite(
                file=str(f),
                scope=str(scope),
                line_start=ls,
                line_end=le,
                label=str(label),
                call_norm=str(call_norm),
            )
        )
    return out


def _read_lines(p: Path) -> list[str]:
    return p.read_text(encoding="utf-8").splitlines()


def _write_lines(p: Path, lines: list[str]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _line_indent_len(s: str) -> int:
    return len(s) - len(s.lstrip(" \t"))


def _is_def_or_class_line(s: str) -> bool:
    st = s.lstrip()
    return st.startswith("def ") or st.startswith("class ")


def _is_noncode_line(s: str) -> bool:
    st = s.strip()
    return st == "" or st.startswith("#")


def _dedup_exact(callsites: list[CallSite]) -> list[CallSite]:
    uniq = {}
    for c in callsites:
        uniq[(c.file, c.line_start, c.line_end, c.call_norm, c.label)] = c
    return list(uniq.values())


def _prune_containing_spans(callsites: list[CallSite]) -> list[CallSite]:
    """
    Drop any callsite whose span strictly contains another callsite span
    in the same (file, scope), regardless of ordering.
    """
    groups: dict[tuple[str, str], list[CallSite]] = defaultdict(list)
    for c in callsites:
        groups[(c.file, c.scope)].append(c)

    kept_all: list[CallSite] = []

    for (file, scope), items in groups.items():
        # O(n^2) but n is small per file/scope
        drop = set()
        for i, a in enumerate(items):
            for j, b in enumerate(items):
                if i == j:
                    continue
                if (a.line_start <= b.line_start and a.line_end >= b.line_end) and (
                    (a.line_start, a.line_end) != (b.line_start, b.line_end)
                ):
                    drop.add(i)
                    break

        for i, c in enumerate(items):
            if i not in drop:
                kept_all.append(c)

    kept_all.sort(key=lambda c: (c.file, c.line_start, c.line_end, c.scope, c.call_norm, c.label))
    return kept_all


def _apply_callsites_to_lines(lines: list[str], callsites: list[CallSite]) -> list[str]:
    if not callsites:
        return lines

    out = list(lines)

    inserted_positions: list[int] = []
    for c in reversed(callsites):
        ls = c.line_start
        le = c.line_end

        if inserted_positions:
            ls += sum(1 for pos in inserted_positions if pos <= ls)
            le += sum(1 for pos in inserted_positions if pos <= le)

        le = min(le, len(out))
        if ls > len(out) or le < ls:
            continue

        if _is_def_or_class_line(out[ls - 1]):
            continue

        base_i = ls - 1
        while base_i <= le - 1 and _is_noncode_line(out[base_i]):
            base_i += 1
        if base_i > le - 1:
            continue

        base_indent_len = _line_indent_len(out[base_i])
        base_indent = out[base_i][:base_indent_len]

        clamp_le = le
        for j in range(base_i, le):
            ln = out[j]
            if _is_noncode_line(ln):
                continue
            if _line_indent_len(ln) < base_indent_len:
                clamp_le = j
                break
        le = clamp_le
        if le < ls:
            continue

        label = (c.label or "").replace("\n", " ").replace("\r", " ")
        with_line = f'{base_indent}with __import__("torch").profiler.record_function("{label}"):'

        new_block: list[str] = []
        for j in range(ls - 1, le):
            ln = out[j]
            if ln.strip() == "":
                new_block.append(base_indent + "    ")
                continue
            if not ln.startswith(base_indent):
                new_block.append(base_indent + "    " + ln.lstrip(" \t"))
            else:
                new_block.append(base_indent + "    " + ln[len(base_indent):])

        out = out[: ls - 1] + [with_line] + new_block + out[le:]
        inserted_positions.append(ls)

    return out


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")
    shutil.copytree(src, dst)


def _group_by_file(callsites: list[CallSite]) -> dict[str, list[CallSite]]:
    by_file: dict[str, list[CallSite]] = {}
    for c in callsites:
        by_file.setdefault(c.file, []).append(c)
    return by_file


def _resolve_plan_file_to_dst(plan_file: Path, src_root: Path, dst_root: Path, package_root_name: str) -> Path | None:
    plan_file = plan_file.resolve()
    src_root = src_root.resolve()

    try:
        rel = plan_file.relative_to(src_root)
        cand = dst_root / rel
        return cand if cand.exists() else None
    except Exception:
        pass

    parts = list(plan_file.parts)
    if package_root_name in parts:
        i = parts.index(package_root_name)
        suffix = Path(*parts[i + 1 :])
        cand = dst_root / suffix
        if cand.exists():
            return cand

    cand = dst_root / plan_file.name
    if cand.exists():
        return cand

    return None


def _write_profiler_runner(dst_root: Path, entry_script: str = "train.py") -> Path:
    runner_path = dst_root / PROFILE_RUNNER_NAME
    runner_code = f"""#!/usr/bin/env python3
import os
import sys
import runpy
import torch
from torch.profiler import profile, ProfilerActivity

def main():
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    if pkg_dir not in sys.path:
        sys.path.insert(0, pkg_dir)
    os.chdir(pkg_dir)

    acts = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        acts.append(ProfilerActivity.CUDA)

    with profile(activities=acts, record_shapes=True, with_stack=True, acc_events=True) as prof:
        runpy.run_path(os.path.join(pkg_dir, "{entry_script}"), run_name="__main__")

    prof.export_chrome_trace(os.path.join(pkg_dir, "trace.json"))
    print("Wrote trace.json in:", pkg_dir)

if __name__ == "__main__":
    main()
"""
    runner_path.write_text(runner_code, encoding="utf-8")
    try:
        runner_path.chmod(runner_path.stat().st_mode | 0o111)
    except Exception:
        pass
    return runner_path


def instrument_tree(plan: Path, src_root: Path, dst_root: Path, verbose: bool = True) -> None:
    callsites = _load_plan(plan)

    # 1) exact dedup
    callsites = _dedup_exact(callsites)

    # 2) loop/header containment prune
    callsites = _prune_containing_spans(callsites)

    by_file = _group_by_file(callsites)
    package_root_name = src_root.resolve().name

    _copy_tree(src_root, dst_root)

    changed = 0
    skipped_missing = 0

    for file_str, sites in sorted(by_file.items()):
        dst_file = _resolve_plan_file_to_dst(
            plan_file=Path(file_str),
            src_root=src_root,
            dst_root=dst_root,
            package_root_name=package_root_name,
        )
        if dst_file is None:
            skipped_missing += 1
            if verbose:
                print(f"[SKIP] Plan file not found in copied tree: {file_str}")
            continue

        if dst_file.suffix != ".py":
            continue

        lines = _read_lines(dst_file)
        before = list(lines)

        sites_sorted = sorted(sites, key=lambda c: (c.line_start, c.line_end, c.call_norm, c.label))
        lines = _apply_callsites_to_lines(lines, sites_sorted)

        if lines != before:
            _write_lines(dst_file, lines)
            changed += 1
            if verbose:
                print(f"[OK] Instrumented: {dst_file}  (+{len(sites_sorted)} call-sites)")
        else:
            if verbose:
                print(f"[NOOP] No change: {dst_file}")

    runner = _write_profiler_runner(dst_root, entry_script="train.py")

    print("\nDone.")
    print(f"  Instrumented files: {changed}")
    print(f"  Missing plan files: {skipped_missing}")
    print(f"  Output tree: {dst_root}")
    print(f"  Profiler runner: {runner}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Instrument Python source tree with call-site TB markers.")
    ap.add_argument("--plan", required=True, type=str, help="Path to call-plan JSON exported by cfg_torch_paths.py")
    ap.add_argument("--src", required=True, type=str, help="Source project root directory to copy+instrument")
    ap.add_argument("--dst", type=str, default="", help="Destination directory. Default: ./instrumented_<src_name>")
    ap.add_argument("--quiet", action="store_true", help="Less logging")
    ap.add_argument("--force", action="store_true", help="If destination exists, remove it before instrumenting")
    args = ap.parse_args()

    plan = Path(args.plan)
    src = Path(args.src)

    if not plan.exists():
        raise SystemExit(f"Plan JSON not found: {plan}")
    if not src.exists() or not src.is_dir():
        raise SystemExit(f"Source root must be a directory: {src}")

    if args.dst:
        dst = Path(args.dst)
    else:
        dst = Path.cwd() / f"instrumented_{src.resolve().name}"

    if dst.exists():
        if args.force:
            if dst.is_dir():
                print(f"[FORCE] Removing existing destination: {dst}")
                shutil.rmtree(dst)
            else:
                raise SystemExit(f"--force refused: destination exists and is not a directory: {dst}")
        else:
            raise SystemExit(
                f"Destination already exists: {dst}\n"
                f"Delete it or re-run with --force to overwrite."
            )

    if not args.quiet:
        print("\nInstrumenting project (call-site TBs, loop-pruned):")
        print(f"  Source:      {src.resolve()}")
        print(f"  Destination: {dst.resolve()}")
        print(f"  Plan:        {plan.resolve()}\n")

    instrument_tree(plan=plan, src_root=src, dst_root=dst, verbose=not args.quiet)


if __name__ == "__main__":
    main()