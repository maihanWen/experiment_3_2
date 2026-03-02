#!/usr/bin/env python3
"""
instrument.py (UPDATED v2 - fixes NameError torch + avoids line shifts)

Key fix:
  - Injects TB markers as:
        with __import__("torch").profiler.record_function("TB:..."):
    so it does NOT require `torch` or `record_function` to be imported in the file.

This avoids:
  - NameError: record_function is not defined
  - NameError: torch is not defined
  - Line-number shifts caused by inserting imports at top

Also:
  - Copies --src directory to --dst (default: ./instrumented_<src_name>)
  - Instruments .py files referenced by plan (file + line_start/end)
  - Writes <dst_root>/run_profile.py automatically to produce trace.json

Usage:
  python instrument.py --plan torch_boundaries.json --src /path/to/toy_transformer_pkg
  cd ./instrumented_toy_transformer_pkg
  python run_profile.py
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


PROFILE_RUNNER_NAME = "run_profile.py"


@dataclass(frozen=True)
class Boundary:
    block_key: str
    file: str
    line_start: int
    line_end: int
    torch_calls: list[str]


def _load_plan(plan_path: Path) -> list[Boundary]:
    data = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Plan JSON must be a list of objects.")
    out: list[Boundary] = []
    for obj in data:
        if not isinstance(obj, dict):
            continue
        f = obj.get("file")
        ls = obj.get("line_start")
        le = obj.get("line_end")
        bk = obj.get("block_key") or obj.get("key") or ""
        tc = obj.get("torch_calls") or []
        if not f or not bk or not isinstance(ls, int) or ls <= 0:
            continue
        if not isinstance(le, int) or le <= 0:
            le = ls
        if le < ls:
            ls, le = le, ls
        out.append(
            Boundary(
                block_key=str(bk),
                file=str(f),
                line_start=ls,
                line_end=le,
                torch_calls=[str(x) for x in tc],
            )
        )
    return out


def _read_lines(p: Path) -> list[str]:
    return p.read_text(encoding="utf-8").splitlines()


def _write_lines(p: Path, lines: list[str]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _indent_of(line: str) -> str:
    return line[: len(line) - len(line.lstrip(" \t"))]


def _make_label(b: Boundary) -> str:
    calls = "|".join(b.torch_calls[:6])
    if len(b.torch_calls) > 6:
        calls += "|..."
    calls = calls.replace("\n", " ").replace("\r", " ")
    return f"TB:{b.block_key} torch={calls}" if calls else f"TB:{b.block_key}"


def _line_indent_len(s: str) -> int:
    return len(s) - len(s.lstrip(" \t"))


def _is_def_or_class_line(s: str) -> bool:
    st = s.lstrip()
    return st.startswith("def ") or st.startswith("class ")


def _is_noncode_line(s: str) -> bool:
    st = s.strip()
    return st == "" or st.startswith("#")


def _apply_boundaries_to_lines(lines: list[str], boundaries: list[Boundary]) -> list[str]:
    """
    Wrap each boundary range (1-based line numbers) with:

        <indent>with __import__("torch").profiler.record_function("TB:..."):
            <original block>

    Fixes:
      - Never wraps a 'def'/'class' header directly (wraps inside the body).
      - Clamps range to stop before indentation dedents (prevents corrupting following defs).
      - Keeps overlapping/nested ranges (no overlap-dropping).
    """
    if not boundaries:
        return lines

    # De-dup exact duplicates
    uniq = {}
    for b in boundaries:
        ls = int(b.line_start)
        le = int(b.line_end)
        if le < ls:
            ls, le = le, ls
        if ls < 1:
            continue
        uniq[(ls, le, b.block_key)] = Boundary(
            block_key=b.block_key,
            file=b.file,
            line_start=ls,
            line_end=le,
            torch_calls=b.torch_calls,
        )

    b_sorted = sorted(uniq.values(), key=lambda b: (b.line_start, b.line_end, b.block_key))
    out = list(lines)

    # Apply bottom->top so earlier insertions don't shift later indices.
    # Track insertion points and offset upcoming boundaries so nested parents
    # still cover their intended end lines after child wrappers add lines.
    inserted_positions: list[int] = []
    for b in reversed(b_sorted):
        ls = b.line_start
        le = b.line_end

        if inserted_positions:
            ls += sum(1 for pos in inserted_positions if pos <= ls)
            le += sum(1 for pos in inserted_positions if pos <= le)

        le = min(le, len(out))
        if ls > len(out) or le < ls:
            continue

        # If boundary starts on a def/class header, move to the first body statement
        if _is_def_or_class_line(out[ls - 1]):
            i = ls  # 0-based: ls is next line
            while i < len(out) and _is_noncode_line(out[i]):
                i += 1
            if i >= len(out):
                continue
            ls = i + 1  # back to 1-based
            if le < ls:
                continue

        # Determine base indentation from the first non-blank/non-comment line in [ls..le]
        base_i = ls - 1
        while base_i <= le - 1 and _is_noncode_line(out[base_i]):
            base_i += 1
        if base_i > le - 1:
            continue

        base_indent_len = _line_indent_len(out[base_i])
        base_indent = out[base_i][:base_indent_len]

        # Clamp le so we do NOT cross a dedent boundary
        # (i.e., do not include lines with indent < base_indent_len)
        clamp_le = le
        for j in range(base_i, le):
            ln = out[j]
            if _is_noncode_line(ln):
                continue
            if _line_indent_len(ln) < base_indent_len:
                clamp_le = j  # stop BEFORE this line
                break
        le = clamp_le
        if le < ls:
            continue

        label = _make_label(b)
        with_line = f'{base_indent}with __import__("torch").profiler.record_function("{label}"):'

        # Re-indent wrapped lines by one extra level relative to base_indent.
        new_block: list[str] = []
        for j in range(ls - 1, le):
            ln = out[j]
            if ln.strip() == "":
                new_block.append(base_indent + "    ")
                continue

            # If the line is less-indented than base_indent (shouldn't happen after clamp),
            # keep it as-is but still indent to avoid syntax break.
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


def _group_by_file(boundaries: list[Boundary]) -> dict[str, list[Boundary]]:
    by_file: dict[str, list[Boundary]] = {}
    for b in boundaries:
        by_file.setdefault(b.file, []).append(b)
    return by_file


def _resolve_plan_file_to_dst(
    plan_file: Path,
    src_root: Path,
    dst_root: Path,
    package_root_name: str,
) -> Path | None:
    """
    Map a plan file path to the instrumented copy.

    Strategy:
      1) If plan_file is under src_root, use that relative path.
      2) Else, locate 'package_root_name' in the plan_file path and use suffix after it.
         Example: /old/.../toy_transformer_pkg/data/x.py -> data/x.py
      3) Else, fallback to dst_root / basename.
    """
    plan_file = plan_file.resolve()
    src_root = src_root.resolve()

    # 1) direct relative
    try:
        rel = plan_file.relative_to(src_root)
        cand = dst_root / rel
        return cand if cand.exists() else None
    except Exception:
        pass

    # 2) suffix after package root name
    parts = list(plan_file.parts)
    if package_root_name in parts:
        i = parts.index(package_root_name)
        suffix = Path(*parts[i + 1 :])
        cand = dst_root / suffix
        if cand.exists():
            return cand

    # 3) basename fallback
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
import types
import torch
from torch.profiler import profile, ProfilerActivity

def main():
    pkg_dir = os.path.dirname(os.path.abspath(__file__))  # .../instrumented_toy_transformer_pkg

    # --- CRITICAL: create an alias package named "toy_transformer_pkg"
    # so imports like `from toy_transformer_pkg.data import ...` work,
    # even though the directory name is instrumented_toy_transformer_pkg.
    pkg_name = "toy_transformer_pkg"
    if pkg_name not in sys.modules:
        m = types.ModuleType(pkg_name)
        m.__path__ = [pkg_dir]  # treat this folder as the package root
        m.__file__ = os.path.join(pkg_dir, "__init__.py")
        sys.modules[pkg_name] = m

    # Also ensure current dir is importable (helps some import mechanisms)
    if pkg_dir not in sys.path:
        sys.path.insert(0, pkg_dir)

    # Keep cwd inside package so relative file access works
    os.chdir(pkg_dir)

    acts = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        acts.append(ProfilerActivity.CUDA)

    with profile(activities=acts, record_shapes=True, with_stack=True, acc_events=True) as prof:
        runpy.run_path(os.path.join(pkg_dir, "train.py"), run_name="__main__")

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
    boundaries = _load_plan(plan)
    by_file = _group_by_file(boundaries)

    package_root_name = src_root.resolve().name

    _copy_tree(src_root, dst_root)

    changed = 0
    skipped_missing = 0

    for file_str, bs in sorted(by_file.items()):
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

        usable = [b for b in bs if isinstance(b.line_start, int) and isinstance(b.line_end, int)]
        if not usable:
            continue

        lines = _read_lines(dst_file)
        before = list(lines)

        lines = _apply_boundaries_to_lines(lines, usable)

        if lines != before:
            _write_lines(dst_file, lines)
            changed += 1
            if verbose:
                print(f"[OK] Instrumented: {dst_file}  (+{len(usable)} ranges)")
        else:
            if verbose:
                print(f"[NOOP] No change: {dst_file}")

    runner = _write_profiler_runner(dst_root, entry_script="train.py")

    print(f"\nDone.")
    print(f"  Instrumented files: {changed}")
    print(f"  Missing plan files: {skipped_missing}")
    print(f"  Output tree: {dst_root}")
    print(f"  Profiler runner: {runner}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Instrument Python source tree with TB markers (and add run_profile.py)."
    )
    ap.add_argument("--plan", required=True, type=str, help="Path to torch_boundaries.json exported by analyzer")
    ap.add_argument("--src", required=True, type=str, help="Source project root directory to copy+instrument")
    ap.add_argument(
        "--dst",
        type=str,
        default="",
        help="Optional destination directory. Default: ./instrumented_<src_name> (in current directory)",
    )
    ap.add_argument("--quiet", action="store_true", help="Less logging")
    ap.add_argument(
        "--force",
        action="store_true",
        help="If destination exists, remove it before instrumenting",
    )
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
        src_name = src.resolve().name
        dst = Path.cwd() / f"instrumented_{src_name}"

    if dst.exists():
        if args.force:
            # Safety: only delete directories, never files
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
        print("\nInstrumenting project:")
        print(f"  Source:      {src.resolve()}")
        print(f"  Destination: {dst.resolve()}")
        print(f"  Plan:        {plan.resolve()}\n")

    instrument_tree(plan=plan, src_root=src, dst_root=dst, verbose=not args.quiet)


if __name__ == "__main__":
    main()