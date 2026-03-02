#!/usr/bin/env python3
"""
join_profile.py (FIXED)

- Uses interval overlap (not just "start inside")
- Normalizes TB names so coverage matches plan keys:
    "TB:train.Block2 torch=..."  ->  "TB:train.Block2"

Usage:
  python join_profile.py --plan torch_boundaries.json --trace trace.json --topk 10 --min-ms 0
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def overlap_us(a0, a1, b0, b1) -> int:
    lo = max(a0, b0)
    hi = min(a1, b1)
    return max(0, hi - lo)


def tb_base(name: str) -> str:
    """Normalize TB label to its base key (strip any suffix like ' torch=...')."""
    s = str(name or "").strip()
    return s.split(" ", 1)[0] if s else s


def extract_regions(trace_events, prefix: str):
    """
    Extract X-phase intervals for events whose name starts with prefix.
    Returns: dict full_name -> list of (start_us, end_us)
    """
    reg = defaultdict(list)
    for e in trace_events:
        if e.get("ph") != "X":
            continue
        name = str(e.get("name", ""))
        if not name.startswith(prefix):
            continue
        ts = int(e.get("ts", 0) or 0)
        dur = int(e.get("dur", 0) or 0)
        if dur <= 0:
            continue
        reg[name].append((ts, ts + dur))
    return reg


def extract_ops(trace_events, op_prefixes=("aten::",)):
    """
    Extract X-phase operator intervals.
    Returns list of (name, start_us, end_us)
    """
    ops = []
    for e in trace_events:
        if e.get("ph") != "X":
            continue
        name = str(e.get("name", ""))
        if not any(name.startswith(p) for p in op_prefixes):
            continue
        ts = int(e.get("ts", 0) or 0)
        dur = int(e.get("dur", 0) or 0)
        if dur <= 0:
            continue
        ops.append((name, ts, ts + dur))
    return ops


def flatten_intervals(regions_dict):
    """Flatten dict[name] -> list[(s,e)] to list[(name,s,e)]"""
    out = []
    for name, spans in regions_dict.items():
        for s, e in spans:
            out.append((name, s, e))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True, help="trace.json from prof.export_chrome_trace")
    ap.add_argument("--plan", default="", help="torch_boundaries.json (optional)")
    ap.add_argument("--topk", type=int, default=10, help="Top aten ops per TB")
    ap.add_argument("--min-ms", type=float, default=0.0, help="Only print TB blocks >= this ms")
    args = ap.parse_args()

    trace = load_json(Path(args.trace))
    events = trace.get("traceEvents", [])

    # TB regions (from trace) keyed by full trace name
    tb_regions_full = extract_regions(events, prefix="TB:")
    tb_spans_full = flatten_intervals(tb_regions_full)

    # Normalize to base key: TB:xxx.BlockY (strip ' torch=...')
    tb_regions_base = defaultdict(list)
    for full_name, spans in tb_regions_full.items():
        base = tb_base(full_name)
        tb_regions_base[base].extend(spans)

    tb_spans_base = flatten_intervals(tb_regions_base)

    # aten ops
    aten_ops = extract_ops(events, op_prefixes=("aten::",))

    print(f"Trace TB events: {sum(len(v) for v in tb_regions_full.values())}")
    print(f"Trace unique TB names (full): {len(tb_regions_full)}")
    print(f"Trace unique TB keys (base): {len(tb_regions_base)}")
    print(f"Trace aten ops: {len(aten_ops)}")
    print()

    # Aggregate per TB base key using overlap
    tb_total = defaultdict(int)  # us
    tb_op_time = defaultdict(lambda: defaultdict(int))  # tb -> op -> us

    # NOTE: O(#tb_spans * #ops). OK for moderate traces; optimize later if needed.
    for tb_name, tb0, tb1 in tb_spans_base:
        for op_name, op0, op1 in aten_ops:
            ov = overlap_us(tb0, tb1, op0, op1)
            if ov > 0:
                tb_total[tb_name] += ov
                tb_op_time[tb_name][op_name] += ov

    ranked = sorted(tb_total.items(), key=lambda x: -x[1])

    print("==============================")
    print("Torch Boundary Time Ranking (overlap-based, aten-only)")
    print("==============================\n")

    printed = 0
    for tb_name, total_us in ranked:
        total_ms = total_us / 1000.0
        if total_ms < args.min_ms:
            continue
        print(f"{total_ms:10.3f} ms  {tb_name}")
        ops = sorted(tb_op_time[tb_name].items(), key=lambda x: -x[1])
        for op, us in ops[: args.topk]:
            print(f"    {us/1000.0:8.3f} ms  {op}")
        print()
        printed += 1

    if printed == 0:
        print("(No TB blocks exceeded min-ms threshold.)\n")

    # Coverage vs plan (optional)
    if args.plan:
        plan = load_json(Path(args.plan))
        plan_keys = [tb_base(f"TB:{x.get('block_key','')}".strip()) for x in plan if isinstance(x, dict)]
        plan_set = set(k for k in plan_keys if k and k != "TB:")
        exec_set = set(tb_regions_base.keys())

        executed = sorted(plan_set & exec_set)
        missing = sorted(plan_set - exec_set)

        print("==============================")
        print("Execution coverage (plan vs trace)")
        print("==============================")
        print(f"Plan TB keys: {len(plan_set)}")
        print(f"Executed TB keys (intersection): {len(executed)}")
        print(f"Missing TB keys (not seen in trace): {len(missing)}\n")

        if executed:
            print("Executed (all):")
            for k in executed:
                print(" ", k)
            print()

        if missing:
            print("Missing (first 30):")
            for k in missing[:30]:
                print(" ", k)
            if len(missing) > 30:
                print(f" ... (+{len(missing)-30} more)")
            print()


if __name__ == "__main__":
    main()