#!/usr/bin/env python3
"""
join_profile.py (timeline mode + containment-based op ownership)

Modes:
- default: ranking by TB wall time (sum of TB durations)
- --timeline: print TB spans in chronological order, with contained ops

Ownership rule:
- An op belongs to a TB span if tb0 <= op0 and op1 <= tb1
- If multiple TB spans contain the op (nested TBs), assign to the smallest span
  (most specific) to avoid double counting.

Usage examples:
  python join_profile.py --trace trace.json --timeline
  python join_profile.py --trace trace.json --timeline --min-ms 0.1 --topk 5
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def tb_base(name: str) -> str:
    """Normalize TB label to its base key (strip suffix like ' torch=...')."""
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


@dataclass(frozen=True)
class TBSpan:
    tb_name: str     # base TB name
    start_us: int
    end_us: int

    @property
    def dur_us(self) -> int:
        return self.end_us - self.start_us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True, help="trace.json from prof.export_chrome_trace")
    ap.add_argument("--plan", default="", help="torch_boundaries.json (optional, for coverage)")
    ap.add_argument("--topk", type=int, default=10, help="Top ops to print per TB span / TB key")
    ap.add_argument("--min-ms", type=float, default=0.0, help="Filter out TB spans/keys < this ms")
    ap.add_argument("--timeline", action="store_true", help="Print TB spans in time order (timeline)")
    ap.add_argument("--relative", action="store_true",
                    help="In timeline mode, print times relative to first TB start")
    args = ap.parse_args()

    trace = load_json(Path(args.trace))
    events = trace.get("traceEvents", [])

    # TB regions keyed by full trace name
    tb_regions_full = extract_regions(events, prefix="TB:")

    # Normalize to base key and flatten into explicit spans
    tb_spans: list[TBSpan] = []
    for full_name, spans in tb_regions_full.items():
        base = tb_base(full_name)
        for s, e in spans:
            tb_spans.append(TBSpan(base, s, e))

    # ops
    aten_ops = extract_ops(events, op_prefixes=("aten::",))

    print(f"Trace TB events: {sum(len(v) for v in tb_regions_full.values())}")
    print(f"Trace unique TB names (full): {len(tb_regions_full)}")
    print(f"Trace unique TB keys (base): {len(set(x.tb_name for x in tb_spans))}")
    print(f"Trace aten ops: {len(aten_ops)}")
    print()

    if not tb_spans:
        print("No TB spans found (no 'TB:' X-phase events).")
        return

    # ---- Assign each op to the smallest containing TB span (nested-safe) ----
    # Map TBSpan -> list of ops (name, start_us, end_us)
    ops_by_span: dict[TBSpan, list[tuple[str, int, int]]] = defaultdict(list)

    # Pre-sort TB spans by span length ascending to find "smallest containing" quickly.
    spans_by_len = sorted(tb_spans, key=lambda s: (s.dur_us, s.start_us, s.end_us, s.tb_name))

    for op_name, op0, op1 in aten_ops:
        if op1 <= op0:
            continue
        assigned = None
        for sp in spans_by_len:
            if sp.start_us <= op0 and op1 <= sp.end_us:
                assigned = sp
                break  # first match is smallest due to spans_by_len sorting
        if assigned is not None:
            ops_by_span[assigned].append((op_name, op0, op1))

    # Timeline reference
    t0 = min(s.start_us for s in tb_spans)

    def fmt_us(us: int) -> str:
        if args.relative:
            us = us - t0
        return f"{us/1000.0:10.3f}ms"

    # ---- Timeline mode ----
    if args.timeline:
        print("==============================")
        print("Timeline (TB spans + contained aten ops)")
        print("==============================")
        if args.relative:
            print(f"(Times are relative to first TB start: {t0} us)\n")
        else:
            print("(Times are absolute trace timestamps in us, shown as ms)\n")

        # Sort by start time
        spans_by_time = sorted(tb_spans, key=lambda s: (s.start_us, s.end_us, s.tb_name))

        shown = 0
        for sp in spans_by_time:
            dur_ms = sp.dur_us / 1000.0
            if dur_ms < args.min_ms:
                continue

            print(f"{fmt_us(sp.start_us)} → {fmt_us(sp.end_us)}  ({dur_ms:8.3f} ms)  {sp.tb_name}")

            # Ops inside this span (already assigned nested-safe)
            ops = ops_by_span.get(sp, [])
            # Sort ops by start time (timeline-ish)
            ops.sort(key=lambda x: (x[1], x[2], x[0]))

            # Optionally limit output volume
            if args.topk > 0:
                ops_to_show = ops[: args.topk]
            else:
                ops_to_show = ops

            for op_name, op0, op1 in ops_to_show:
                rel0 = op0 - sp.start_us
                rel1 = op1 - sp.start_us
                print(
                    f"    +{rel0/1000.0:8.3f}ms → +{rel1/1000.0:8.3f}ms"
                    f"  ({(op1-op0)/1000.0:8.3f} ms)  {op_name}"
                )

            if ops and args.topk > 0 and len(ops) > args.topk:
                print(f"    ... (+{len(ops)-args.topk} more ops)")
            print()
            shown += 1

        if shown == 0:
            print("(No TB spans exceeded min-ms threshold.)\n")

    # ---- Default mode: ranking by TB wall time (sum durations per TB key) ----
    else:
        tb_total_wall = defaultdict(int)  # us
        tb_total_aten = defaultdict(int)  # us (assigned full durations)
        tb_op_time = defaultdict(lambda: defaultdict(int))

        for sp in tb_spans:
            tb_total_wall[sp.tb_name] += sp.dur_us
            for op_name, op0, op1 in ops_by_span.get(sp, []):
                d = op1 - op0
                tb_total_aten[sp.tb_name] += d
                tb_op_time[sp.tb_name][op_name] += d

        ranked = sorted(tb_total_wall.items(), key=lambda x: -x[1])

        print("==============================")
        print("Torch Boundary Time Ranking (TB wall time; ops assigned by containment)")
        print("==============================\n")

        printed = 0
        for tb_name, wall_us in ranked:
            wall_ms = wall_us / 1000.0
            if wall_ms < args.min_ms:
                continue

            aten_us = tb_total_aten.get(tb_name, 0)
            cov = (aten_us / wall_us * 100.0) if wall_us > 0 else 0.0
            print(f"{wall_ms:10.3f} ms  {tb_name}")
            print(f"    aten assigned: {aten_us/1000.0:8.3f} ms  ({cov:5.1f}%)")

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
        exec_set = set(s.tb_name for s in tb_spans)

        executed = sorted(plan_set & exec_set)
        missing = sorted(plan_set - exec_set)

        print("==============================")
        print("Execution coverage (plan vs trace)")
        print("==============================")
        print(f"Plan TB keys: {len(plan_set)}")
        print(f"Executed TB keys (intersection): {len(executed)}")
        print(f"Missing TB keys (not seen in trace): {len(missing)}\n")


if __name__ == "__main__":
    main()