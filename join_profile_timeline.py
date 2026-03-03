#!/usr/bin/env python3
"""
join_profile_timeline.py

Outputs:
  1) Human-readable timelines to stdout (t0 treated as 0)
     - TB timeline (TB spans + owned CPU ops)
     - Memory transfers timeline (global)
     - Combined timeline (TB + transfers [+ kernels]) sorted by time
  2) timeline.json (structured timeline data)

Extraction (Chrome trace X events with dur>0):
- TB spans:   name startswith "TB:"
- CPU ops:    name startswith "aten::" (configurable via --op-prefix)
- Transfers:  name matches memcpy patterns (prefix/contains/category filters)
- Kernels:    optional via --kernels, cat in --kernel-cat (default: "kernel")

Ownership rules (ONLY for ops under TB):
- STRICT CONTAINMENT:
    tb_start <= op_start and op_end <= tb_end
- pid/tid must match (same track)
- Nested TBs: smallest containing TB wins

Transfers/kernels:
- Not owned by TBs (additional info)
- Still shown in combined timeline.

Usage:
  python join_profile_timeline.py --trace trace.json
  python join_profile_timeline.py --trace trace.json --kernels
  python join_profile_timeline.py --trace trace.json --topk 0
  python join_profile_timeline.py --trace trace.json --min-ms 0.2 --topk 10
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from typing import Iterable, Optional


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def tb_base(name: str) -> str:
    """Normalize TB label to its base key (strip suffix like ' torch=...')."""
    s = str(name or "").strip()
    return s.split(" ", 1)[0] if s else s


def _safe_int(x, default=0) -> int:
    try:
        return int(x or default)
    except Exception:
        return default


def _safe_str(x) -> str:
    return str(x) if x is not None else ""


@dataclass(frozen=True)
class TBSpan:
    tb_name: str
    start_us: int
    end_us: int
    cat: str
    pid: int
    tid: int

    @property
    def dur_us(self) -> int:
        return self.end_us - self.start_us

    @property
    def is_gpu_tb(self) -> bool:
        return "gpu" in (self.cat or "").lower()


@dataclass(frozen=True)
class IntervalEvt:
    kind: str  # "op" | "xfer" | "kernel" | "tb"
    name: str
    start_us: int
    end_us: int
    cat: str
    pid: int
    tid: int

    @property
    def dur_us(self) -> int:
        return self.end_us - self.start_us


def extract_tb_spans(trace_events) -> list[TBSpan]:
    spans: list[TBSpan] = []
    for e in trace_events:
        if e.get("ph") != "X":
            continue
        name = _safe_str(e.get("name", ""))
        if not name.startswith("TB:"):
            continue
        ts = _safe_int(e.get("ts", 0))
        dur = _safe_int(e.get("dur", 0))
        if dur <= 0:
            continue
        spans.append(
            TBSpan(
                tb_name=tb_base(name),
                start_us=ts,
                end_us=ts + dur,
                cat=_safe_str(e.get("cat", "")),
                pid=_safe_int(e.get("pid", -1), default=-1),
                tid=_safe_int(e.get("tid", -1), default=-1),
            )
        )
    return spans


def extract_xphase_intervals(
    trace_events,
    *,
    kind: str,
    startswith_prefixes: tuple[str, ...] = (),
    contains_tokens: tuple[str, ...] = (),
    cat_allow: tuple[str, ...] = (),
) -> list[IntervalEvt]:
    """
    Extract X-phase intervals with dur>0.

    Matching:
      - If startswith_prefixes provided: accept if name startswith any prefix
      - If contains_tokens provided: accept if any token is substring of name
      - If both provided: accept if either matches
      - If cat_allow provided: cat must match exactly one of them
      - If cat_allow is used alone: accept all names within that cat
    """
    if not startswith_prefixes and not contains_tokens and not cat_allow:
        return []

    out: list[IntervalEvt] = []
    for e in trace_events:
        if e.get("ph") != "X":
            continue
        ts = _safe_int(e.get("ts", 0))
        dur = _safe_int(e.get("dur", 0))
        if dur <= 0:
            continue

        name = _safe_str(e.get("name", ""))
        cat = _safe_str(e.get("cat", ""))

        if cat_allow and cat not in cat_allow:
            continue

        if startswith_prefixes or contains_tokens:
            ok = False
            if startswith_prefixes and any(name.startswith(p) for p in startswith_prefixes):
                ok = True
            if (not ok) and contains_tokens and any(tok in name for tok in contains_tokens):
                ok = True
            if not ok:
                continue

        out.append(
            IntervalEvt(
                kind=kind,
                name=name,
                start_us=ts,
                end_us=ts + dur,
                cat=cat,
                pid=_safe_int(e.get("pid", -1), default=-1),
                tid=_safe_int(e.get("tid", -1), default=-1),
            )
        )
    return out


def assign_ops_to_smallest_containing_tb(
    tb_spans: list[TBSpan],
    ops: Iterable[IntervalEvt],
) -> dict[TBSpan, list[IntervalEvt]]:
    """
    Assign each op to the SMALLEST containing TB span (nested-safe).
    STRICT containment + pid/tid match.
    """
    by_span: dict[TBSpan, list[IntervalEvt]] = defaultdict(list)
    spans_by_len = sorted(tb_spans, key=lambda s: (s.dur_us, s.start_us, s.end_us, s.tb_name, s.pid, s.tid))

    for op in ops:
        if op.kind != "op" or op.end_us <= op.start_us:
            continue
        assigned: Optional[TBSpan] = None
        for sp in spans_by_len:
            if sp.pid != op.pid or sp.tid != op.tid:
                continue
            if sp.start_us <= op.start_us and op.end_us <= sp.end_us:
                assigned = sp
                break
        if assigned is not None:
            by_span[assigned].append(op)

    return by_span


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True, help="trace.json from prof.export_chrome_trace")
    ap.add_argument("--out", default="timeline.json", help="Output json path (default: timeline.json)")
    ap.add_argument("--topk", type=int, default=10, help="Max ops to print per TB span (0=all)")
    ap.add_argument("--min-ms", type=float, default=0.0, help="Filter out TB spans < this ms (printing + json)")

    # CPU ops
    ap.add_argument("--op-prefix", action="append", default=["aten::"],
                    help="Operator prefix to include (repeatable). Default: aten::")

    # Transfers (memcpy)
    ap.add_argument("--xfer-prefix", action="append", default=["cudaMemcpy", "Memcpy", "memcpy"],
                    help="Transfer name prefix to include (repeatable). Default: cudaMemcpy, Memcpy, memcpy")
    ap.add_argument("--xfer-contains", action="append", default=["HtoD", "DtoH", "DtoD"],
                    help="Transfer name substring to include (repeatable). Default: HtoD, DtoH, DtoD")
    ap.add_argument("--xfer-cat", action="append", default=[],
                    help="(Optional) Restrict transfers to these exact categories (repeatable). Default: no restriction")

    # Kernels
    ap.add_argument("--kernels", action="store_true", help="Also extract GPU kernels (cat match)")
    ap.add_argument("--kernel-cat", action="append", default=["kernel"],
                    help="Kernel category to include (repeatable). Default: kernel")

    # Combined timeline printing controls
    ap.add_argument("--combined-topk", type=int, default=0,
                    help="Max rows to print in combined timeline (0=all). Default: 0")

    args = ap.parse_args()

    trace_path = Path(args.trace)
    out_path = Path(args.out)

    trace = load_json(trace_path)
    events = trace.get("traceEvents", [])

    tb_spans = extract_tb_spans(events)
    if not tb_spans:
        print("No TB spans found (no 'TB:' X-phase events with dur>0).")
        write_json(out_path, {"meta": {"trace": str(trace_path)}, "t0_us": None, "tb_spans": [], "transfers": [], "kernels": [], "combined_timeline": []})
        print(f"Wrote {out_path}")
        return

    # t0 (best effort: earliest positive ts in trace)
    ts_candidates = []
    for e in events:
        if "ts" not in e:
            continue
        ts = _safe_int(e.get("ts", 0))
        if ts > 0:
            ts_candidates.append(ts)
    t0 = min(ts_candidates) if ts_candidates else min(s.start_us for s in tb_spans)

    def ms(us: int) -> float:
        return (us - t0) / 1000.0

    # Filter TB spans
    spans_by_time = sorted(tb_spans, key=lambda s: (s.start_us, s.end_us, s.tb_name, s.pid, s.tid))
    filtered_spans = [sp for sp in spans_by_time if (sp.dur_us / 1000.0) >= args.min_ms]

    # Extract intervals
    op_prefixes = tuple(args.op_prefix)
    cpu_ops = extract_xphase_intervals(events, kind="op", startswith_prefixes=op_prefixes)

    xfer_prefixes = tuple(args.xfer_prefix)
    xfer_contains = tuple(args.xfer_contains)
    xfer_cats = tuple(args.xfer_cat) if args.xfer_cat else ()
    transfers = extract_xphase_intervals(
        events, kind="xfer",
        startswith_prefixes=xfer_prefixes,
        contains_tokens=xfer_contains,
        cat_allow=xfer_cats,
    )

    kernels: list[IntervalEvt] = []
    kernel_cats = tuple(args.kernel_cat)
    if args.kernels:
        kernels = extract_xphase_intervals(events, kind="kernel", cat_allow=kernel_cats)

    # Assign ops to TBs
    ops_by_span = assign_ops_to_smallest_containing_tb(tb_spans, cpu_ops)

    # ---- Print TB timeline ----
    print("==============================")
    print("TB Timeline (program start treated as 0)")
    print("==============================")
    print(f"Trace: {trace_path}")
    print(f"t0: {t0} us (all timestamps shown as ms since t0)")
    print(f"TB spans: {len(tb_spans)} (printed: {len(filtered_spans)}; min-ms={args.min_ms})")
    print(f"CPU ops: {len(cpu_ops)} (prefixes={list(op_prefixes)})")
    print(f"Transfers: {len(transfers)} (global)")
    if args.kernels:
        print(f"Kernels: {len(kernels)} (global)")
    print()

    for sp in filtered_spans:
        dur_ms = sp.dur_us / 1000.0
        tag = "GPU" if sp.is_gpu_tb else "CPU"
        print(
            f"{ms(sp.start_us):10.3f}ms → {ms(sp.end_us):10.3f}ms  "
            f"({dur_ms:8.3f} ms)  [{tag}] {sp.tb_name}  (cat={sp.cat}, pid={sp.pid}, tid={sp.tid})"
        )

        ops_in = sorted(ops_by_span.get(sp, []), key=lambda x: (x.start_us, x.end_us, x.name))
        to_show = ops_in if args.topk == 0 else ops_in[: args.topk]
        for op in to_show:
            print(
                f"    [OP ] {ms(op.start_us):10.3f}ms → {ms(op.end_us):10.3f}ms  "
                f"({op.dur_us/1000.0:8.3f} ms)  {op.name}"
            )
        if args.topk != 0 and len(ops_in) > args.topk:
            print(f"    ... (+{len(ops_in)-args.topk} more ops)")
        print()

    # ---- Print transfers timeline ----
    print("==============================")
    print("Memory transfers timeline (global)")
    print("==============================")
    transfers_sorted = sorted(transfers, key=lambda e: (e.start_us, e.end_us, e.name))
    for ev in transfers_sorted:
        print(
            f"{ms(ev.start_us):10.3f}ms → {ms(ev.end_us):10.3f}ms  "
            f"({ev.dur_us/1000.0:8.3f} ms)  {ev.name}  (cat={ev.cat}, pid={ev.pid}, tid={ev.tid})"
        )
    print()

    # ---- Combined timeline (main execution + memory) ----
    # "Main execution" here = TB spans (as blocks) + transfers (+ kernels).
    combined: list[IntervalEvt] = []

    for sp in filtered_spans:
        combined.append(
            IntervalEvt(
                kind="tb",
                name=sp.tb_name,
                start_us=sp.start_us,
                end_us=sp.end_us,
                cat=sp.cat,
                pid=sp.pid,
                tid=sp.tid,
            )
        )
    combined.extend(transfers)
    combined.extend(kernels)

    combined.sort(key=lambda e: (e.start_us, e.end_us, e.kind, e.name))

    def _kind_label(ev: IntervalEvt) -> str:
        if ev.kind == "tb":
            return "TB  "
        if ev.kind == "op":
            return "OP  "
        if ev.kind == "xfer":
            return "XFER"
        if ev.kind == "kernel":
            return "KERN"
        return ev.kind.upper()[:4]

    print("==============================")
    print("Combined timeline (TB + transfers" + (" + kernels" if args.kernels else "") + ")")
    print("==============================")
    to_print = combined if args.combined_topk == 0 else combined[: args.combined_topk]
    for ev in to_print:
        extra = ""
        if ev.kind == "tb":
            tag = "GPU" if ("gpu" in (ev.cat or "").lower()) else "CPU"
            extra = f"  [{tag}]"
        print(
            f"{ms(ev.start_us):10.3f}ms → {ms(ev.end_us):10.3f}ms  "
            f"({ev.dur_us/1000.0:8.3f} ms)  [{_kind_label(ev)}]{extra} {ev.name}  "
            f"(cat={ev.cat}, pid={ev.pid}, tid={ev.tid})"
        )
    if args.combined_topk != 0 and len(combined) > args.combined_topk:
        print(f"... (+{len(combined)-args.combined_topk} more rows)")
    print()

    # ---- Build timeline.json ----
    def ev_to_json(ev: IntervalEvt):
        return {
            "kind": ev.kind,
            "name": ev.name,
            "cat": ev.cat,
            "pid": ev.pid,
            "tid": ev.tid,
            "start_us": ev.start_us,
            "end_us": ev.end_us,
            "start_ms": ms(ev.start_us),
            "end_ms": ms(ev.end_us),
            "dur_us": ev.dur_us,
            "dur_ms": ev.dur_us / 1000.0,
        }

    tb_spans_json = []
    for sp in filtered_spans:
        ops_in = sorted(ops_by_span.get(sp, []), key=lambda x: (x.start_us, x.end_us, x.name))
        tb_spans_json.append(
            {
                "tb_name": sp.tb_name,
                "tb_cat": sp.cat,
                "tb_pid": sp.pid,
                "tb_tid": sp.tid,
                "is_gpu_tb": sp.is_gpu_tb,
                "start_us": sp.start_us,
                "end_us": sp.end_us,
                "start_ms": ms(sp.start_us),
                "end_ms": ms(sp.end_us),
                "dur_us": sp.dur_us,
                "dur_ms": sp.dur_us / 1000.0,
                "ops": [ev_to_json(ev) for ev in ops_in],
            }
        )

    timeline_obj = {
        "meta": {
            "trace": str(trace_path),
            "t0_definition": "min(ts) across traceEvents (best-effort); all *_ms are relative to t0",
            "min_ms_filter": args.min_ms,
            "print_topk_ops_per_tb": args.topk,
            "op_prefixes": list(op_prefixes),
            "xfer_prefixes": list(xfer_prefixes),
            "xfer_contains": list(xfer_contains),
            "xfer_cats": list(xfer_cats) if xfer_cats else [],
            "kernels_enabled": bool(args.kernels),
            "kernel_cats": list(kernel_cats),
            "ownership": "ops only: strict containment + pid/tid match; transfers/kernels are global (additional info)",
        },
        "t0_us": t0,
        "tb_spans": tb_spans_json,
        "transfers": [ev_to_json(ev) for ev in transfers_sorted],
        "kernels": [ev_to_json(ev) for ev in sorted(kernels, key=lambda e: (e.start_us, e.end_us, e.name))] if args.kernels else [],
        "combined_timeline": [ev_to_json(ev) for ev in combined],
    }

    write_json(out_path, timeline_obj)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()