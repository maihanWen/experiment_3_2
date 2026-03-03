#!/usr/bin/env python3
"""
join_profile_timeline.py

Flow goal:
- TB (CPU) contains:
    - CPU ops (aten::...) [strict ownership]
    - GPU kernels (optional): mapped to CPU ops (heuristic) + also listed at TB-level
    - GPU transfers: NOT mapped to CPU ops; only direction (HtoD/DtoH/DtoD/...)
      Transfers are associated to TB by overlap or TB-end grace.

Changes for your request:
- Show ALL ops (no "... more").
- Show ALL kernels under each op (no cap).
- Show ALL transfers under each TB (no cap).
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
    s = str(name or "").strip()
    return s.split(" ", 1)[0] if s else s


def _safe_int(x, default=0) -> int:
    try:
        return int(x or default)
    except Exception:
        return default


def _safe_float(x, default=0.0) -> float:
    try:
        return float(x if x is not None else default)
    except Exception:
        return float(default)


def _safe_str(x) -> str:
    return str(x) if x is not None else ""


@dataclass(frozen=True)
class TBSpan:
    tb_name: str
    start_us: float
    end_us: float
    cat: str
    pid: int
    tid: int

    @property
    def dur_us(self) -> float:
        return self.end_us - self.start_us

    @property
    def is_gpu_tb(self) -> bool:
        return "gpu" in (self.cat or "").lower()


@dataclass(frozen=True)
class IntervalEvt:
    kind: str  # "op" | "xfer" | "kernel" | "tb"
    name: str
    start_us: float
    end_us: float
    cat: str
    pid: int
    tid: int

    @property
    def dur_us(self) -> float:
        return self.end_us - self.start_us


def _overlap_us(a0: float, a1: float, b0: float, b1: float) -> float:
    lo = max(a0, b0)
    hi = min(a1, b1)
    return max(0.0, hi - lo)


def classify_kernel(name: str) -> str:
    n = (name or "").lower()
    if "gemm" in n or "sgemm" in n or "matmul" in n or "mma" in n:
        return "GEMM/MatMul"
    if "conv" in n:
        return "Convolution"
    if "softmax" in n:
        return "Softmax"
    if "layernorm" in n or "layer_norm" in n or "rmsnorm" in n:
        return "Norm"
    if "attention" in n or "flash" in n:
        return "Attention"
    if "loss" in n or "nll_loss" in n or "cross_entropy" in n:
        return "Loss"
    if "reduce" in n or "reduction" in n:
        return "Reduction"
    if "relu" in n or "gelu" in n or "silu" in n or "swish" in n:
        return "Activation"
    if "index" in n or "gather" in n or "scatter" in n:
        return "Indexing"
    if "copy" in n or "memcpy" in n:
        return "Copy"
    return "Other"


def xfer_direction(name: str, cat: str = "") -> str:
    s = f"{name} {cat}".lower()
    if "htod" in s or "h2d" in s:
        return "HtoD"
    if "dtoh" in s or "d2h" in s:
        return "DtoH"
    if "dtod" in s or "d2d" in s:
        return "DtoD"
    return "Unknown"


def extract_tb_spans(trace_events, *, keep_gpu_tb: bool = False) -> list[TBSpan]:
    spans: list[TBSpan] = []
    for e in trace_events:
        if e.get("ph") != "X":
            continue
        name = _safe_str(e.get("name", ""))
        if not name.startswith("TB:"):
            continue
        ts = _safe_float(e.get("ts", 0.0))
        dur = _safe_float(e.get("dur", 0.0))
        if dur <= 0:
            continue

        cat = _safe_str(e.get("cat", ""))
        cat_l = cat.lower()
        if not keep_gpu_tb:
            if cat_l.startswith("gpu_") or cat_l == "gpu_user_annotation":
                continue

        spans.append(
            TBSpan(
                tb_name=tb_base(name),
                start_us=ts,
                end_us=ts + dur,
                cat=cat,
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
    if not startswith_prefixes and not contains_tokens and not cat_allow:
        return []

    out: list[IntervalEvt] = []
    for e in trace_events:
        if e.get("ph") != "X":
            continue
        ts = _safe_float(e.get("ts", 0.0))
        dur = _safe_float(e.get("dur", 0.0))
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


def associate_gpu_events_to_tb(
    tb_spans: list[TBSpan],
    kernels: list[IntervalEvt],
    transfers: list[IntervalEvt],
    *,
    grace_us: float,
) -> tuple[dict[TBSpan, list[IntervalEvt]], dict[TBSpan, list[IntervalEvt]]]:
    kernels_sorted = sorted(kernels, key=lambda e: (e.start_us, e.end_us, e.name))
    xfers_sorted = sorted(transfers, key=lambda e: (e.start_us, e.end_us, e.name))

    tb_to_kernels: dict[TBSpan, list[IntervalEvt]] = {}
    tb_to_xfers: dict[TBSpan, list[IntervalEvt]] = {}

    for tb in tb_spans:
        ks: list[IntervalEvt] = []
        xs: list[IntervalEvt] = []
        tb0, tb1 = tb.start_us, tb.end_us

        for k in kernels_sorted:
            if _overlap_us(tb0, tb1, k.start_us, k.end_us) > 0 or (tb1 <= k.start_us <= tb1 + grace_us):
                ks.append(k)

        for x in xfers_sorted:
            if _overlap_us(tb0, tb1, x.start_us, x.end_us) > 0 or (tb1 <= x.start_us <= tb1 + grace_us):
                xs.append(x)

        tb_to_kernels[tb] = ks
        tb_to_xfers[tb] = xs

    return tb_to_kernels, tb_to_xfers


def assign_kernels_to_smallest_op(
    ops_in_tb: list[IntervalEvt],
    kernels: list[IntervalEvt],
    *,
    grace_us: float,
) -> tuple[dict[IntervalEvt, list[IntervalEvt]], list[IntervalEvt]]:
    op_to: dict[IntervalEvt, list[IntervalEvt]] = defaultdict(list)
    leftovers: list[IntervalEvt] = []

    ops_sorted = sorted(ops_in_tb, key=lambda o: (o.dur_us, o.start_us, o.end_us, o.name))
    if not ops_sorted:
        return op_to, list(kernels)

    for k in sorted(kernels, key=lambda e: (e.start_us, e.end_us, e.name)):
        assigned: Optional[IntervalEvt] = None
        for op in ops_sorted:
            if _overlap_us(op.start_us, op.end_us, k.start_us, k.end_us) > 0 or (op.end_us <= k.start_us <= op.end_us + grace_us):
                assigned = op
                break
        if assigned is None:
            leftovers.append(k)
        else:
            op_to[assigned].append(k)

    return op_to, leftovers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out", default="timeline.json")
    ap.add_argument("--topk", type=int, default=0, help="0 = show ALL ops (default).")
    ap.add_argument("--min-ms", type=float, default=0.0)

    ap.add_argument("--op-prefix", action="append", default=["aten::"])

    ap.add_argument("--xfer-prefix", action="append", default=["cudaMemcpy", "Memcpy", "memcpy"])
    ap.add_argument("--xfer-contains", action="append", default=["HtoD", "DtoH", "DtoD", "H2D", "D2H", "D2D"])
    ap.add_argument("--xfer-cat", action="append", default=[])

    ap.add_argument("--kernels", action="store_true", default=True)
    ap.add_argument("--kernel-cat", action="append", default=["kernel"])

    ap.add_argument("--combined-topk", type=int, default=0)

    ap.add_argument("--tb-gpu-grace-us", type=float, default=2000.0)
    ap.add_argument("--op-gpu-grace-us", type=float, default=2000.0)

    ap.add_argument("--keep-gpu-tb", action="store_true")

    args = ap.parse_args()

    trace_path = Path(args.trace)
    out_path = Path(args.out)

    trace = load_json(trace_path)
    events = trace.get("traceEvents", [])

    tb_spans = extract_tb_spans(events, keep_gpu_tb=bool(args.keep_gpu_tb))
    if not tb_spans:
        print("No TB spans found.")
        write_json(out_path, {"meta": {"trace": str(trace_path)}, "t0_us": None, "tb_spans": [], "combined_timeline": []})
        print(f"Wrote {out_path}")
        return

    ts_candidates: list[float] = []
    for e in events:
        if "ts" in e:
            ts = _safe_float(e.get("ts", 0.0))
            if ts > 0:
                ts_candidates.append(ts)
    t0 = min(ts_candidates) if ts_candidates else min(s.start_us for s in tb_spans)

    def ms(us: float) -> float:
        return (us - t0) / 1000.0

    spans_by_time = sorted(tb_spans, key=lambda s: (s.start_us, s.end_us, s.tb_name, s.pid, s.tid))
    filtered_spans = [sp for sp in spans_by_time if (sp.dur_us / 1000.0) >= args.min_ms]

    op_prefixes = tuple(args.op_prefix)
    cpu_ops = extract_xphase_intervals(events, kind="op", startswith_prefixes=op_prefixes)

    xfer_prefixes = tuple(args.xfer_prefix)
    xfer_contains = tuple(args.xfer_contains)
    xfer_cats = tuple(args.xfer_cat) if args.xfer_cat else ()
    transfers = extract_xphase_intervals(events, kind="xfer", startswith_prefixes=xfer_prefixes, contains_tokens=xfer_contains, cat_allow=xfer_cats)
    transfers_sorted = sorted(transfers, key=lambda e: (e.start_us, e.end_us, e.name))

    kernels: list[IntervalEvt] = []
    kernel_cats = tuple(args.kernel_cat)
    if args.kernels:
        kernels = extract_xphase_intervals(events, kind="kernel", cat_allow=kernel_cats)
    kernels_sorted = sorted(kernels, key=lambda e: (e.start_us, e.end_us, e.name))

    ops_by_span = assign_ops_to_smallest_containing_tb(tb_spans, cpu_ops)

    tb_to_kernels, tb_to_xfers = associate_gpu_events_to_tb(
        filtered_spans,
        kernels_sorted,
        transfers_sorted,
        grace_us=float(args.tb_gpu_grace_us),
    )

    print("==============================")
    print("TB → CPU ops → (GPU kernels) + (GPU transfers direction)")
    print("==============================")

    for sp in filtered_spans:
        ops_in = sorted(ops_by_span.get(sp, []), key=lambda x: (x.start_us, x.end_us, x.name))
        tb_k = tb_to_kernels.get(sp, [])
        tb_x = tb_to_xfers.get(sp, [])

        op_to_k, k_left = assign_kernels_to_smallest_op(ops_in, tb_k, grace_us=float(args.op_gpu_grace_us))

        dir_counts = defaultdict(int)
        for x in tb_x:
            dir_counts[xfer_direction(x.name, x.cat)] += 1
        dir_summary = ", ".join(f"{d}={dir_counts[d]}" for d in sorted(dir_counts.keys())) if tb_x else "none"

        print(
            f"{ms(sp.start_us):10.3f}ms → {ms(sp.end_us):10.3f}ms  "
            f"({sp.dur_us/1000.0:8.3f} ms)  [TB] {sp.tb_name}  "
            f"(cat={sp.cat}, pid={sp.pid}, tid={sp.tid})  "
            f"| cpu_ops={len(ops_in)} kernels={len(tb_k)} xfers={len(tb_x)} [{dir_summary}]"
        )

        # show ALL ops unless user sets --topk
        to_show = ops_in if args.topk == 0 else ops_in[: args.topk]
        for op in to_show:
            ks = op_to_k.get(op, [])
            print(
                f"    [CPU OP] {ms(op.start_us):10.3f}ms → {ms(op.end_us):10.3f}ms  "
                f"({op.dur_us/1000.0:8.3f} ms)  {op.name} | gpu_kernels={len(ks)}"
            )
            # show ALL kernels for this op
            for k in ks:
                print(
                    f"        [GPU K ] {ms(k.start_us):10.3f}ms → {ms(k.end_us):10.3f}ms  "
                    f"({k.dur_us/1000.0:8.3f} ms)  type={classify_kernel(k.name)}  {k.name}"
                )

        # show ALL transfers under TB
        for x in tb_x:
            print(
                f"    [XFER ] {ms(x.start_us):10.3f}ms → {ms(x.end_us):10.3f}ms  "
                f"({x.dur_us/1000.0:8.3f} ms)  dir={xfer_direction(x.name, x.cat)}  {x.name}"
            )

        if args.kernels and k_left:
            print(f"    [TB GPU kernels not matched to an op: {len(k_left)}]")
        print()

    # ---- Combined timeline ----
    combined: list[IntervalEvt] = []
    for sp in filtered_spans:
        combined.append(IntervalEvt("tb", sp.tb_name, sp.start_us, sp.end_us, sp.cat, sp.pid, sp.tid))
    combined.extend(cpu_ops)
    combined.extend(transfers)
    combined.extend(kernels)
    combined.sort(key=lambda e: (e.start_us, e.end_us, e.kind, e.name))

    def _kind_label(ev: IntervalEvt) -> str:
        return {"tb": "TB  ", "op": "CPU ", "xfer": "XFER", "kernel": "GPU "}.get(ev.kind, ev.kind.upper()[:4])

    print("==============================")
    print("Combined timeline (TB + CPU ops + transfers + kernels)")
    print("==============================")
    to_print = combined if args.combined_topk == 0 else combined[: args.combined_topk]
    for ev in to_print:
        extra = ""
        if ev.kind == "kernel":
            extra = f" | type={classify_kernel(ev.name)}"
        elif ev.kind == "xfer":
            extra = f" | dir={xfer_direction(ev.name, ev.cat)}"
        print(
            f"{ms(ev.start_us):10.3f}ms → {ms(ev.end_us):10.3f}ms  "
            f"({ev.dur_us/1000.0:8.3f} ms)  [{_kind_label(ev)}] {ev.name}{extra}  "
            f"(cat={ev.cat}, pid={ev.pid}, tid={ev.tid})"
        )
    print()

    # ---- JSON ----
    def ev_to_json(ev: IntervalEvt):
        d = {
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
        if ev.kind == "kernel":
            d["kernel_type"] = classify_kernel(ev.name)
        if ev.kind == "xfer":
            d["xfer_direction"] = xfer_direction(ev.name, ev.cat)
        return d

    tb_spans_json = []
    for sp in filtered_spans:
        ops_in = sorted(ops_by_span.get(sp, []), key=lambda x: (x.start_us, x.end_us, x.name))
        tb_k = tb_to_kernels.get(sp, [])
        tb_x = tb_to_xfers.get(sp, [])
        op_to_k, k_left = assign_kernels_to_smallest_op(ops_in, tb_k, grace_us=float(args.op_gpu_grace_us))

        ops_json = []
        for op in ops_in:
            ops_json.append(
                {
                    **ev_to_json(op),
                    "exec_side": "cpu",
                    "gpu_kernels": [ev_to_json(k) for k in op_to_k.get(op, [])] if args.kernels else [],
                }
            )

        dir_counts = defaultdict(int)
        for x in tb_x:
            dir_counts[xfer_direction(x.name, x.cat)] += 1

        tb_spans_json.append(
            {
                "tb_name": sp.tb_name,
                "tb_cat": sp.cat,
                "tb_pid": sp.pid,
                "tb_tid": sp.tid,
                "start_us": sp.start_us,
                "end_us": sp.end_us,
                "start_ms": ms(sp.start_us),
                "end_ms": ms(sp.end_us),
                "dur_us": sp.dur_us,
                "dur_ms": sp.dur_us / 1000.0,
                "cpu_ops": ops_json,
                "tb_gpu_kernels": [ev_to_json(k) for k in tb_k] if args.kernels else [],
                "tb_gpu_kernels_unmatched": [ev_to_json(k) for k in k_left] if args.kernels else [],
                "tb_gpu_transfers": [ev_to_json(x) for x in tb_x],
                "tb_gpu_transfer_direction_counts": {k: dir_counts[k] for k in sorted(dir_counts.keys())},
            }
        )

    timeline_obj = {
        "meta": {
            "trace": str(trace_path),
            "kernels_enabled": bool(args.kernels),
            "kernel_cats": list(kernel_cats),
            "tb_gpu_grace_us": float(args.tb_gpu_grace_us),
            "op_gpu_grace_us": float(args.op_gpu_grace_us),
            "note": "Printing shows all ops/kernels/transfers (no truncation). Use --topk to cap ops if needed.",
        },
        "t0_us": t0,
        "tb_spans": tb_spans_json,
        "transfers": [ev_to_json(ev) for ev in transfers_sorted],
        "kernels": [ev_to_json(ev) for ev in kernels_sorted] if args.kernels else [],
        "combined_timeline": [ev_to_json(ev) for ev in combined],
    }

    write_json(out_path, timeline_obj)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()