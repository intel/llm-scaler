#!/usr/bin/env python3
"""Phase 4 Lens C — aggregate a unitrace chrome trace into the bottleneck verdict.

Usage:
    python3 unitrace_agg.py <trace.json | dir> [tail_frac]

- Auto-detects the device pid (the pid with the largest total ph=="X" duration;
  host-API events live on a different pid).
- Phase-slices to the tail `tail_frac` of the timeline (default 0.06) so the
  ~tens of seconds of model load + graph-capture warmup don't dominate. Pass a
  larger frac (or 1.0) to widen the window.
- Prints device-busy% (the launch-bound vs compute-bound verdict), the kernel
  family self-time breakdown (what to even consider writing), and the >1ms
  inter-op gap profile (host dead-air the GPU never sees).

Verdict reading:
  busy% low  (~15%) -> launch-bound / host-bound: lever is fewer launches
                       (graph/fusion) or host work (preprocess/IPC); a faster
                       single kernel buys ~nothing.
  busy% high (~80%) -> compute/BW-bound: now do a kernel roofline (Lens B).

Caveats (from the unitrace field guide, /llm/models/test/unitrace.md):
  - unitrace inflates absolute device-time; trust RELATIVE % only, not seconds.
  - --chrome-call-logging floods tiny-op counts; cross-check a suspicious
    "94% elementwise" against torch.profiler(record_shapes=True).
  - empty/135-byte trace or "ld.so libunitrace_tool.so" in serve.log => the .so
    was not on the library path; the run was uninstrumented.

Needs: pip install ijson  (streaming so GB-scale server traces don't OOM).
"""
import sys
import os
import glob
import collections


def fam(name):
    """Coarse kernel family bucket. Tune per model as needed."""
    low = name.lower()
    if "gemm" in low:
        return "gemm:" + name[:50]
    if any(k in low for k in ("fmha", "flash", "mha", "sdp", "attention", "varlen")):
        return "attention:" + name[:45]
    if "softmax" in low:
        return "softmax"
    if any(k in low for k in ("norm", "rms")):
        return "norm:" + name[:40]
    if any(k in low for k in ("gelu", "silu", "activation")):
        return "activation"
    if "conv" in low:
        return "conv:" + name[:40]
    if any(k in low for k in ("resize", "interp", "resample")):
        return "resize/interp"
    if "elementwise" in low or "vectorized" in low:
        return "elementwise:" + name[:40]
    if "reduce" in low:
        return "reduce"
    if any(k in low for k in ("copy", "fill", "memcpy")):
        return "copy/fill"
    if any(k in low for k in ("index", "gather", "scatter")):
        return "index/gather/scatter"
    if any(k in low for k in ("quant", "fp8", "dequant")):
        return "quant/fp8:" + name[:40]
    return name.split("<")[0].split("(")[0].split("[")[0][:50]


def main():
    import ijson

    F = sys.argv[1]
    if os.path.isdir(F):
        cands = sorted(glob.glob(os.path.join(F, "python3.*.json")),
                       key=lambda p: -os.path.getsize(p))
        if not cands:
            print("no python3.*.json in dir"); sys.exit(1)
        F = cands[0]
    TAIL = float(sys.argv[2]) if len(sys.argv) > 2 else 0.06

    # Pass 1: device pid = pid with largest total ph==X dur.
    pd = collections.defaultdict(float)
    pc = collections.defaultdict(int)
    with open(F, "rb") as fh:
        for ev in ijson.items(fh, "traceEvents.item"):
            if ev.get("ph") == "X":
                pd[ev.get("pid")] += float(ev.get("dur", 0))
                pc[ev.get("pid")] += 1
    if not pd:
        print("NO ph==X EVENTS (empty/uninstrumented trace?)"); sys.exit(1)
    DP = max(pd, key=pd.get)
    print(f"trace: {F}")
    print(f"device pid={DP} ph=X events={pc[DP]} total_dur={pd[DP]/1e6:.2f}s")

    # Pass 2: collect device events.
    evs = []
    with open(F, "rb") as fh:
        for ev in ijson.items(fh, "traceEvents.item"):
            if ev.get("ph") == "X" and ev.get("pid") == DP:
                evs.append((float(ev["ts"]), float(ev.get("dur", 0)), ev["name"]))
    evs.sort()
    t0, t1 = evs[0][0], evs[-1][0] + evs[-1][1]
    cut = t1 - (t1 - t0) * TAIL
    win = [e for e in evs if e[0] >= cut]
    wall = (t1 - cut) / 1e3
    busy = sum(d for _, d, _ in win)
    print(f"span={(t1-t0)/1e6:.2f}s  window wall={wall:.1f}ms ops={len(win)} "
          f"busy={busy/1e3:.2f}ms")
    print(f"\n=== device-busy% = {100*busy/1e3/wall:.1f}%  "
          f"(low~15=launch/host-bound, high~80=compute/BW-bound) ===")

    tot = collections.defaultdict(float)
    cnt = collections.defaultdict(int)
    for _, d, n in win:
        k = fam(n); tot[k] += d; cnt[k] += 1
    grand = sum(tot.values()) or 1.0
    print(f"{'family':<52}{'ms':>9}{'%':>6}{'calls':>8}{'us/call':>9}")
    for k in sorted(tot, key=tot.get, reverse=True)[:25]:
        print(f"{k:<52}{tot[k]/1e3:>9.2f}{100*tot[k]/grand:>5.1f}%"
              f"{cnt[k]:>8}{tot[k]/cnt[k]:>9.1f}")

    # gap profile: >1ms inter-op gaps with before/after kernel. A recurring
    # multi-ms gap between a D2H (read token) and the next M2D (next request)
    # is host dead-air (preprocess/IPC/schedule), not a kernel you can fix.
    print("\n=== >1ms inter-op gaps (host dead-air if between D2H and M2D) ===")
    prev = win[0][0] + win[0][1]; prevname = win[0][2]; idle = 0.0; nbig = 0
    for ts, d, n in win[1:]:
        g = ts - prev
        if g > 0:
            idle += g
        if g > 1000:
            nbig += 1
            if nbig <= 20:
                print(f"  gap={g/1e3:8.2f}ms  before=[{prevname[:40]}] after=[{n[:40]}]")
        if ts + d >= prev:
            prevname = n
        prev = max(prev, ts + d)
    print(f"idle={idle/1e3:.1f}ms ({100*idle/1e3/wall:.0f}% of wall), "
          f">1ms gaps={nbig}")


if __name__ == "__main__":
    main()
