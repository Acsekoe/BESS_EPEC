"""Console digest of the path-scan checkpoint (no solves)."""

from __future__ import annotations

import pickle

import numpy as np

import regret_common as rc

RAW = rc.OUTPUT_ROOT / "paths" / "raw" / "scan_state.pkl"


def main() -> None:
    with RAW.open("rb") as h:
        state = pickle.load(h)
    store, brackets, paths = state["store"], state["brackets"], state["paths"]
    print(f"{'path':<34} {'rho':>5} {'n':>4} {'pi0':>10} {'gain(1)':>9} {'maxgain':>9} {'@s':>8}"
          f" {'maxjump':>9} {'jump@[a,b]':>26} {'dLMPmax':>8} {'LPreg':>5} {'GENreg':>6} {'pdgap':>8} {'verr':>7}")
    for (name, rho), pts in sorted(store.items()):
        inv = paths[name]["investor"]
        ss = sorted(s for s in pts if pts[s]["res"].get("status") == "optimal")
        prof = {s: pts[s]["res"]["settle"][inv]["profit"] for s in ss}
        pi0 = prof[0.0]
        best = max(ss, key=lambda s: prof[s])
        jumps = [(abs(prof[b] - prof[a]), a, b) for a, b in zip(ss[:-1], ss[1:])]
        mj = max(jumps)
        dl = max(float(np.max(np.abs(pts[s]["res"]["lmp"] - pts[0.0]["res"]["lmp"]))) for s in ss)
        lp = len({pts[s]["sets"]["lines_priced"] for s in ss})
        gen = len({pts[s]["sets"]["generation"] for s in ss})
        pd = max(abs(pts[s]["res"]["pd_gap"]) for s in ss)
        verr = max(float(pts[s]["res"].get("verify_max_abs_profit_diff", 0.0) or 0.0) for s in ss)
        print(f"{name:<34} {rho:>5g} {len(ss):>4} {pi0:>10.1f} {prof[ss[-1]]-pi0:>9.1f} {prof[best]-pi0:>9.1f} {best:>8.3g}"
              f" {mj[0]:>9.2f} {f'[{mj[1]:.4g},{mj[2]:.4g}]':>26} {dl:>8.3f} {lp:>5} {gen:>6} {pd:>8.1e} {verr:>7.3f}")
    print()
    for (name, rho), brs in sorted(brackets.items()):
        for br in brs:
            if br["trace"]:
                t = br["trace"][-1]
                print(f"{name:<34} rho={rho:<5g} {br['id']:<8} level={br['level']:>2} w={t['width']:.2e}"
                      f" |dpi|={abs(t['dpi']):.4g} dLMP={t['dlmp']:.3g} LPchg={t['lines_priced_change']} GENchg={t['generation_change']}")


if __name__ == "__main__":
    main()
