"""Exact-reclear scans along audited unilateral deviation paths.

Every path moves ONE investor along the segment x(s) = x0 + s (x1 - x0),
s in [0, 1], with all rivals frozen at the final profile of
init5mw15mwh_jacobi_rho100_d025_s100.  Feasibility of every sample follows
from convexity: both endpoints satisfy nonnegativity and the 2-8 h duration
cone at every node, which are convex, and the shared nodal cap is affine, so
sum_i P_in(s) <= max(endpoint sums) <= 200 MW.  Each sample is nevertheless
re-checked with the maintained feasibility guard.

Payoffs come only from the maintained procedural reclear and settlement.

Stages
  1. coarse grid with log-spaced points near s = 0;
  2. adaptive shrinking-bracket traces for the largest apparent cliffs
     ("cliff" brackets, follow the half with the larger profit change) and for
     the largest price-forming active-set changes ("regime" brackets, follow the
     half that still contains the change);
  3. classification of each bracket from the scaling of |dprofit| with width.

Outputs (output/paths/): path_samples.csv, brackets.csv,
bracket_classification.csv, active_set_transitions.csv, rho_summary.csv,
long-format market/line/generator/storage CSV.gz, raw pickles.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import math
import pickle
import time

import numpy as np

import regret_common as rc

OUT = rc.OUTPUT_ROOT / "paths"
RAW = OUT / "raw"
RHOS = (100.0, 50.0, 25.0)
COARSE = sorted(
    {0.0, 1e-6, 1e-5, 1e-4, 1e-3, 2.5e-3, 5e-3, 1e-2, 2e-2, 3e-2, 5e-2, 7.5e-2}
    | {round(0.1 + 0.025 * k, 10) for k in range(37)}
)
MAX_LEVELS = 24
N_CLIFF = 3
N_REGIME = 4
CLIFF_MIN_EUR = 10.0
MIN_WIDTH = 5e-10


def path_definitions(nodes) -> dict:
    power, energy = rc.incumbent()

    def inc(inv):
        return rc.investor_vector(power, energy, inv, nodes)

    def rec(inv, start):
        r = rc.audit_record(inv, start, "refine")
        return ({n: float(r["power"][n]) for n in nodes}, {n: float(r["energy"][n]) for n in nodes},
                float(r["exact_profit"]))

    paths = {}
    for inv, start in (("I1", "incumbent"), ("I3", "incumbent"), ("I2", "incumbent")):
        p1, e1, pi = rec(inv, start)
        paths[f"{inv}_incumbent_start"] = dict(investor=inv, start=inc(inv), end=(p1, e1),
                                               end_label=f"{inv} incumbent-start audited refine",
                                               audit_end_profit=pi)
    for inv in ("I1", "I3", "I2"):
        sel = rc.selected_record(inv)
        paths[f"{inv}_selected_{sel['start']}"] = dict(
            investor=inv, start=inc(inv),
            end=({n: float(sel["power"][n]) for n in nodes}, {n: float(sel["energy"][n]) for n in nodes}),
            end_label=f"{inv} selected audited response ({sel['start']})",
            audit_end_profit=float(sel["exact_profit"]))
    p0, e0, _ = rec("I2", "incumbent")
    sel = rc.selected_record("I2")
    paths["I2_bridge_between_responses"] = dict(
        investor="I2", start=(p0, e0),
        end=({n: float(sel["power"][n]) for n in nodes}, {n: float(sel["energy"][n]) for n in nodes}),
        end_label="I2 incumbent-start response -> I2 selected response",
        audit_end_profit=float(sel["exact_profit"]))
    return paths, power, energy


def point_capacities(path, s, power, energy):
    p0, e0 = path["start"]
    p1, e1 = path["end"]
    cp = {n: max(0.0, p0[n] + s * (p1[n] - p0[n])) for n in p0}
    ce = {n: max(0.0, e0[n] + s * (e1[n] - e0[n])) for n in e0}
    full_p, full_e = rc.with_candidate(power, energy, path["investor"], cp, ce)
    return full_p, full_e, cp, ce


def ingest(store, key, s, res, static, phase, feasible):
    entry = {"res": res, "phase": phase, "feasible": feasible}
    if res.get("status") == "optimal":
        entry["sets"] = rc.active_sets(res, static)
    store.setdefault(key, {})[s] = entry


def profit(entry, inv):
    res = entry["res"]
    return res["settle"][inv]["profit"] if res.get("status") == "optimal" else math.nan


def run_scan(workers: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    data100 = rc.market_data(100.0)
    config100 = rc.game_config(data100)
    nodes = list(data100.nodes)
    static = rc.static_arrays(data100)
    paths, power, energy = path_definitions(nodes)
    store: dict = {}
    t0 = time.perf_counter()
    with rc.make_pool(workers) as pool:
        # ---------------------------------------------------------------- coarse
        tasks, meta = [], {}
        for name, path in paths.items():
            for rho in RHOS:
                for s in COARSE:
                    full_p, full_e, cp, ce = point_capacities(path, s, power, energy)
                    try:
                        rc.check_feasible(data100, config100, path["investor"], cp, ce, full_p)
                        feasible = True
                    except ValueError:
                        feasible = False
                    key = (name, rho, s)
                    meta[key] = feasible
                    tasks.append((key, rho, full_p, full_e))
        results = rc.evaluate_many(pool, tasks)
        for (name, rho, s), res in results.items():
            ingest(store, (name, rho), s, res, static, "coarse", meta[(name, rho, s)])
        print(f"coarse done: {len(tasks)} clears in {time.perf_counter()-t0:.0f}s", flush=True)

        # ------------------------------------------------------ bracket seeding
        brackets: dict = {}
        for name, path in paths.items():
            inv = path["investor"]
            for rho in RHOS:
                pts = store[(name, rho)]
                ss = sorted(pts)
                stats = []
                for a, b in zip(ss[:-1], ss[1:]):
                    ea, eb = pts[a], pts[b]
                    if "sets" not in ea or "sets" not in eb:
                        continue
                    dpi = profit(eb, inv) - profit(ea, inv)
                    dl = float(np.max(np.abs(eb["res"]["lmp"] - ea["res"]["lmp"])))
                    regime = (ea["sets"]["lines_priced"] != eb["sets"]["lines_priced"]) or (
                        ea["sets"]["generation"] != eb["sets"]["generation"])
                    stats.append(dict(a=a, b=b, w=b - a, dpi=dpi, dlmp=dl, regime=regime))
                wide = [abs(x["dpi"]) / x["w"] for x in stats if x["w"] >= 0.02]
                smooth_slope = float(np.median(wide)) if wide else 0.0
                for x in stats:
                    x["score"] = abs(x["dpi"]) - smooth_slope * x["w"]
                chosen = []
                for x in sorted(stats, key=lambda z: -z["score"]):
                    if len([c for c in chosen if c["kind"] == "cliff"]) >= N_CLIFF:
                        break
                    if x["score"] > CLIFF_MIN_EUR:
                        chosen.append(dict(kind="cliff", a=x["a"], b=x["b"], seed_score=x["score"]))
                for x in sorted([z for z in stats if z["regime"]], key=lambda z: -z["dlmp"]):
                    if len([c for c in chosen if c["kind"] == "regime"]) >= N_REGIME:
                        break
                    if any(c["a"] == x["a"] and c["b"] == x["b"] for c in chosen):
                        continue
                    chosen.append(dict(kind="regime", a=x["a"], b=x["b"], seed_score=x["dlmp"]))
                for k, c in enumerate(chosen):
                    c.update(id=f"{c['kind']}{k}", level=0, done=False, trace=[], smooth_slope=smooth_slope)
                brackets[(name, rho)] = chosen

        # ------------------------------------------------------------- traces
        for level in range(MAX_LEVELS):
            tasks, meta = [], {}
            for (name, rho), brs in brackets.items():
                path = paths[name]
                for br in brs:
                    if br["done"]:
                        continue
                    m = 0.5 * (br["a"] + br["b"])
                    br["mid"] = m
                    if m in store[(name, rho)]:
                        continue
                    full_p, full_e, cp, ce = point_capacities(path, m, power, energy)
                    try:
                        rc.check_feasible(data100, config100, path["investor"], cp, ce, full_p)
                        feasible = True
                    except ValueError:
                        feasible = False
                    key = (name, rho, m)
                    if key in meta:
                        continue
                    meta[key] = feasible
                    tasks.append((key, rho, full_p, full_e, level == MAX_LEVELS - 1))
            if not tasks:
                break
            results = rc.evaluate_many(pool, tasks)
            for (name, rho, s), res in results.items():
                ingest(store, (name, rho), s, res, static, f"trace_level_{level+1}", meta[(name, rho, s)])
            for (name, rho), brs in brackets.items():
                inv = paths[name]["investor"]
                pts = store[(name, rho)]
                for br in brs:
                    if br["done"]:
                        continue
                    a, m, b = br["a"], br["mid"], br["b"]
                    ea, em, eb = pts[a], pts[m], pts[b]
                    if "sets" not in em:
                        br["done"] = True
                        br["stop"] = "midpoint clear failed"
                        continue
                    left = dict(a=a, b=m, dpi=profit(em, inv) - profit(ea, inv),
                                dlmp=float(np.max(np.abs(em["res"]["lmp"] - ea["res"]["lmp"]))),
                                lp=ea["sets"]["lines_priced"] != em["sets"]["lines_priced"],
                                gen=ea["sets"]["generation"] != em["sets"]["generation"])
                    right = dict(a=m, b=b, dpi=profit(eb, inv) - profit(em, inv),
                                 dlmp=float(np.max(np.abs(eb["res"]["lmp"] - em["res"]["lmp"]))),
                                 lp=em["sets"]["lines_priced"] != eb["sets"]["lines_priced"],
                                 gen=em["sets"]["generation"] != eb["sets"]["generation"])
                    if br["kind"] == "cliff":
                        pick = left if abs(left["dpi"]) >= abs(right["dpi"]) else right
                    else:
                        lp_sides = [x for x in (left, right) if x["lp"]]
                        gen_sides = [x for x in (left, right) if x["gen"]]
                        if len(lp_sides) == 1:
                            pick = lp_sides[0]
                        elif len(gen_sides) == 1 and not lp_sides:
                            pick = gen_sides[0]
                        else:
                            pick = left if left["dlmp"] >= right["dlmp"] else right
                    br["a"], br["b"] = pick["a"], pick["b"]
                    br["level"] = level + 1
                    br["trace"].append(dict(level=level + 1, a=pick["a"], b=pick["b"], width=pick["b"] - pick["a"],
                                            dpi=pick["dpi"], dlmp=pick["dlmp"], lines_priced_change=pick["lp"],
                                            generation_change=pick["gen"]))
                    if pick["b"] - pick["a"] < MIN_WIDTH:
                        br["done"], br["stop"] = True, "minimum width"
                    elif br["kind"] == "regime" and not (pick["lp"] or pick["gen"]) and pick["dlmp"] < 1e-9:
                        br["done"], br["stop"] = True, "change vanished"
            print(f"trace level {level+1}: {len(tasks)} clears, t={time.perf_counter()-t0:.0f}s", flush=True)
            if level % 4 == 3:
                with (RAW / "scan_state.pkl").open("wb") as h:
                    pickle.dump({"store": store, "brackets": brackets, "paths": paths}, h, protocol=pickle.HIGHEST_PROTOCOL)

    with (RAW / "scan_state.pkl").open("wb") as h:
        pickle.dump({"store": store, "brackets": brackets, "paths": paths}, h, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"scan complete in {time.perf_counter()-t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------
# Analysis and export


def classify_bracket(br, entries, inv, max_dp_mw):
    trace = br["trace"]
    if not trace:
        return dict(classification="not_traced")
    finest = trace[-1]
    usable = [t for t in trace if t["width"] > 0]
    tail = usable[-10:]
    slope = math.nan
    if len(tail) >= 4:
        x = np.log10([t["width"] for t in tail])
        y = np.log10([max(abs(t["dpi"]), 1e-9) for t in tail])
        slope = float(np.polyfit(x, y, 1)[0])
    ea, eb = entries[finest["a"]], entries[finest["b"]]
    pd_max = max(abs(ea["res"]["pd_gap"]), abs(eb["res"]["pd_gap"]))
    comp_max = max(ea["res"]["max_complementarity"], eb["res"]["max_complementarity"])
    fdpi = abs(finest["dpi"])
    verify_failed = any(str(e["res"].get("verify_status", "")).startswith("failed") for e in (ea, eb))
    verify_err = max(float(e["res"].get("verify_max_abs_profit_diff", 0.0) or 0.0) for e in (ea, eb))
    # An inexact reference clear only matters if its measured profit error is
    # comparable with the change being classified.
    if verify_failed or verify_err > max(0.5, 0.2 * fdpi):
        label = "numerically_unreliable_reclear"
    elif fdpi < 1e-3 and finest["dlmp"] < 1e-6:
        label = "change_resolves_to_zero"
    elif not math.isnan(slope) and slope >= 0.8:
        label = "steep_continuous"
    elif not math.isnan(slope) and slope <= 0.2 and fdpi > 1.0:
        label = "discontinuity_like"
    else:
        label = "unresolved"
    lip = fdpi / finest["width"] if finest["width"] > 0 else math.nan
    return dict(
        classification=label,
        loglog_slope_last10=slope,
        finest_width_s=finest["width"],
        finest_abs_dprofit=fdpi,
        finest_dlmp=finest["dlmp"],
        local_slope_eur_per_day_per_unit_s=lip,
        local_slope_eur_per_day_per_mw=lip / max_dp_mw if max_dp_mw > 0 else math.nan,
        finest_pd_gap_max=pd_max,
        finest_complementarity_max=comp_max,
        lines_priced_entered=";".join(sorted(eb["sets"]["lines_priced"] - ea["sets"]["lines_priced"])),
        lines_priced_left=";".join(sorted(ea["sets"]["lines_priced"] - eb["sets"]["lines_priced"])),
        generation_changes=";".join(sorted(ea["sets"]["generation"] ^ eb["sets"]["generation"]))[:2000],
        storage_changes=len(ea["sets"]["storage"] ^ eb["sets"]["storage"]),
    )


def export(state) -> None:
    store, brackets, paths = state["store"], state["brackets"], state["paths"]
    data = rc.market_data(100.0)
    N, T, L, G = list(data.nodes), list(data.times), list(data.lines), list(data.generators)
    static = rc.static_arrays(data)
    total_load = float(static["demand"].sum())

    sample_rows, transition_rows, bracket_rows, class_rows, rho_rows = [], [], [], [], []
    mkt = gzip.open(OUT / "path_market_long.csv.gz", "wt", newline="", encoding="utf-8")
    lin = gzip.open(OUT / "path_lines_long.csv.gz", "wt", newline="", encoding="utf-8")
    gen = gzip.open(OUT / "path_generators_long.csv.gz", "wt", newline="", encoding="utf-8")
    sto = gzip.open(OUT / "path_storage_long.csv.gz", "wt", newline="", encoding="utf-8")
    wm, wl, wg, ws = (csv.writer(f) for f in (mkt, lin, gen, sto))
    wm.writerow(["path", "rho", "s", "node", "hour", "lmp", "demand", "demand_adjustment", "net_injection",
                 "generation_mw", "charge_I1", "discharge_I1", "charge_I2", "discharge_I2", "charge_I3", "discharge_I3"])
    wl.writerow(["path", "rho", "s", "line", "hour", "flow", "limit", "mu_up", "mu_dn", "binding_up", "binding_dn",
                 "priced_up", "priced_dn"])
    wg.writerow(["path", "rho", "s", "generator", "hour", "dispatch", "capacity", "nu", "state"])
    ws.writerow(["path", "rho", "s", "investor", "node", "hour", "power", "energy", "charge", "discharge", "soc_end",
                 "rho_ch", "sig_dis", "del_soc_end"])
    gen_nodes = [N.index(data.nodes_by_generator()[g][0]) for g in G]
    fmt = lambda x: f"{x:.10g}"

    for (name, rho), pts in sorted(store.items()):
        path = paths[name]
        inv = path["investor"]
        ss = sorted(pts)
        p0, e0 = path["start"]
        p1, e1 = path["end"]
        max_dp = max(abs(p1[n] - p0[n]) for n in N)
        max_de = max(abs(e1[n] - e0[n]) for n in N)
        base = pts[0.0]
        prev = None
        for s in ss:
            entry = pts[s]
            res = entry["res"]
            row = dict(path=name, rho=rho, investor=inv, s=repr(s), phase=entry["phase"],
                       feasible=entry["feasible"], status=res.get("status"), seconds=res.get("seconds"),
                       max_nodal_power_change_full_path_mw=max_dp, max_nodal_energy_change_full_path_mwh=max_de)
            for n in N:
                row[f"P_{n}"] = p0[n] + s * (p1[n] - p0[n])
            for n in N:
                row[f"E_{n}"] = e0[n] + s * (e1[n] - e0[n])
            row["total_power_mw"] = sum(row[f"P_{n}"] for n in N)
            row["total_energy_mwh"] = sum(row[f"E_{n}"] for n in N)
            if res.get("status") == "optimal":
                st = res["settle"][inv]
                row.update(profit=st["profit"], gain_vs_incumbent=st["profit"] - base["res"]["settle"][inv]["profit"],
                           relative_gain=(st["profit"] - base["res"]["settle"][inv]["profit"]) /
                           max(1.0, abs(base["res"]["settle"][inv]["profit"])),
                           storage_settlement=st["storage_settlement"], degradation=st["degradation"],
                           storage_operating_surplus=st["storage_operating_surplus"],
                           owned_generation_rent=st["owned_generation_rent"], capex=st["capex"])
                for other in rc.INVESTORS:
                    row[f"profit_{other}"] = res["settle"][other]["profit"]
                dl0 = np.abs(res["lmp"] - base["res"]["lmp"])
                k = np.unravel_index(np.argmax(dl0), dl0.shape)
                row.update(max_abs_lmp_change_vs_incumbent=float(dl0.max()),
                           argmax_lmp_change=f"{N[k[0]]}@{T[k[1]]}",
                           max_abs_lmp_change_vs_previous=(float(np.max(np.abs(res["lmp"] - prev["res"]["lmp"])))
                                                           if prev is not None and prev["res"].get("status") == "optimal" else math.nan),
                           iso_objective=res["objective"], pd_gap=res["pd_gap"],
                           max_primal_violation=res["max_primal_violation"],
                           max_dual_infeasibility=res["max_dual_infeasibility"],
                           max_stationarity_residual=res["max_stationarity_residual"],
                           max_complementarity=res["max_complementarity"],
                           total_abs_demand_adjustment_mwh=res["total_abs_da_mwh"],
                           demand_adjustment_share_of_load=res["total_abs_da_mwh"] / total_load,
                           max_abs_demand_adjustment_mw=res["max_abs_da_mw"],
                           n_lines_priced=len(entry["sets"]["lines_priced"]),
                           n_lines_binding=len(entry["sets"]["lines_binding"]),
                           lines_priced=";".join(sorted(entry["sets"]["lines_priced"])),
                           lines_priced_hash=rc.signature_hash(entry["sets"]["lines_priced"]),
                           generation_state_hash=rc.signature_hash(entry["sets"]["generation"]),
                           storage_state_hash=rc.signature_hash(entry["sets"]["storage"]),
                           repeat_max_abs_lmp_diff=res.get("repeat_max_abs_lmp_diff", ""),
                           repeat_max_abs_profit_diff=res.get("repeat_max_abs_profit_diff", ""),
                           reference_inexact=res.get("reference_inexact", ""),
                           verify_status=res.get("verify_status", ""),
                           verify_pd_gap=res.get("verify_pd_gap", ""),
                           verify_max_abs_lmp_diff=res.get("verify_max_abs_lmp_diff", ""),
                           verify_max_abs_profit_diff=res.get("verify_max_abs_profit_diff", ""))
                # long tables
                srep = repr(s)
                for a, n in enumerate(N):
                    for b, t in enumerate(T):
                        gsum = sum(res["gen"][g, b] for g in range(len(G)) if gen_nodes[g] == a)
                        wm.writerow([name, rho, srep, n, t, fmt(res["lmp"][a, b]), fmt(static["demand"][a, b]),
                                     fmt(res["da"][a, b]), fmt(res["ni"][a, b]), fmt(gsum),
                                     *[fmt(res[k2][u, a, b]) for u in range(3) for k2 in ("charge", "discharge")]])
                for a, l in enumerate(L):
                    for b, t in enumerate(T):
                        f = res["flow"][a, b]
                        lim = static["limit"][a]
                        wl.writerow([name, rho, srep, l, t, fmt(f), lim, fmt(res["mu_up"][a, b]), fmt(res["mu_dn"][a, b]),
                                     int(lim - f <= rc.SLACK_TOL), int(f + lim <= rc.SLACK_TOL),
                                     int(-res["mu_up"][a, b] > rc.DUAL_TOL), int(res["mu_dn"][a, b] > rc.DUAL_TOL)])
                for a, g in enumerate(G):
                    for b, t in enumerate(T):
                        c = static["cap"][a, b]
                        d = res["gen"][a, b]
                        state = "nocap" if c <= rc.SLACK_TOL else ("cap" if c - d <= rc.SLACK_TOL else ("off" if d <= rc.SLACK_TOL else "marg"))
                        wg.writerow([name, rho, srep, g, t, fmt(d), fmt(c), fmt(res["nu"][a, b]), state])
                for u, uname in enumerate(rc.INVESTORS):
                    for a, n in enumerate(N):
                        if res["power"][u, a] <= 1e-3:
                            continue
                        for b, t in enumerate(T):
                            ws.writerow([name, rho, srep, uname, n, t, fmt(res["power"][u, a]), fmt(res["energy"][u, a]),
                                         fmt(res["charge"][u, a, b]), fmt(res["discharge"][u, a, b]),
                                         fmt(res["soc"][u, a, b + 1]), fmt(res["rho_ch"][u, a, b]),
                                         fmt(res["sig_dis"][u, a, b]), fmt(res["del_soc"][u, a, b + 1])])
                if prev is not None and "sets" in prev:
                    ps = prev["sets"]
                    cs = entry["sets"]
                    transition_rows.append(dict(
                        path=name, rho=rho, investor=inv, s_from=repr(prev_s), s_to=repr(s), width=s - prev_s,
                        dprofit=st["profit"] - prev["res"]["settle"][inv]["profit"],
                        dlmp=float(np.max(np.abs(res["lmp"] - prev["res"]["lmp"]))),
                        lines_priced_entered=";".join(sorted(cs["lines_priced"] - ps["lines_priced"])),
                        lines_priced_left=";".join(sorted(ps["lines_priced"] - cs["lines_priced"])),
                        lines_binding_changes=len(cs["lines_binding"] ^ ps["lines_binding"]),
                        generation_changes=len(cs["generation"] ^ ps["generation"]),
                        generation_changed=";".join(sorted(cs["generation"] ^ ps["generation"]))[:1000],
                        storage_changes=len(cs["storage"] ^ ps["storage"])))
            sample_rows.append(row)
            if res.get("status") == "optimal":
                prev, prev_s = entry, s

        for br in brackets.get((name, rho), []):
            for t in br["trace"]:
                bracket_rows.append(dict(path=name, rho=rho, investor=inv, bracket=br["id"], kind=br["kind"], **t))
            cls = classify_bracket(br, pts, inv, max_dp)
            class_rows.append(dict(path=name, rho=rho, investor=inv, bracket=br["id"], kind=br["kind"],
                                   seed_score=br["seed_score"], levels=br["level"], stop=br.get("stop", ""),
                                   final_a=repr(br["a"]), final_b=repr(br["b"]), **cls))

        end = pts[max(ss)]
        rho_rows.append(dict(
            path=name, rho=rho, investor=inv,
            incumbent_profit=base["res"]["settle"][inv]["profit"],
            endpoint_profit=end["res"]["settle"][inv]["profit"] if end["res"].get("status") == "optimal" else math.nan,
            endpoint_gain=(end["res"]["settle"][inv]["profit"] - base["res"]["settle"][inv]["profit"])
            if end["res"].get("status") == "optimal" else math.nan,
            max_gain_on_path=max(pts[s]["res"]["settle"][inv]["profit"] - base["res"]["settle"][inv]["profit"]
                                 for s in ss if pts[s]["res"].get("status") == "optimal"),
            argmax_s=repr(max((s for s in ss if pts[s]["res"].get("status") == "optimal"),
                              key=lambda s: pts[s]["res"]["settle"][inv]["profit"])),
            incumbent_total_abs_da_mwh=base["res"]["total_abs_da_mwh"],
            incumbent_da_share_of_load=base["res"]["total_abs_da_mwh"] / total_load,
            incumbent_max_abs_da_mw=base["res"]["max_abs_da_mw"],
            samples=len(ss),
            max_pd_gap=max(abs(pts[s]["res"]["pd_gap"]) for s in ss if pts[s]["res"].get("status") == "optimal"),
            n_lines_priced_regimes=len({pts[s]["sets"]["lines_priced"] for s in ss if "sets" in pts[s]}),
            n_generation_regimes=len({pts[s]["sets"]["generation"] for s in ss if "sets" in pts[s]}),
        ))

    for f in (mkt, lin, gen, sto):
        f.close()
    for fname, rows in (("path_samples.csv", sample_rows), ("active_set_transitions.csv", transition_rows),
                        ("brackets.csv", bracket_rows), ("bracket_classification.csv", class_rows),
                        ("rho_summary.csv", rho_rows)):
        if not rows:
            continue
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with (OUT / fname).open("w", newline="", encoding="utf-8") as h:
            w = csv.DictWriter(h, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    print("export complete:", {k: len(v) for k, v in (("samples", sample_rows), ("transitions", transition_rows),
                                                      ("bracket_levels", bracket_rows), ("brackets", class_rows))})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    if not args.export_only:
        run_scan(args.workers)
    with (RAW / "scan_state.pkl").open("rb") as h:
        state = pickle.load(h)
    export(state)


if __name__ == "__main__":
    main()
