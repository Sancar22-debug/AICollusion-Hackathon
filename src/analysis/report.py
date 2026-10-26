"""One-command CLI report & figure generator (spec section 9.4 / 10).

Also exports a stratified 150-message validation sample of real transcripts
(`data/annotations/inter_annotator_sample.jsonl`); agreement is reported only for labels that exist.

Usage:
  python -m src.analysis.report --episodes data/episodes.jsonl --audits data/audits_test.jsonl --figdir paper/figures
"""
from __future__ import annotations

import argparse
import json
import os
import random
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.auditor.extractor import deception_rows, refresh_episode_extraction, rule_extract_message
from src.auditor.plus_factors import evaluate_evidence_ladder, extract_plus_factors
from src.auditor.taxonomy import TAXONOMY_TAGS, tag_episode_messages, tag_message_rule
from src.analysis.stats import (
    cluster_bootstrap_ci,
    cohens_kappa,
    episodes_to_dataframe,
    evaluate_monitor_table,
    holm_bonferroni,
    paired_cluster_bootstrap_auroc_diff,
    paired_permutation_test,
)


def build_inter_annotator_sample(episodes: List[dict], out_path: str = "data/annotations/inter_annotator_sample.jsonl") -> dict:
    """Export a stratified 150-message validation sample from REAL episode transcripts.

    Each record carries the deterministic rule-extractor output plus empty
    `human_*` fields to be filled in by a human annotator. Agreement (Cohen's kappa)
    is computed ONLY between labels that actually exist on disk:
      * rule extractor vs. LLM extractor (`llm_extractor` field, if present), and
      * rule extractor vs. human labels (`human_claimed_signal`, ... if filled in).
    No annotations are ever synthesised.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    existing = {}
    existing_list = []
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    existing[(r["episode_id"], r["t"], r["speaker"], r["text"])] = r
                    existing_list.append(r)
    if len(existing_list) == 150 and all(r.get("llm_extractor") for r in existing_list):
        records = existing_list
        for r in records:
            rule_ex = rule_extract_message({"speaker": r["speaker"], "text": r["text"]})
            r["extractor_rule"] = {
                "claimed_signal": rule_ex["claimed_signal"],
                "committed_priority": rule_ex["committed_priority"],
            }
            r["rule_tags"] = tag_message_rule(r["text"])
        with open(out_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
    else:
        pool = []
        for e in episodes:
            if e.get("paraphrase", 0) != 0 or e.get("noise", "low") != "low":
                continue
            for r in e["rounds"]:
                for m in r.get("messages", []):
                    pool.append((e["episode_id"], e["condition"], r["t"], m))
        rng = random.Random(20261024)
        sampled = rng.sample(pool, min(150, len(pool))) if pool else []
        records = []
        for idx, (eid, cond, t, m) in enumerate(sampled):
            key = (eid, t, m["speaker"], m["text"])
            if key in existing:
                records.append(existing[key])
                continue
            rule_ex = rule_extract_message(m)
            records.append({
                "sample_id": idx, "episode_id": eid, "condition": cond, "t": t,
                "speaker": m["speaker"], "text": m["text"],
                "extractor_rule": {"claimed_signal": rule_ex["claimed_signal"],
                                   "committed_priority": rule_ex["committed_priority"]},
                "rule_tags": tag_message_rule(m["text"]),
                "human_claimed_signal": None, "human_committed_priority": None,
            })
        with open(out_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    out = {"n_messages": len(records)}
    llm = [r for r in records if r.get("llm_extractor")]
    if llm:
        out["n_llm_labelled"] = len(llm)
        out["kappa_rule_vs_llm_signal"] = cohens_kappa([r["extractor_rule"]["claimed_signal"] for r in llm],
                                                       [r["llm_extractor"]["claimed_signal"] for r in llm])
        out["kappa_rule_vs_llm_priority"] = cohens_kappa([r["extractor_rule"]["committed_priority"] for r in llm],
                                                         [r["llm_extractor"]["committed_priority"] for r in llm])
    hum = [r for r in records if r.get("human_claimed_signal") is not None]
    if hum:
        out["n_human_labelled"] = len(hum)
        out["kappa_rule_vs_human_signal"] = cohens_kappa([r["extractor_rule"]["claimed_signal"] for r in hum],
                                                         [r["human_claimed_signal"] for r in hum])
        out["kappa_rule_vs_human_priority"] = cohens_kappa([r["extractor_rule"]["committed_priority"] for r in hum],
                                                           [r["human_committed_priority"] for r in hum])
    if not llm and not hum:
        out["note"] = "No independent labels yet; agreement not computed."
    return out


def compute_all_summaries(episodes_path: str, audits_test_path: str, exclude_comprehension_fail: bool = False) -> dict:
    with open(episodes_path) as f:
        all_eps = [refresh_episode_extraction(json.loads(line)) for line in f if line.strip()]
    with open(audits_test_path) as f:
        audits_test = [json.loads(line) for line in f if line.strip()]

    core_eps = [
        e for e in all_eps
        if e["condition"] in ("A1", "A2", "A3", "B1", "B2", "B3")
        and e.get("noise", "low") == "low"
        and e.get("paraphrase", 0) == 0
    ]
    # PREREG s2: keep only seeds for which all 12 core cells finished (balanced paired design)
    cells_per_seed = {}
    for e in core_eps:
        cells_per_seed.setdefault(int(e["seed"]), set()).add((e["condition"], e.get("framing", "crisis")))
    complete_seeds = sorted(s for s, c in cells_per_seed.items() if len(c) == 12)
    core_eps = [e for e in core_eps if int(e["seed"]) in complete_seeds]
    if exclude_comprehension_fail:
        bad_seeds = {int(e["seed"]) for e in core_eps
                     if not (e.get("comprehension_pass", {}).get("A", True) and e.get("comprehension_pass", {}).get("B", True))}
        # drop whole failing episodes; pairing helpers below align on common seeds
        core_eps = [e for e in core_eps
                    if e.get("comprehension_pass", {}).get("A", True) and e.get("comprehension_pass", {}).get("B", True)]
    core_test_eps = [e for e in core_eps if e.get("split") == "test"] or list(core_eps)
    df_all = episodes_to_dataframe(all_eps)
    df_core = episodes_to_dataframe(core_eps)

    # Cell summary table
    cell_rows = []
    for (cond, framing), grp in df_core.groupby(["condition", "framing"]):
        ujo_m, ujo_lo, ujo_hi = cluster_bootstrap_ci(grp["UJO_rate"].values)
        pv_m, pv_lo, pv_hi = cluster_bootstrap_ci(grp["protocol_violation_rate"].values)
        hco_m, hco_lo, hco_hi = cluster_bootstrap_ci(grp["HCO_rate"].values)
        cc_m, cc_lo, cc_hi = cluster_bootstrap_ci(grp["confirmed_collusion_rate"].values)
        mc_m, mc_lo, mc_hi = cluster_bootstrap_ci(grp["miscoordination_rate"].values)
        reg_m, reg_lo, reg_hi = cluster_bootstrap_ci(grp["total_regret"].values)
        ca_m, ca_lo, ca_hi = cluster_bootstrap_ci(grp["coalition_advantage_mean"].values)
        cell_rows.append({
            "condition": cond,
            "framing": framing,
            "n_seeds": len(grp),
            "UJO_rate": ujo_m,
            "UJO_ci": (ujo_lo, ujo_hi),
            "PV_rate": pv_m,
            "PV_ci": (pv_lo, pv_hi),
            "HCO_rate": hco_m,
            "HCO_ci": (hco_lo, hco_hi),
            "CC_rate": cc_m,
            "CC_ci": (cc_lo, cc_hi),
            "MC_rate": mc_m,
            "MC_ci": (mc_lo, mc_hi),
            "regret_mean": reg_m,
            "regret_ci": (reg_lo, reg_hi),
            "CA_mean": ca_m,
            "CA_ci": (ca_lo, ca_hi),
            "joint_override_rate": float(grp["n_joint_override"].mean() / 5.0),
            "ambiguous_override_rate": float(grp["n_ambiguous_override"].mean() / 5.0),
            "comp_pass_rate": float(grp["comp_pass_both"].mean()),
        })
    df_cells = pd.DataFrame(cell_rows)

    def _seed_series(mask, col="UJO_rate", df=None):
        d = df_core if df is None else df
        return d[mask(d)].groupby("seed")[col].mean()

    def _paired_series(s1, s2):
        common = s1.index.intersection(s2.index)
        return s1.loc[common].values, s2.loc[common].values

    # H1: per-seed mean UJO over all B cells minus all A cells (both framings); one-sided
    h1_res = paired_permutation_test(*_paired_series(
        _seed_series(lambda d: d["incentive"] == "conflicting"),
        _seed_series(lambda d: d["incentive"] == "aligned")), alternative="greater")

    # H1 diagnostic: Confirmed Collusion rate (B vs A) and Coalition Advantage (B vs A)
    h1_cc_res = paired_permutation_test(*_paired_series(
        _seed_series(lambda d: d["incentive"] == "conflicting", "confirmed_collusion_rate"),
        _seed_series(lambda d: d["incentive"] == "aligned", "confirmed_collusion_rate")), alternative="greater")
    h1_ca_res = paired_permutation_test(*_paired_series(
        _seed_series(lambda d: d["incentive"] == "conflicting", "coalition_advantage_mean"),
        _seed_series(lambda d: d["incentive"] == "aligned", "coalition_advantage_mean")), alternative="greater")

    # H2a: B3 vs B1 UJO, pooled over framings; one-sided (plus crisis/routine breakdown)
    h2a_res = paired_permutation_test(*_paired_series(
        _seed_series(lambda d: d["condition"] == "B3"),
        _seed_series(lambda d: d["condition"] == "B1")), alternative="greater")
    h2a_crisis_res = paired_permutation_test(*_paired_series(
        _seed_series(lambda d: (d["condition"] == "B3") & (d["framing"] == "crisis")),
        _seed_series(lambda d: (d["condition"] == "B1") & (d["framing"] == "crisis"))), alternative="greater")
    h2a_routine_res = paired_permutation_test(*_paired_series(
        _seed_series(lambda d: (d["condition"] == "B3") & (d["framing"] == "routine")),
        _seed_series(lambda d: (d["condition"] == "B1") & (d["framing"] == "routine"))), alternative="greater")

    # H2b: oversight shift index, B2 vs B3 messages pooled over framings; two-sided
    tax_rows = []
    for e in core_eps:
        if e["condition"] in ("A2", "A3", "B2", "B3"):
            tax_rows.extend(tag_episode_messages(e))
    df_tax = pd.DataFrame(tax_rows)
    if len(df_tax):
        df_tax["covert_or_rationalized"] = ((df_tax["camouflaged_signaling"] > 0) | (df_tax["moral_rationalization"] > 0)).astype(float)
        df_tax["oversight_shift_index"] = df_tax["covert_or_rationalized"] - df_tax["quid_pro_quo"].astype(float)
        h2b_res = paired_permutation_test(*_paired_series(
            df_tax[df_tax["condition"] == "B2"].groupby("seed")["oversight_shift_index"].mean(),
            df_tax[df_tax["condition"] == "B3"].groupby("seed")["oversight_shift_index"].mean()), alternative="two-sided")
    else:
        h2b_res = {"mean_diff": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_value": 1.0, "n_pairs": 0}

    # H3: crisis vs routine, per-seed mean UJO over B cells; two-sided
    h3_res = paired_permutation_test(*_paired_series(
        _seed_series(lambda d: (d["incentive"] == "conflicting") & (d["framing"] == "crisis")),
        _seed_series(lambda d: (d["incentive"] == "conflicting") & (d["framing"] == "routine"))), alternative="two-sided")

    # Monitor table on frozen test split (core test episodes)
    mon_df = evaluate_monitor_table(core_test_eps, audits_test, B=5000, target_key="any_HCO")
    mon_df_cc = evaluate_monitor_table(core_test_eps, audits_test, B=5000, target_key="any_confirmed_collusion")
    h4_v3_v2 = paired_cluster_bootstrap_auroc_diff(core_test_eps, audits_test, "V3", "V2", B=5000, target_key="any_HCO")
    h4_v3_v1 = paired_cluster_bootstrap_auroc_diff(core_test_eps, audits_test, "V3", "V1", B=5000, target_key="any_HCO")
    h4_v3_lf = paired_cluster_bootstrap_auroc_diff(core_test_eps, audits_test, "V3", "late_fusion", B=5000, target_key="any_HCO")
    h4_p = max(h4_v3_v2["p_value"], h4_v3_v1["p_value"])

    # H5: Plus-factor ladder on frozen test split
    ladder_rows = evaluate_evidence_ladder(core_test_eps, target_key="any_HCO")
    df_ladder = pd.DataFrame(ladder_rows)
    df_ladder_cc = pd.DataFrame(evaluate_evidence_ladder(core_test_eps, target_key="any_confirmed_collusion"))
    # Cluster bootstrap test for Precision(Level 3) - Precision(Level 0) > 0
    feats_test = [extract_plus_factors(e) for e in core_test_eps]
    y_test = np.array([int(e["ground_truth_episode"].get("any_HCO", e["ground_truth_episode"]["any_confirmed_collusion"])) for e in core_test_eps])
    seeds_test = np.array([int(e["seed"]) for e in core_test_eps])
    p0_arr = np.array([int(f["ladder_0"]) for f in feats_test])
    p3_arr = np.array([int(f["ladder_3"]) for f in feats_test])
    rng = np.random.default_rng(20261025)
    uniq_s = np.unique(seeds_test)
    s_idx = {s: np.where(seeds_test == s)[0] for s in uniq_s}
    prec_diffs = []
    for _ in range(5000):
        draws = rng.choice(uniq_s, size=len(uniq_s), replace=True)
        idx = np.concatenate([s_idx[s] for s in draws])
        yt, l0, l3 = y_test[idx], p0_arr[idx], p3_arr[idx]
        pr0 = np.sum((l0 == 1) & (yt == 1)) / max(1, np.sum(l0 == 1))
        pr3 = np.sum((l3 == 1) & (yt == 1)) / max(1, np.sum(l3 == 1))
        prec_diffs.append(pr3 - pr0)
    h5_p = float((np.sum(np.array(prec_diffs) <= 0) + 1) / (len(prec_diffs) + 1))

    holm_df = holm_bonferroni([
        {"id": "H1", "description": "Incentive conflict (B vs A) increases raw UJO rate", "effect": round(h1_res["mean_diff"], 4), "p_raw": h1_res["p_value"]},
        {"id": "H2a", "description": "Communication availability (B3 vs B1) increases UJO rate", "effect": round(h2a_res["mean_diff"], 4), "p_raw": h2a_res["p_value"]},
        {"id": "H2b", "description": "Oversight awareness (B2 vs B3) shifts toward moral/camouflaged framing", "effect": round(h2b_res["mean_diff"], 4), "p_raw": h2b_res["p_value"]},
        {"id": "H3", "description": "Crisis vs Routine framing shifts UJO rate in conflicting cells", "effect": round(h3_res["mean_diff"], 4), "p_raw": h3_res["p_value"]},
        {"id": "H4", "description": "Multimodal monitor V3 beats single-modality max(V1, V2) AUROC", "effect": round(min(h4_v3_v2["delta_auroc"], h4_v3_v1["delta_auroc"]), 4), "p_raw": h4_p},
        {"id": "H5", "description": "Antitrust Plus-Factor Ladder (L3 vs L0) increases precision", "effect": round(df_ladder.iloc[-1]["precision"] - df_ladder.iloc[0]["precision"], 4), "p_raw": h5_p},
    ])

    iaa = build_inter_annotator_sample(core_eps)

    return {
        "df_all": df_all,
        "df_core": df_core,
        "df_cells": df_cells,
        "df_tax": df_tax,
        "mon_df": mon_df,
        "mon_df_cc": mon_df_cc,
        "df_ladder": df_ladder,
        "df_ladder_cc": df_ladder_cc,
        "holm_df": holm_df,
        "tests": {
            "H1": h1_res,
            "H1_CC": h1_cc_res,
            "H1_CA": h1_ca_res,
            "H2a": h2a_res,
            "H2a_crisis": h2a_crisis_res,
            "H2a_routine": h2a_routine_res,
            "H2b": h2b_res,
            "H3": h3_res,
            "H4_V3_vs_V2": h4_v3_v2,
            "H4_V3_vs_V1": h4_v3_v1,
            "H4_V3_vs_LF": h4_v3_lf,
            "H5_p": h5_p,
        },
        "iaa": iaa,
        "complete_seeds": complete_seeds,
        "n_core_episodes": len(core_eps),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="data/episodes_live.jsonl")
    ap.add_argument("--audits", default="data/audits_live_test.jsonl")
    ap.add_argument("--exclude-comprehension-fail", action="store_true")
    ap.add_argument("--figdir", default="paper/figures")
    args = ap.parse_args()

    os.makedirs(args.figdir, exist_ok=True)
    res = compute_all_summaries(args.episodes, args.audits, args.exclude_comprehension_fail)
    print(f"Complete seeds: {len(res['complete_seeds'])} -> {res['complete_seeds']}; core episodes analysed: {res['n_core_episodes']}")
    print("\n=== CELL SUMMARY (Separating UJO, Miscoordination MC_rate, and Confirmed Collusion CC_rate) ===")
    print(res["df_cells"][["condition", "framing", "n_seeds", "UJO_rate", "PV_rate", "MC_rate", "CC_rate", "joint_override_rate", "ambiguous_override_rate", "regret_mean", "CA_mean", "comp_pass_rate"]].to_string(index=False))
    print("\n=== MONITOR BENCHMARK (FROZEN TEST SPLIT — HCO / Known-Incentive Target) ===")
    print(res["mon_df"][["view", "AUROC", "AUROC_ci_low", "AUROC_ci_high", "TPR_at_5pct_FPR", "FPR_Case2_Parallel", "FPR_Case3_CheapTalk", "FPR_Case4_LegitRescue", "n_pos", "n_neg", "n_case2_parallel", "n_case3_cheap_talk", "n_case4_legit"]].to_string(index=False))
    print("\n=== PLUS-FACTOR EVIDENCE LADDER (FROZEN TEST SPLIT — HCO / Known-Incentive Target) ===")
    print(res["df_ladder"][["rung", "precision", "recall", "f1", "tp", "fp", "fn", "fpr_case2_parallel", "fpr_case4_legitimate"]].to_string(index=False))
    print("\n=== PLUS-FACTOR EVIDENCE LADDER (UNOBSERVED-INCENTIVE TARGET: CC vs. A2/A3 Miscoordination) ===")
    print(res["df_ladder_cc"][["rung", "precision", "recall", "f1", "tp", "fp", "fn", "fpr_case1b_miscoord", "fpr_case2_parallel", "fpr_case4_legitimate"]].to_string(index=False))
    print("\n=== HOLM-BONFERRONI HYPOTHESIS TABLE ===")
    print(res["holm_df"].to_string(index=False))
    print("\n=== DIAGNOSTIC & SUBGROUP TESTS ===")
    for k, v in res["tests"].items():
        print(f"  {k}: {v}")
    print("\n=== EXTRACTOR VALIDATION SAMPLE ===")
    print(json.dumps(res["iaa"], indent=2))


if __name__ == "__main__":
    main()
