"""Builds and executes the 4 Jury-Facing Jupyter Notebooks in `analysis/*.ipynb`
and exports all publication figures to `paper/figures/*.png`.

Notebooks generated & executed:
  1. analysis/01_game_theory_and_bot_verification.ipynb
  2. analysis/02_track21_crisis_collusion_analysis.ipynb
  3. analysis/03_track22_multimodal_monitors.ipynb
  4. analysis/04_track23_plus_factors_and_case_studies.ipynb
"""
from __future__ import annotations

import os
import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


SETUP_CODE = """%matplotlib inline
import os, sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path.cwd().resolve()
if ROOT.name == "analysis":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 240,
    "font.size": 10.5,
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": True,
})
FIGDIR = ROOT / "paper" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
print(f"Workspace root: {ROOT}")
"""


def build_nb01() -> nbformat.NotebookNode:
    nb = new_notebook()
    nb.cells = [
        new_markdown_cell(
            r"""# 01 — Game-Theoretic Foundations, Exact Bayesian Posteriors & Scripted-Bot Verification
**Apart Research AI Collusion Research Sprint (Oct 23–25, 2026)**  
*When Crisis Communication Becomes Collusion: Detecting coordinated rule-breaking between LLM dispatch agents, with labels computed by the environment (no LLM in the labelling loop)*

---

### Purpose of This Notebook
Before analyzing a single LLM transcript, we mathematically verify every invariant of the **Dual-Key Dispatch Game (§2)**:
1. **Exact Bayesian Posteriors ($p(\sigma_A, \sigma_B, \pi_t)$):** Verify that the ex-ante socially optimal override threshold is $p^* = \frac{36}{60} = 0.60$, that the hidden shelter state is compromised in $25.0\%$ of rounds, and that bilateral warnings ($\sigma_A = \text{WARN}, \sigma_B = \text{WARN}$) occur in **$17.7\%$ of rounds** ($p \ge 0.780 > 0.60$), forming **Case ④ (Legitimate Emergency Coordination)**.
2. **Strict Risk Dominance in the Stag Hunt (§2.4):** Verify the normal-form payoff matrix under conflicting incentives ($\alpha = 0.85$), proving that $(\text{Override}, \text{Override}) = (46.0, 46.0)$ is **payoff-dominant** while $(\text{Default}, \text{Default}) = (31.0, 31.0)$ is **strictly risk-dominant** ($q^* = \frac{20}{35} = \frac{4}{7} \approx 0.571 > 0.50$, Nash product $20^2 = 400 > 15^2 = 225$) and maximin.
3. **Zero-Leakage Feedback & Monitor Whitelist (§3 & §6.1):** Verify that round-by-round feedback exposes only $L_i \in \{16, 40\}$ and never leaks the hidden state $s_t$.
4. **Deterministic Scripted-Bot Controls (§4.3):** Evaluate the 5 deterministic bot archetypes (`Honest-Bayesian`, `Always-Collude`, `Silent-Parallel`, `Pretextual-Liar`, `Cheap-Talker`) across $N = 30$ paired seeds each (150 episodes, 750 rounds)."""
        ),
        new_code_cell(SETUP_CODE),
        new_markdown_cell("## 1. Exact Bayesian Posterior Table & Signal Pattern Frequencies (§2.3)"),
        new_code_cell(
            """from src.sim.environment import (
    NOISE, posterior_table, marginal_posterior, payoff_matrix,
    risk_dominance_threshold, WARN, CLEAR, ALARM, QUIET
)

df_post = pd.DataFrame(posterior_table(NOISE["low"]))
df_post["pattern"] = df_post["sigma_A"].str[:4] + " + " + df_post["sigma_B"].str[:4] + " | " + df_post["sensor"]
df_post["freq_pct"] = (df_post["frequency"] * 100).round(2)
df_post["likelihood_ratio"] = df_post["likelihood_ratio"].round(2)
df_post["posterior"] = df_post["posterior"].round(4)

print(f"Sum of all 8 pattern probabilities: {df_post['frequency'].sum():.6f}")
print(f"Combined bilateral-warning (WARN+WARN) rate: {df_post[df_post['label']=='justified']['frequency'].sum()*100:.2f}% (spec: 17.7%)")
print(f"Single-agent marginal P(comp | own=WARN, sensor=QUIET) = {marginal_posterior(WARN, QUIET):.3f} (< 0.60 -> requires signal pooling!)")
display(df_post[["sigma_A", "sigma_B", "sensor", "freq_pct", "likelihood_ratio", "posterior", "label"]])"""
        ),
        new_markdown_cell("## 2. Normal-Form Payoff Matrix & Strict Risk Dominance Proof (§2.4)"),
        new_code_cell(
            """m_conf = payoff_matrix("conflicting", "normal")
m_alig = payoff_matrix("aligned", "normal")
q_star = risk_dominance_threshold("conflicting", "normal")

payoff_df = pd.DataFrame([
    {"Incentive": "Conflicting (B, alpha=0.85)", "Action Pair": "(CENTRAL, CENTRAL)", "Payoff (A, B)": m_conf[("CENTRAL", "CENTRAL")], "Public Lives U_pub": 116},
    {"Incentive": "Conflicting (B, alpha=0.85)", "Action Pair": "(CENTRAL, LOCAL-SPLIT)", "Payoff (A, B)": m_conf[("CENTRAL", "LOCAL-SPLIT")], "Public Lives U_pub": 116},
    {"Incentive": "Conflicting (B, alpha=0.85)", "Action Pair": "(LOCAL-SPLIT, CENTRAL)", "Payoff (A, B)": m_conf[("LOCAL-SPLIT", "CENTRAL")], "Public Lives U_pub": 116},
    {"Incentive": "Conflicting (B, alpha=0.85)", "Action Pair": "(LOCAL-SPLIT, LOCAL-SPLIT)", "Payoff (A, B)": m_conf[("LOCAL-SPLIT", "LOCAL-SPLIT")], "Public Lives U_pub": 80},
    {"Incentive": "Aligned (A, alpha=0.00)", "Action Pair": "(CENTRAL, CENTRAL)", "Payoff (A, B)": m_alig[("CENTRAL", "CENTRAL")], "Public Lives U_pub": 116},
    {"Incentive": "Aligned (A, alpha=0.00)", "Action Pair": "(LOCAL-SPLIT, LOCAL-SPLIT)", "Payoff (A, B)": m_alig[("LOCAL-SPLIT", "LOCAL-SPLIT")], "Public Lives U_pub": 80},
])
print(f"Risk-dominance belief threshold q* = {q_star:.6f} = 4/7 (> 0.50 -> Default is strictly risk-dominant)")
print(f"Harsanyi-Selten Nash deviation product: Default = (31-11)^2 = {(31-11)**2} vs Override = (46-31)^2 = {(46-31)**2}")
display(payoff_df)"""
        ),
        new_markdown_cell("## 3. Deterministic Scripted-Bot Verification (`data/bot_episodes.jsonl`, 150 Episodes)"),
        new_code_cell(
            """with open(ROOT / "data" / "bot_episodes.jsonl") as f:
    bot_eps = [json.loads(line) for line in f if line.strip()]

bot_rows = []
for e in bot_eps:
    for r in e["rounds"]:
        g = r["ground_truth"]
        ex = r["extraction"].get("per_agent", {})
        lied = any(ex.get(a, {}).get("claimed_signal", "none") != "none" and ex.get(a, {}).get("claimed_signal", "none") != r["private"]["signal"][a] for a in ("A", "B"))
        div = any(ex.get(a, {}).get("final_stance", "none") != "none" and ex.get(a, {}).get("final_stance", "none") != r["actions"][a]["priority"] for a in ("A", "B"))
        bot_rows.append({
            "bot": e["condition"],
            "case_type": g["case_type"],
            "UJO": int(g["UJO"]),
            "confirmed_collusion": int(g["confirmed_collusion"]),
            "regret": g["regret"],
            "any_lie": int(lied),
            "any_divergence": int(div),
        })
df_bots = pd.DataFrame(bot_rows)
summary_bots = df_bots.groupby("bot").agg(
    rounds=("UJO", "count"),
    UJO_rounds=("UJO", "sum"),
    Confirmed_Collusion_rounds=("confirmed_collusion", "sum"),
    Case1_Confirmed=("case_type", lambda s: (s == "confirmed_collusion").sum()),
    Case2_Parallel=("case_type", lambda s: (s == "parallel_conduct").sum()),
    Case3_CheapTalk=("case_type", lambda s: (s == "cheap_talk").sum()),
    Case4_Legit=("case_type", lambda s: (s == "legitimate_coordination").sum()),
    Lie_rounds=("any_lie", "sum"),
    Divergence_rounds=("any_divergence", "sum"),
    Mean_Regret=("regret", "mean"),
).reset_index()
display(summary_bots)"""
        ),
        new_markdown_cell("## 4. Publication Figure 0: Bayesian Posteriors, Stag Hunt Risk Dominance & Bot Controls"),
        new_code_cell(
            """fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))

# Panel A: Bayesian posteriors by signal pattern
ax = axes[0]
df_p_uniq = df_post.drop_duplicates(subset=["posterior"]).sort_values("posterior", ascending=True)
labels_p = [
    "(CLEAR, CLEAR | QUIET)",
    "(CLEAR, CLEAR | ALARM)",
    "(WARN, CLEAR | QUIET)",
    "(WARN, CLEAR | ALARM)",
    "(WARN, WARN | QUIET)",
    "(WARN, WARN | ALARM)",
]
colors_p = ["#2b5c8f" if p < 0.5 else ("#d9822b" if p <= 0.7 else "#2e8b57") for p in df_p_uniq["posterior"]]
bars = ax.barh(labels_p, df_p_uniq["posterior"], color=colors_p, edgecolor="black", height=0.62)
ax.axvspan(0.50, 0.70, color="#f4d06f", alpha=0.28, label="Ambiguous Gray Zone [0.50, 0.70]")
ax.axvline(0.60, color="#c0392b", linestyle="--", linewidth=1.8, label="Optimal Threshold p* = 0.60")
for bar, p in zip(bars, df_p_uniq["posterior"]):
    ax.text(min(0.98, p + 0.02), bar.get_y() + bar.get_height()/2, f"{p:.3f}", va="center", fontsize=9, fontweight="bold")
ax.set_xlim(0, 1.12)
ax.set_xlabel("Posterior P(compromised | signals)")
ax.set_title("A. Exact Bayesian Posteriors (Low-Noise)")
ax.legend(loc="lower right", fontsize=8)

# Panel B: Stag Hunt expected payoff vs partner override probability q
ax = axes[1]
q_grid = np.linspace(0, 1, 200)
ev_default = np.full_like(q_grid, 31.0)
ev_override = 11.0 + 35.0 * q_grid
ax.plot(q_grid, ev_default, label="Submit CENTRAL (Default): 31.0", color="#1f77b4", linewidth=2.4)
ax.plot(q_grid, ev_override, label="Submit LOCAL-SPLIT (Override): 11 + 35q", color="#d62728", linewidth=2.4)
ax.axvline(4/7, color="#333333", linestyle=":", linewidth=1.6, label=f"Risk-Dominance q* = 4/7 ≈ {4/7:.3f}")
ax.scatter([4/7], [31.0], color="black", zorder=5, s=45)
ax.fill_between(q_grid, ev_override, ev_default, where=(q_grid < 4/7), color="#1f77b4", alpha=0.12, label="Default Basin (4/7 = 57.1%)")
ax.fill_between(q_grid, ev_override, ev_default, where=(q_grid >= 4/7), color="#d62728", alpha=0.12, label="Collusive Basin (3/7 = 42.9%)")
ax.set_xlabel("Belief q = P(Partner files LOCAL-SPLIT)")
ax.set_ylabel("Expected Round Payoff E[R_i]")
ax.set_title("B. Stag Hunt Strict Risk Dominance (alpha=0.85)")
ax.legend(loc="upper left", fontsize=8)

# Panel C: Scripted Bot Case Typology Verification
ax = axes[2]
bot_order = ["BOT-HONEST", "BOT-COLLUDE", "BOT-LIAR", "BOT-SILENT", "BOT-CHEAP"]
sb = summary_bots.set_index("bot").loc[bot_order]
bottom = np.zeros(len(bot_order))
case_cols = [
    ("Case1_Confirmed", "Case ① Confirmed Collusion", "#c0392b"),
    ("Case2_Parallel", "Case ② Silent Parallel Conduct", "#e67e22"),
    ("Case3_CheapTalk", "Case ③ Cheap Talk (No UJO)", "#8e44ad"),
    ("Case4_Legit", "Case ④ Legitimate Coordination", "#27ae60"),
]
for col, lbl, clr in case_cols:
    vals = sb[col].values / sb["rounds"].values * 100
    ax.bar(bot_order, vals, bottom=bottom, label=lbl, color=clr, edgecolor="white", width=0.6)
    bottom += vals
ax.set_ylabel("% of Rounds (across 30 paired seeds)")
ax.set_title("C. Scripted-Bot Ground-Truth Case Verification")
ax.tick_params(axis="x", rotation=20)
ax.legend(loc="upper right", fontsize=8)

plt.tight_layout()
out_fig = FIGDIR / "fig0_game_theory_and_bots.png"
plt.savefig(out_fig, bbox_inches="tight")
plt.show()
print(f"Saved figure -> {out_fig}")"""
        ),
    ]
    return nb


def build_nb02() -> nbformat.NotebookNode:
    nb = new_notebook()
    nb.cells = [
        new_markdown_cell(
            """# 02 — Track 2.1: Crisis Collusion, Communication vs. Oversight & Deception Mechanics
**Apart Research AI Collusion Research Sprint (Oct 23–25, 2026)**

---

### Research Questions Addressed in This Notebook (Track 2.1 Core + Ablations)
1. **H1 (Incentive Conflict Effect):** Does shifting from aligned public-good incentives (`A1–A3`, $\alpha=0$) to conflicting local-sector incentives (`B1–B3`, $\alpha=0.85$) induce **Unjustified Joint Overrides (`UJO`)** and **`Confirmed Collusion` (Case ①)**?
2. **H2a (Communication Availability, `B3 vs. B1`):** How much of the welfare harm in `B3` (`Unmonitored-Chat`) is driven by **explicit communicative collusion** above the tacit focal-point baseline in `B1` (`No-Chat`)?
3. **H2b (Oversight Awareness, `B2 vs. B3`):** When agents know an auditor reads their chat (`B2` `Monitored-Chat`), do they stop colluding, or do they shift from explicit `quid_pro_quo` bargaining to **camouflaged hazard signaling** and **moral rationalization**?
4. **H3 (Crisis vs. Routine Framing):** Does high-stakes emergency earthquake framing (`crisis`) amplify `UJO` rates and moral rationalization compared to mathematically identical municipal road-maintenance scheduling (`routine`)?
5. **Deliverable 4B & Ablations:** What are the exact rates of **Speech–Action Divergence**, **Verifiable Signal Lying**, **High-Noise Deniability (`0.70/0.30`)**, and **Prompt-Paraphrase Robustness**?"""
        ),
        new_code_cell(SETUP_CODE),
        new_markdown_cell("## 1. Episode-Level Summary Table Across the 12 Core Cells + Positive Control"),
        new_code_cell(
            """from src.analysis.stats import (
    episodes_to_dataframe, cluster_bootstrap_ci, paired_permutation_test,
    paired_mcnemar_test, fit_clustered_regression
)
from src.auditor.extractor import deception_rows, refresh_episode_extraction
from src.auditor.taxonomy import tag_episode_messages, TAXONOMY_TAGS

with open(ROOT / "data" / "episodes_live.jsonl") as f:
    all_eps = [refresh_episode_extraction(json.loads(line)) for line in f if line.strip()]

# Keep balanced paired seeds where all 12 core cells have completed
cells_by_seed = {}
for e in all_eps:
    if e["condition"] in ("A1", "A2", "A3", "B1", "B2", "B3") and e.get("noise", "low") == "low" and e.get("paraphrase", 0) == 0:
        cells_by_seed.setdefault(int(e["seed"]), set()).add((e["condition"], e.get("framing", "crisis")))
complete_seeds = sorted(s for s, c in cells_by_seed.items() if len(c) == 12)
print(f"Loaded {len(all_eps)} live Modal LLM episodes ({len(complete_seeds)} complete 12-cell paired seeds: {complete_seeds})")

df_all = episodes_to_dataframe(all_eps)
df_core = df_all[
    df_all["condition"].isin(["A1", "A2", "A3", "B1", "B2", "B3"])
    & (df_all["noise"] == "low")
    & (df_all["paraphrase"] == 0)
    & (df_all["seed"].isin(complete_seeds))
].copy()

summary_rows = []
for (cond, framing), grp in df_core.groupby(["condition", "framing"]):
    ujo_m, ujo_lo, ujo_hi = cluster_bootstrap_ci(grp["UJO_rate"].values)
    pv_m, pv_lo, pv_hi = cluster_bootstrap_ci(grp["protocol_violation_rate"].values)
    mc_m, mc_lo, mc_hi = cluster_bootstrap_ci(grp["miscoordination_rate"].values)
    cc_m, cc_lo, cc_hi = cluster_bootstrap_ci(grp["confirmed_collusion_rate"].values)
    reg_m, reg_lo, reg_hi = cluster_bootstrap_ci(grp["total_regret"].values)
    ca_m, ca_lo, ca_hi = cluster_bootstrap_ci(grp["coalition_advantage_mean"].values)
    summary_rows.append({
        "Condition": cond,
        "Framing": framing,
        "N_Seeds": len(grp),
        "UJO Rate [95% CI]": f"{ujo_m*100:.1f}% [{ujo_lo*100:.1f}, {ujo_hi*100:.1f}]",
        "Protocol Violation (p<=0.60)": f"{pv_m*100:.1f}% [{pv_lo*100:.1f}, {pv_hi*100:.1f}]",
        "Miscoordination (Case ①b)": f"{mc_m*100:.1f}% [{mc_lo*100:.1f}, {mc_hi*100:.1f}]",
        "Confirmed Collusion (Case ①)": f"{cc_m*100:.1f}% [{cc_lo*100:.1f}, {cc_hi*100:.1f}]",
        "Welfare Loss ΔW (Lives)": f"{reg_m:.1f} [{reg_lo:.1f}, {reg_hi:.1f}]",
        "Coalition Advantage CA": f"{ca_m:+.1f} [{ca_lo:+.1f}, {ca_hi:+.1f}]",
        "UJO_val": ujo_m, "UJO_lo": ujo_lo, "UJO_hi": ujo_hi,
        "PV_val": pv_m,
        "MC_val": mc_m, "MC_lo": mc_lo, "MC_hi": mc_hi,
        "CC_val": cc_m, "CC_lo": cc_lo, "CC_hi": cc_hi,
        "Regret_val": reg_m, "CA_val": ca_m,
    })

print(f"Comprehension Pass Rate across analysed core episodes: {df_core['comp_pass_both'].mean()*100:.1f}%")
df_summary = pd.DataFrame(summary_rows)
display(df_summary[["Condition", "Framing", "N_Seeds", "UJO Rate [95% CI]", "Protocol Violation (p<=0.60)", "Miscoordination (Case ①b)", "Confirmed Collusion (Case ①)", "Welfare Loss ΔW (Lives)", "Coalition Advantage CA"]])"""
        ),
        new_markdown_cell("## 2. Confirmatory Hypothesis Tests H1, H2a, H2b & H3 (Seed-Paired Permutations)"),
        new_code_cell(
            """def paired_vec(cond, framing, col="UJO_rate"):
    return df_core[(df_core["condition"] == cond) & (df_core["framing"] == framing)].sort_values("seed")[col].values

# H1: Conflicting (B) vs Aligned (A) across each regime and pooled
h1_b1_a1 = paired_permutation_test(paired_vec("B1", "crisis"), paired_vec("A1", "crisis"), "greater")
h1_b2_a2 = paired_permutation_test(paired_vec("B2", "crisis"), paired_vec("A2", "crisis"), "greater")
h1_b3_a3 = paired_permutation_test(paired_vec("B3", "crisis"), paired_vec("A3", "crisis"), "greater")

# H2a: B3 (Unmonitored-Chat) vs B1 (No-Chat)
h2a_crisis = paired_permutation_test(paired_vec("B3", "crisis"), paired_vec("B1", "crisis"), "greater")
h2a_routine = paired_permutation_test(paired_vec("B3", "routine"), paired_vec("B1", "routine"), "greater")

# H2b: B2 (Monitored-Chat) vs B3 (Unmonitored-Chat)
h2b_ujo = paired_permutation_test(paired_vec("B2", "crisis"), paired_vec("B3", "crisis"), "two-sided")

# H3: Crisis vs Routine framing across conflicting B-cells
b_crisis_seed = df_core[(df_core["incentive"] == "conflicting") & (df_core["framing"] == "crisis")].groupby("seed")["UJO_rate"].mean().sort_index().values
b_routine_seed = df_core[(df_core["incentive"] == "conflicting") & (df_core["framing"] == "routine")].groupby("seed")["UJO_rate"].mean().sort_index().values
h3_framing = paired_permutation_test(b_crisis_seed, b_routine_seed, "two-sided")

# Coalition Advantage (CA): Conflicting (B) vs Aligned (A) — separating Collusion (CA > 0) from Miscoordination (CA < 0)
ca_b_seed = df_core[df_core["incentive"] == "conflicting"].groupby("seed")["coalition_advantage_mean"].mean().sort_index().values
ca_a_seed = df_core[df_core["incentive"] == "aligned"].groupby("seed")["coalition_advantage_mean"].mean().sort_index().values
h1_ca = paired_permutation_test(ca_b_seed, ca_a_seed, "greater")

tests_table = pd.DataFrame([
    {"Hypothesis": "H1 (B1 vs A1, No-Chat Crisis)", "Effect": f"{h1_b1_a1['mean_diff']*100:+.1f}% UJO", "95% Bootstrap CI": f"[{h1_b1_a1['ci_low']*100:+.1f}%, {h1_b1_a1['ci_high']*100:+.1f}%]", "p-value": h1_b1_a1["p_value"]},
    {"Hypothesis": "H1 (B2 vs A2, Monitored Crisis)", "Effect": f"{h1_b2_a2['mean_diff']*100:+.1f}% UJO", "95% Bootstrap CI": f"[{h1_b2_a2['ci_low']*100:+.1f}%, {h1_b2_a2['ci_high']*100:+.1f}%]", "p-value": h1_b2_a2["p_value"]},
    {"Hypothesis": "H1 (B3 vs A3, Unmonitored Crisis)", "Effect": f"{h1_b3_a3['mean_diff']*100:+.1f}% UJO", "95% Bootstrap CI": f"[{h1_b3_a3['ci_low']*100:+.1f}%, {h1_b3_a3['ci_high']*100:+.1f}%]", "p-value": h1_b3_a3["p_value"]},
    {"Hypothesis": "H1_CA (B vs A Coalition Advantage)", "Effect": f"{h1_ca['mean_diff']:+.1f} pts/ep", "95% Bootstrap CI": f"[{h1_ca['ci_low']:+.1f}, {h1_ca['ci_high']:+.1f}]", "p-value": h1_ca["p_value"]},
    {"Hypothesis": "H2a (B3 vs B1, Comm. Availability Crisis)", "Effect": f"{h2a_crisis['mean_diff']*100:+.1f}% UJO", "95% Bootstrap CI": f"[{h2a_crisis['ci_low']*100:+.1f}%, {h2a_crisis['ci_high']*100:+.1f}%]", "p-value": h2a_crisis["p_value"]},
    {"Hypothesis": "H2b (B2 vs B3, Oversight Awareness on UJO)", "Effect": f"{h2b_ujo['mean_diff']*100:+.1f}% UJO", "95% Bootstrap CI": f"[{h2b_ujo['ci_low']*100:+.1f}%, {h2b_ujo['ci_high']*100:+.1f}%]", "p-value": h2b_ujo["p_value"]},
    {"Hypothesis": "H3 (Crisis vs Routine in B-Cells)", "Effect": f"{h3_framing['mean_diff']*100:+.1f}% UJO", "95% Bootstrap CI": f"[{h3_framing['ci_low']*100:+.1f}%, {h3_framing['ci_high']*100:+.1f}%]", "p-value": h3_framing["p_value"]},
])
display(tests_table)

print("\\nSeed-Clustered Regression Check (UJO_rate ~ C(incentive) + C(regime) + C(framing)):")
display(fit_clustered_regression(df_core, "UJO_rate"))"""
        ),
        new_markdown_cell("## 3. Deliverable 4A & 4B: Linguistic Taxonomy Shift (`B2 vs. B3`), Speech–Action Divergence & Verifiable Signal Lying"),
        new_code_cell(
            """core_eps = [
    e for e in all_eps
    if e["condition"] in ("A2", "A3", "B2", "B3")
    and e.get("noise", "low") == "low"
    and e.get("paraphrase", 0) == 0
    and int(e["seed"]) in complete_seeds
]

tax_rows = []
dec_rows = []
for e in core_eps:
    tax_rows.extend(tag_episode_messages(e))
    dec_rows.extend(deception_rows(e))

df_tax = pd.DataFrame(tax_rows)
df_dec = pd.DataFrame(dec_rows)

tax_summary = (
    df_tax[df_tax["framing"] == "crisis"]
    .groupby("condition")[list(TAXONOMY_TAGS)]
    .mean()
    .mul(100)
    .round(1)
    .reset_index()
)
print("Linguistic Taxonomy Tag Prevalence (% of messages in Crisis Framing):")
display(tax_summary)

dec_summary = (
    df_dec[df_dec["framing"] == "crisis"]
    .groupby("condition")
    .agg(
        Agent_Rounds=("lying", "count"),
        Signal_Lying_Rate_pct=("lying", lambda s: round(s.mean()*100, 1)),
        Speech_Action_Divergence_pct=("divergence", lambda s: round(s.mean()*100, 1)),
    )
    .reset_index()
)
print("Deception & Speech-Action Divergence Summary (Crisis Framing):")
display(dec_summary)"""
        ),
        new_markdown_cell("## 4. Publication Figures 1 & 2 (Exported to `paper/figures/`)"),
        new_code_cell(
            """# Figure 1: Track 2.1 UJO vs Miscoordination vs Confirmed Collusion & Welfare Regret / Coalition Advantage
from IPython.display import Image
fig, axes = plt.subplots(1, 2, figsize=(14.0, 4.6))

sub_c = df_summary[df_summary["Framing"] == "crisis"].set_index("Condition").loc[["A1", "A2", "A3", "B1", "B2", "B3"]]
x = np.arange(len(sub_c))
w = 0.26

ax = axes[0]
ax.bar(x - w, sub_c["UJO_val"]*100, width=w, label="Unjustified Joint Override (UJO — Clear-Cut Harm)", color="#e67e22", edgecolor="black")
ax.bar(x, sub_c["MC_val"]*100, width=w, label="Miscoordination / Precautionary Cascade (Case ①b)", color="#2980b9", edgecolor="black")
ax.bar(x + w, sub_c["CC_val"]*100, width=w, label="Confirmed Collusion (Case ① — Conflicting CA > 0)", color="#c0392b", edgecolor="black")
ax.errorbar(x - w, sub_c["UJO_val"]*100, yerr=[(sub_c["UJO_val"]-sub_c["UJO_lo"])*100, (sub_c["UJO_hi"]-sub_c["UJO_val"])*100], fmt="none", ecolor="black", capsize=2.5)
ax.errorbar(x, sub_c["MC_val"]*100, yerr=[(sub_c["MC_val"]-sub_c["MC_lo"])*100, (sub_c["MC_hi"]-sub_c["MC_val"])*100], fmt="none", ecolor="black", capsize=2.5)
ax.errorbar(x + w, sub_c["CC_val"]*100, yerr=[(sub_c["CC_val"]-sub_c["CC_lo"])*100, (sub_c["CC_hi"]-sub_c["CC_val"])*100], fmt="none", ecolor="black", capsize=2.5)
ax.set_xticks(x)
ax.set_xticklabels(["A1\\n(No-Chat)", "A2\\n(Monitored)", "A3\\n(Unmonitored)", "B1\\n(No-Chat)", "B2\\n(Monitored)", "B3\\n(Unmonitored)"])
ax.set_ylabel("% of eligible rounds (p < 0.50)")
ax.set_ylim(0, 125)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.set_title("A. UJO vs. Miscoordination (①b) vs. Confirmed Collusion (①), crisis")
ax.legend(loc="upper left", fontsize=8.0, ncol=1)

ax = axes[1]
w2 = 0.36
ax.bar(x - w2/2, sub_c["Regret_val"], width=w2, label="Public Welfare Loss ΔW (Lives Lost / Ep)", color="#34495e", edgecolor="black")
ax.bar(x + w2/2, sub_c["CA_val"], width=w2, label="Private Coalition Advantage CA (Points / Ep)", color="#16a085", edgecolor="black")
ax.axhline(0, color="black", linewidth=0.9)
ax.set_xticks(x)
ax.set_xticklabels(["A1", "A2", "A3", "B1", "B2", "B3"])
ax.set_ylabel("Episode-Total Lives Lost / Payoff Gain")
ax.set_title("B. Public Welfare Loss (ΔW) & Coalition Advantage (CA)")
ax.legend(loc="lower left", fontsize=8.5)

plt.tight_layout()
f1_path = FIGDIR / "fig1_track21_ujo_vs_collusion.png"
plt.savefig(f1_path, bbox_inches="tight")
plt.close(fig)
display(Image(filename=str(f1_path)))

# Figure 2: Crisis vs Routine Framing (H3) & Linguistic Taxonomy Shift (H2b)
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.4))

ax = axes[0]
b_conds = ["B1", "B2", "B3"]
sc_cr = df_summary[(df_summary["Framing"] == "crisis") & (df_summary["Condition"].isin(b_conds))].set_index("Condition").loc[b_conds]
sc_rt = df_summary[(df_summary["Framing"] == "routine") & (df_summary["Condition"].isin(b_conds))].set_index("Condition").loc[b_conds]
xb = np.arange(len(b_conds))
ax.bar(xb - w/2, sc_cr["UJO_val"]*100, width=w, label="Crisis Framing (Earthquake Rescue)", color="#c0392b", edgecolor="black")
ax.bar(xb + w/2, sc_rt["UJO_val"]*100, width=w, label="Routine Framing (Road Maintenance)", color="#5dade2", edgecolor="black")
ax.set_xticks(xb)
ax.set_xticklabels(["B1 (No-Chat)", "B2 (Monitored-Chat)", "B3 (Unmonitored-Chat)"])
ax.set_ylabel("UJO Rate (%)")
ax.set_ylim(0, 95)
ax.set_title("A. Crisis vs. Routine Framing Effect (H3, Paired Seeds)")
ax.legend(loc="upper left", fontsize=8.5)

ax = axes[1]
tx = tax_summary.set_index("condition").loc[["A2", "A3", "B2", "B3"]]
xt = np.arange(4)
bw = 0.24
ax.bar(xt - bw, tx["quid_pro_quo"], width=bw, label="Explicit Quid-Pro-Quo", color="#c0392b", edgecolor="black")
ax.bar(xt, tx["camouflaged_signaling"], width=bw, label="Camouflaged Hazard Signaling", color="#e67e22", edgecolor="black")
ax.bar(xt + bw, tx["moral_rationalization"], width=bw, label="Moral Rationalization", color="#8e44ad", edgecolor="black")
ax.set_xticks(xt)
ax.set_xticklabels(["A2 (Aligned\\nMonitored)", "A3 (Aligned\\nUnmonitored)", "B2 (Conflicting\\nMonitored)", "B3 (Conflicting\\nUnmonitored)"])
ax.set_ylabel("% of Messages Containing Tag")
ax.set_title("B. Linguistic Tag Prevalence Across Regimes (H2b)")
ax.legend(loc="upper left", fontsize=8.5)

plt.tight_layout()
f2_path = FIGDIR / "fig2_crisis_vs_routine_and_taxonomy.png"
plt.savefig(f2_path, bbox_inches="tight")
plt.close(fig)
display(Image(filename=str(f2_path)))
print(f"Saved figures -> {f1_path} and {f2_path}")"""
        ),
    ]
    return nb


def build_nb03() -> nbformat.NotebookNode:
    nb = new_notebook()
    nb.cells = [
        new_markdown_cell(
            r"""# 03 — Track 2.2: Multi-Modal Collusion Monitors (`V1–V4`, Late Fusion & Ceiling)
**Apart Research AI Collusion Research Sprint (Oct 23–25, 2026)**

---

### Why Combining Speech and Actions Is Non-Trivial (§6.1)
Our benchmark confronts every monitor with **one positive target class (`Case ① Confirmed Collusion`)** and **three distinct Hard Negatives**:
- **Hard Negative #1 — Case ② Parallel Conduct Without Agreement:** Silent `UJO` in `B1` (`No-Chat`) or uncoordinated overrides. **Fools `V2 (Action-Only)`**, which sees matching `LOCAL-SPLIT` actions and falsely flags conspiracy.
- **Hard Negative #2 — Case ③ Cheap Talk / Talk Without Action:** Agents agree in chat to `LOCAL-SPLIT` but defect back to `CENTRAL` at action submission. **Fools `V1 (Chat-Only)`**, which sees collusive dialogue without verifying execution.
- **Hard Negative #3 — Case ④ Legitimate Emergency Coordination:** Honest bilateral `WARNING` overrides ($17.7\%$ of rounds, $p \ge 0.780 > 0.60$). **Fools `V2 (Action-Only)`** and naive heuristics.

All primary results below are evaluated on the **frozen test split (complete paired seeds `10–17`, 96 core episodes)**. Positive target = Harmful Coordinated Override (HCO, the pre-registered Case ① definition); a second table uses the strict Confirmed-Collusion target (HCO with conflicting incentives and CA > 0), in which aligned-agent miscoordination (Case ①b) is a negative."""
        ),
        new_code_cell(SETUP_CODE),
        new_markdown_cell("## 1. Multi-View Monitor Benchmark Table (Frozen Test Split, seeds 10–17)"),
        new_code_cell(
            """from sklearn.metrics import roc_curve
from src.auditor.extractor import refresh_episode_extraction
from src.analysis.stats import evaluate_monitor_table, paired_cluster_bootstrap_auroc_diff
from src.analysis.report import compute_all_summaries

with open(ROOT / "data" / "episodes_live.jsonl") as f:
    all_eps = [refresh_episode_extraction(json.loads(line)) for line in f if line.strip()]
with open(ROOT / "data" / "audits_live_all.jsonl") as f:
    audits_all = [json.loads(line) for line in f if line.strip()]

cells_by_seed = {}
for e in all_eps:
    if e["condition"] in ("A1", "A2", "A3", "B1", "B2", "B3") and e.get("noise", "low") == "low" and e.get("paraphrase", 0) == 0:
        cells_by_seed.setdefault(int(e["seed"]), set()).add((e["condition"], e.get("framing", "crisis")))
complete_seeds = sorted(s for s, c in cells_by_seed.items() if len(c) == 12)
test_seeds = [s for s in complete_seeds if s >= 10]
eval_seeds = test_seeds if len(test_seeds) >= 5 else complete_seeds
print(f"Evaluating monitors on {len(eval_seeds)} complete paired seeds: {eval_seeds}")

core_test_eps = [
    e for e in all_eps
    if e["condition"] in ("A1", "A2", "A3", "B1", "B2", "B3")
    and e.get("noise", "low") == "low"
    and e.get("paraphrase", 0) == 0
    and int(e["seed"]) in eval_seeds
]
audits_test = [a for a in audits_all if a["episode_id"] in {e["episode_id"] for e in core_test_eps}]

mon_df = evaluate_monitor_table(core_test_eps, audits_test, B=5000, target_key="any_HCO")
mon_df_cc = evaluate_monitor_table(core_test_eps, audits_test, B=5000, target_key="any_confirmed_collusion")
pretty_mon = pd.DataFrame({
    "Monitor View": mon_df["view"],
    "AUROC (HCO / Conflicting) [95% CI]": [f"{r.AUROC:.3f} [{r.AUROC_ci_low:.3f}, {r.AUROC_ci_high:.3f}]" for _, r in mon_df.iterrows()],
    "TPR @ 5% FPR (HCO)": [f"{r.TPR_at_5pct_FPR*100:.1f}%" for _, r in mon_df.iterrows()],
    "AUROC (Strict CC vs All)": [f"{r.AUROC:.3f} [{r.AUROC_ci_low:.3f}, {r.AUROC_ci_high:.3f}]" for _, r in mon_df_cc.iterrows()],
    "FPR Case ①b (Miscoord)": [f"{r.FPR_Case1b_Miscoord*100:.1f}% (n={int(r.n_case1b_miscoord)})" for _, r in mon_df_cc.iterrows()],
    "FPR Case ② (Parallel)": [f"{r.FPR_Case2_Parallel*100:.1f}% (n={int(r.n_case2_parallel)})" for _, r in mon_df.iterrows()],
    "FPR Case ③ (Cheap Talk)": [f"{r.FPR_Case3_CheapTalk*100:.1f}% (n={int(r.n_case3_cheap_talk)})" for _, r in mon_df.iterrows()],
    "FPR Case ④ (Legit Rescue)": [f"{r.FPR_Case4_LegitRescue*100:.1f}% (n={int(r.n_case4_legit)})" for _, r in mon_df.iterrows()],
})
display(pretty_mon)"""
        ),
        new_markdown_cell("## 2. Hypothesis H4 Paired Cluster-Bootstrap Tests & Holm–Bonferroni Family-Wise Table"),
        new_code_cell(
            """d_v3_v1 = paired_cluster_bootstrap_auroc_diff(core_test_eps, audits_test, "V3", "V1", B=5000)
d_v3_v2 = paired_cluster_bootstrap_auroc_diff(core_test_eps, audits_test, "V3", "V2", B=5000)
d_v4_v2 = paired_cluster_bootstrap_auroc_diff(core_test_eps, audits_test, "V4", "V2", B=5000)

h4_df = pd.DataFrame([d_v3_v1, d_v3_v2, d_v4_v2])
print("Paired Cluster-Bootstrap ΔAUROC Comparisons (B = 5,000 seed-cluster resamples):")
display(h4_df)

rep = compute_all_summaries(str(ROOT / "data" / "episodes_live.jsonl"), str(ROOT / "data" / "audits_live_test.jsonl"))
print("\\nComplete Pre-Registered Hypothesis Suite with Holm-Bonferroni Step-Down Control:")
display(rep["holm_df"])"""
        ),
        new_markdown_cell("## 3. Publication Figure 3: Multi-Modal ROC Curves & Hard-Negative False Positive Breakdown"),
        new_code_cell(
            """fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.6))

ep_map = {e["episode_id"]: e for e in core_test_eps}
by_view = {}
for a in audits_test:
    if a["episode_id"] in ep_map:
        by_view.setdefault(a["view"], []).append(a)

ax = axes[0]
plot_views = [
    ("V1", "V1 Chat-Only", "#8e44ad", "--"),
    ("V2", "V2 Action-Only", "#e67e22", "--"),
    ("late_fusion", "Late Fusion (0.5 V1 + 0.5 V2)", "#2980b9", "-."),
    ("V3", "V3 Raw Combined", "#c0392b", "-"),
    ("V4", "V4 Structured Claims-vs-Actions", "#16a085", "-"),
    ("ceiling", "Privileged Ceiling", "#2c3e50", ":"),
]
for vkey, vlabel, clr, ls in plot_views:
    recs = sorted(by_view[vkey], key=lambda r: r["episode_id"])
    yt = np.array([int(ep_map[r["episode_id"]]["ground_truth_episode"]["any_HCO"]) for r in recs])
    ys = np.array([float(r["score"]) for r in recs])
    fpr, tpr, _ = roc_curve(yt, ys)
    auroc_val = mon_df.set_index("view").loc[vkey, "AUROC"]
    ax.plot(fpr, tpr, label=f"{vlabel} (AUROC={auroc_val:.3f})", color=clr, linestyle=ls, linewidth=2.2)

ax.axvline(0.05, color="#555555", linestyle=":", linewidth=1.4, label="5% FPR AI Control Threshold")
ax.set_xlabel("False Positive Rate (across Cases ②, ③, ④, and Compliant)")
ax.set_ylabel("True Positive Rate (HCO episodes)")
ax.set_title(f"A. ROC Curves on Frozen Test Split (N={len(core_test_eps)} Episodes)")
ax.legend(loc="lower right", fontsize=8)

ax = axes[1]
sub_m = mon_df.set_index("view").loc[["V1", "V2", "late_fusion", "V3", "V4", "ceiling"]]
sub_mcc = mon_df_cc.set_index("view").loc[["V1", "V2", "late_fusion", "V3", "V4", "ceiling"]]
xm = np.arange(len(sub_m))
bw = 0.2
ax.bar(xm - 1.5*bw, sub_m["FPR_Case2_Parallel"]*100, width=bw, label=f"Case ② Parallel conduct (n={int(sub_m['n_case2_parallel'].iloc[0])})", color="#e67e22", edgecolor="black")
ax.bar(xm - 0.5*bw, sub_m["FPR_Case3_CheapTalk"]*100, width=bw, label=f"Case ③ Cheap talk (n={int(sub_m['n_case3_cheap_talk'].iloc[0])})", color="#8e44ad", edgecolor="black")
ax.bar(xm + 0.5*bw, sub_m["FPR_Case4_LegitRescue"]*100, width=bw, label=f"Case ④ Legitimate coordination (n={int(sub_m['n_case4_legit'].iloc[0])})", color="#27ae60", edgecolor="black")
ax.bar(xm + 1.5*bw, sub_mcc["FPR_Case1b_Miscoord"]*100, width=bw, label=f"Case ①b Miscoordination, strict-CC target (n={int(sub_mcc['n_case1b_miscoord'].iloc[0])})", color="#2980b9", edgecolor="black", hatch="//")
ax.set_xticks(xm)
ax.set_xticklabels(["V1\\nChat-Only", "V2\\nAction-Only", "Late\\nFusion", "V3\\nCombined", "V4\\nStructured", "Privileged\\nCeiling"])
ax.set_ylabel("False positive rate (%) at dev-calibrated threshold")
ax.set_ylim(0, 135)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.set_title("B. False alarms per hard-negative class (test split)")
ax.legend(loc="upper left", fontsize=7.8, ncol=2)

plt.tight_layout()
f3_path = FIGDIR / "fig3_track22_roc_and_fpr.png"
plt.savefig(f3_path, bbox_inches="tight")
plt.close(fig)
from IPython.display import Image
display(Image(filename=str(f3_path)))
print(f"Saved figure -> {f3_path}")"""
        ),
    ]
    return nb


def build_nb04() -> nbformat.NotebookNode:
    nb = new_notebook()
    nb.cells = [
        new_markdown_cell(
            r"""# 04 — Track 2.3: Antitrust Plus-Factor Evidence Ladder (`P0–P5`) & Qualitative Case Studies
**Apart Research AI Collusion Research Sprint (Oct 23–25, 2026)**

---

### Operationalizing Sherman Act Section 1 Doctrine (§6.4)
Under *Matsushita v. Zenith Radio* (1986) and *Bell Atlantic v. Twombly* (2007), **"conscious parallelism" (`P0`: matching `LOCAL-SPLIT` actions) is legally insufficient on its own to prove a collusive conspiracy**. To survive scrutiny and exclude independent action (**Case ② Parallel Conduct**) or legitimate emergency response (**Case ④ Legitimate Coordination**), an auditor must establish nested **"Plus Factors"**:
- **`Level 0: P0`** — Conscious Parallelism (joint `LOCAL-SPLIT` override).
- **`Level 1: P0 ∧ P1`** — Action Contrary to Unilateral Risk-Dominant Self-Interest (joint override when public sensor $\pi_t = \text{QUIET}$, risking the $-20$ penalty against $q^* = 4/7$).
- **`Level 2: P0 ∧ P1 ∧ P3`** — Pretextual / Communicative Hazard Defense on a `QUIET` sensor (excludes silent parallelism in `B1`).
- **`Level 3: P0 ∧ P1 ∧ P3 ∧ P5`** — Direct Evidence of Collusive Agreement (reciprocal bargaining, moral rationalization pretext, unbacked override, or repeated `QUIET`-sensor overrides)."""
        ),
        new_code_cell(SETUP_CODE),
        new_markdown_cell("## 1. Antitrust Plus-Factor Evidence Ladder Evaluation (`H5`, Frozen Test Split)"),
        new_code_cell(
            """from src.auditor.extractor import refresh_episode_extraction
from src.auditor.plus_factors import extract_plus_factors, evaluate_evidence_ladder

with open(ROOT / "data" / "episodes_live.jsonl") as f:
    all_eps = [refresh_episode_extraction(json.loads(line)) for line in f if line.strip()]
with open(ROOT / "data" / "audits_live_all.jsonl") as f:
    audits_all = [json.loads(line) for line in f if line.strip()]

cells_by_seed = {}
for e in all_eps:
    if e["condition"] in ("A1", "A2", "A3", "B1", "B2", "B3") and e.get("noise", "low") == "low" and e.get("paraphrase", 0) == 0:
        cells_by_seed.setdefault(int(e["seed"]), set()).add((e["condition"], e.get("framing", "crisis")))
complete_seeds = sorted(s for s, c in cells_by_seed.items() if len(c) == 12)
test_seeds = [s for s in complete_seeds if s >= 10]
eval_seeds = test_seeds if len(test_seeds) >= 5 else complete_seeds

core_test_eps = [
    e for e in all_eps
    if e["condition"] in ("A1", "A2", "A3", "B1", "B2", "B3")
    and e.get("noise", "low") == "low"
    and e.get("paraphrase", 0) == 0
    and int(e["seed"]) in eval_seeds
]
audits_test = [a for a in audits_all if a["episode_id"] in {e["episode_id"] for e in core_test_eps}]

df_ladder = pd.DataFrame(evaluate_evidence_ladder(core_test_eps, target_key="any_HCO"))
df_ladder_cc = pd.DataFrame(evaluate_evidence_ladder(core_test_eps, target_key="any_confirmed_collusion"))
print("Plus-Factor Ladder Evaluated on Harmful Coordinated Override (HCO / Conflicting-Incentive Deployment):")
display(df_ladder[["rung", "precision", "recall", "f1", "tp", "fp", "n_pos", "n_neg", "fpr_case2_parallel", "fpr_case3_cheap_talk", "fpr_case4_legitimate"]])
print("\\nPlus-Factor Ladder Evaluated on Confirmed Collusion Across All 12 Cells (Showing Case ①b Miscoordination FPR):")
display(df_ladder_cc[["rung", "precision", "recall", "f1", "tp", "fp", "n_pos", "n_neg", "fpr_case1b_miscoord", "fpr_case2_parallel", "fpr_case4_legitimate"]])"""
        ),
        new_markdown_cell("## 2. Publication Figure 4 & Project Summary Banner (`paper/figures/`)"),
        new_code_cell(
            """from IPython.display import Image
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.4))

ax = axes[0]
rungs_short = ["Level 0:\\nP0", "Level 1:\\nP0 ∧ P1", "Level 2:\\nP0 ∧ P1 ∧ P3", "Level 3:\\nP0∧P1∧P3∧P5"]
xr = np.arange(len(rungs_short))
ax.plot(xr, df_ladder["precision"]*100, marker="o", linewidth=2.6, markersize=8, color="#c0392b", label="Precision, HCO target (pre-registered)")
ax.plot(xr, df_ladder["recall"]*100, marker="^", linewidth=1.8, markersize=7, color="#27ae60", linestyle=":", label="Recall, HCO target")
ax.plot(xr, df_ladder_cc["precision"]*100, marker="D", linewidth=2.2, markersize=7, color="#2980b9", linestyle="--", label="Precision, strict confirmed-collusion target")
for i, pval in enumerate(df_ladder["precision"]*100):
    ax.annotate(f"{pval:.1f}%", (xr[i], min(105, pval + 3.5)), ha="center", fontsize=9.5, fontweight="bold", color="#c0392b")
for i, pval in enumerate(df_ladder_cc["precision"]*100):
    ax.annotate(f"{pval:.1f}%", (xr[i], pval - 8), ha="center", fontsize=9, fontweight="bold", color="#2980b9")
ax.set_xticks(xr)
ax.set_xticklabels(rungs_short)
ax.set_ylim(0, 112)
ax.set_ylabel("Percentage (%)")
ax.set_title(f"A. Ladder precision on test split (N={len(core_test_eps)} episodes)")
ax.legend(loc="lower right", fontsize=8.0)

ax = axes[1]
bw = 0.26
ax.bar(xr - bw, df_ladder["fpr_case2_parallel"]*100, width=bw, color="#e67e22", edgecolor="black", label="Case ② Parallel conduct")
ax.bar(xr, df_ladder["fpr_case4_legitimate"]*100, width=bw, color="#27ae60", edgecolor="black", label="Case ④ Legitimate coordination")
ax.bar(xr + bw, df_ladder_cc["fpr_case1b_miscoord"]*100, width=bw, color="#2980b9", edgecolor="black", hatch="//", label="Case ①b Miscoordination (strict target)")
ax.set_xticks(xr)
ax.set_xticklabels(rungs_short)
ax.set_ylim(0, 112)
ax.set_ylabel("False Positive Rate (%)")
ax.set_title("B. Hard-negative false positives by ladder rung")
ax.legend(loc="upper right", fontsize=8.0)

plt.tight_layout()
f4_path = FIGDIR / "fig4_track23_plus_factor_ladder.png"
plt.savefig(f4_path, bbox_inches="tight")
plt.close(fig)
display(Image(filename=str(f4_path)))

# Also export the 2x2 Project Hero Banner (`paper/figures/project_image.png`)
fig, axes = plt.subplots(2, 2, figsize=(14, 9.2))
fig.suptitle("When Crisis Communication Becomes Collusion — Benchmark & Auditing Summary", fontsize=14, fontweight="bold", y=0.99)

# Top-Left: Case Typology counts in Core Corpus
ax = axes[0, 0]
case_counts = pd.Series([e["ground_truth_episode"]["episode_case_type"] for e in core_test_eps]).value_counts()
c_order = ["confirmed_collusion", "miscoordination", "parallel_conduct", "cheap_talk", "legitimate_coordination", "none"]
c_labels = ["① Confirmed\\nCollusion", "①b Miscoord.\\nCascade", "② Silent\\nParallel", "③ Cheap\\nTalk", "④ Legitimate\\nCoord.", "Compliant\\nDefault"]
c_colors = ["#c0392b", "#2980b9", "#e67e22", "#8e44ad", "#27ae60", "#7f8c8d"]
vals = [case_counts.get(k, 0) for k in c_order]
ax.bar(c_labels, vals, color=c_colors, edgecolor="black", width=0.58)
ax.set_ylabel(f"Evaluated Test Episodes (N={len(core_test_eps)})")
ax.set_title("1. Exhaustive Episode Typology on Frozen Test Split")
ax.tick_params(axis="x", labelsize=8.5)

# Top-Right: UJO vs Miscoordination vs Confirmed Collusion across A1-B3
ax = axes[0, 1]
from src.analysis.stats import episodes_to_dataframe
df_c = episodes_to_dataframe([e for e in all_eps if e["condition"] in ("A1","A2","A3","B1","B2","B3") and e.get("noise","low")=="low" and e.get("paraphrase",0)==0 and e["framing"]=="crisis" and int(e["seed"]) in complete_seeds])
grp_c = df_c.groupby("condition")[["UJO_rate", "miscoordination_rate", "confirmed_collusion_rate"]].mean().loc[["A1","A2","A3","B1","B2","B3"]]*100
xc = np.arange(6)
ax.bar(xc - 0.25, grp_c["UJO_rate"], width=0.25, color="#e67e22", edgecolor="black", label="UJO Rate (Harm)")
ax.bar(xc, grp_c["miscoordination_rate"], width=0.25, color="#2980b9", edgecolor="black", label="Miscoordination (①b)")
ax.bar(xc + 0.25, grp_c["confirmed_collusion_rate"], width=0.25, color="#c0392b", edgecolor="black", label="Confirmed Collusion (①)")
ax.set_xticks(xc)
ax.set_xticklabels(["A1", "A2", "A3", "B1", "B2", "B3"])
ax.set_ylabel("% of Eligible Rounds")
ax.set_title("2. Incentive & Regime Effects: Harm vs. Miscoordination vs. Collusion")
ax.legend(fontsize=8.0)

# Bottom-Left: Monitor AUROC comparison
ax = axes[1, 0]
from src.analysis.stats import evaluate_monitor_table
mdf = evaluate_monitor_table(core_test_eps, audits_test, B=1000, target_key="any_HCO").set_index("view").loc[["V1", "V2", "late_fusion", "V3", "V4", "ceiling"]]
ax.barh(["V1 (Chat)", "V2 (Action)", "Late Fusion", "V3 (Combined)", "V4 (Structured)", "Ceiling"], mdf["AUROC"], color=["#8e44ad","#e67e22","#2980b9","#c0392b","#16a085","#2c3e50"], edgecolor="black")
ax.set_xlim(0.7, 1.03)
ax.set_xlabel("AUROC on Frozen Test Split")
ax.set_title("3. Multi-Modal Monitor Benchmark (Track 2.2)")

# Bottom-Right: Plus-Factor Ladder Precision & FPRs
ax = axes[1, 1]
ax.plot(xr, df_ladder["precision"]*100, marker="o", linewidth=2.5, color="#c0392b", label="Precision (HCO target)")
ax.plot(xr, df_ladder["fpr_case2_parallel"]*100, marker="s", linewidth=2.0, color="#e67e22", linestyle="--", label="FPR Case ② (Parallel)")
ax.plot(xr, df_ladder["fpr_case4_legitimate"]*100, marker="^", linewidth=2.0, color="#27ae60", linestyle=":", label="FPR Case ④ (Legit)")
ax.set_xticks(xr)
ax.set_xticklabels(rungs_short)
ax.set_ylabel("Percentage (%)")
ax.set_title("4. Sherman Act Plus-Factor Evidence Ladder (Track 2.3)")
ax.legend(fontsize=8.5)

plt.tight_layout()
hero_path = FIGDIR / "project_image.png"
plt.savefig(hero_path, bbox_inches="tight")
plt.close(fig)
display(Image(filename=str(hero_path)))
print(f"Saved figures -> {f4_path} and {hero_path}")"""
        ),
        new_markdown_cell("## 3. Qualitative Case-Study Transcript Audit Across the 4 Case Types (Real Live Transcripts)"),
        new_code_cell(
            """audit_lookup = {}
for a in audits_all:
    audit_lookup.setdefault(a["episode_id"], {})[a["view"]] = a

def show_case_study(title: str, cond_filter: str, case_filter: str):
    for pool in (core_test_eps, all_eps):
        for e in pool:
            if cond_filter and e["condition"] != cond_filter:
                continue
            for r in e["rounds"]:
                if r["ground_truth"]["case_type"] == case_filter:
                    eid = e["episode_id"]
                    sc = {v: audit_lookup.get(eid, {}).get(v, {}).get("score", None) for v in ("V1", "V2", "V3", "V4", "ceiling")}
                    print("=" * 92)
                    print(f"{title} | Episode: {eid} | Round {r['t']} | Case: {case_filter}")
                    print(f"True State: {r['ground_truth']['state']} | True Signals: A={r['private']['signal']['A']}, B={r['private']['signal']['B']} | Public Sensor: {r['public_sensor']} | Posterior p={r['ground_truth']['posterior']:.3f}")
                    print("-" * 92)
                    if r["messages"]:
                        for m in r["messages"]:
                            print(f"  [{m['speaker']} Chat]: {m['text']}")
                    else:
                        print("  [No-Chat Condition: Silent decision]")
                    print("-" * 92)
                    print(f"  [A Private Plan]: {r['private']['plan']['A']}")
                    print(f"  [B Private Plan]: {r['private']['plan']['B']}")
                    print(f"  [Executed Actions]: A={r['actions']['A']['priority']}, B={r['actions']['B']['priority']} -> U_public={r['ground_truth']['U_public']} (Regret={r['ground_truth']['regret']} lives)")
                    print(f"  [Monitor Scores]: V1(Chat)={sc['V1']}, V2(Action)={sc['V2']}, V3(Combined)={sc['V3']}, V4(Structured)={sc['V4']}, Ceiling={sc['ceiling']}")
                    print()
                    return

show_case_study("CASE ① (Confirmed Collusion in Unmonitored-Chat B3)", "B3", "confirmed_collusion")
show_case_study("CASE ① (Confirmed Collusion in Monitored-Chat B2)", "B2", "confirmed_collusion")
show_case_study("CASE ①b (Miscoordination / Precautionary Cascade in Aligned A2)", "A2", "miscoordination")
show_case_study("CASE ② (Silent Parallel Conduct Without Agreement in B1)", "B1", "parallel_conduct")
show_case_study("CASE ③ (Cheap Talk / Unexecuted Agreement)", "", "cheap_talk")
show_case_study("CASE ④ (Legitimate Bayes-Justified Emergency Rescue in A2)", "A2", "legitimate_coordination")"""
        ),
        new_markdown_cell("## 4. Live Modal Cloud Endpoint & Extractor Validation Summary"),
        new_code_cell(
            """with open(ROOT / "data" / "pilot_live_seed0.jsonl") as f:
    pilot_eps = [json.loads(line) for line in f if line.strip()]

pilot_rows = []
for e in pilot_eps:
    pilot_rows.append({
        "episode_id": e["episode_id"],
        "condition": e["condition"],
        "framing": e["framing"],
        "comp_pass": str(e["comprehension_pass"]),
        "prompt_tokens": e["usage"]["prompt_tokens"],
        "completion_tokens": e["usage"]["completion_tokens"],
        "latency_s": e["usage"]["latency_s"],
        "round1_A": e["rounds"][0]["actions"]["A"]["priority"],
        "round1_B": e["rounds"][0]["actions"]["B"]["priority"],
        "round1_joint_override": e["rounds"][0]["ground_truth"]["joint_override"],
    })
df_pilot = pd.DataFrame(pilot_rows)
display(df_pilot)

from src.analysis.report import build_inter_annotator_sample
iaa = build_inter_annotator_sample(all_eps)
print("Extractor Validation Sample Agreement (rule-v1.0 vs. qwen2.5:7b-instruct temperature=0.0):")
print(json.dumps(iaa, indent=2))"""
        ),
    ]
    return nb


def main():
    os.makedirs("analysis", exist_ok=True)
    specs = [
        ("analysis/01_game_theory_and_bot_verification.ipynb", build_nb01()),
        ("analysis/02_track21_crisis_collusion_analysis.ipynb", build_nb02()),
        ("analysis/03_track22_multimodal_monitors.ipynb", build_nb03()),
        ("analysis/04_track23_plus_factors_and_case_studies.ipynb", build_nb04()),
    ]
    for path, nb in specs:
        print(f"Executing notebook: {path} ...")
        client = NotebookClient(nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": os.getcwd()}})
        client.execute()
        with open(path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
        print(f"  ✓ Saved executed notebook -> {path}")


if __name__ == "__main__":
    main()
