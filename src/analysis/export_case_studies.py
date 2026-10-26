"""Emit LaTeX case-study boxes for the appendix directly from data (no hand-copied numbers)."""
import json, re, sys
sys.path.insert(0, ".")

eps = {json.loads(l)["episode_id"]: json.loads(l) for l in open("data/episodes_live.jsonl") if l.strip()}
aud = {}
for l in open("data/audits_live_all.jsonl"):
    a = json.loads(l); aud.setdefault(a["episode_id"], {})[a["view"]] = a["score"]

def esc(s):
    s = s.replace("\\", " ").replace("&", "\\&").replace("%", "\\%").replace("$", "\\$").replace("#", "\\#")
    s = s.replace("_", "\\_").replace("{", "(").replace("}", ")").replace("~", "-").replace("^", "")
    s = s.replace("’", "'").replace("“", "``").replace("”", "''").replace("—", "---").replace("–", "--")
    return re.sub(r"[^\x00-\x7F]", "", s)

def trunc(s, n=42):
    w = s.split()
    return " ".join(w[:n]) + (" [...]" if len(w) > n else "")

def find(cond, case, prefer_test=True):
    cands = []
    for eid, e in eps.items():
        if e["condition"] != cond or e.get("noise", "low") != "low" or e.get("paraphrase", 0) != 0:
            continue
        for r in e["rounds"]:
            if r["ground_truth"]["case_type"] == case:
                cands.append((0 if int(e["seed"]) >= 10 else 1, int(e["seed"]), eid, r))
    cands.sort(key=lambda x: (x[0], x[1], x[3]["t"]))
    return cands[0] if cands else None

specs = [
    ("Case 1, confirmed collusion (B3, unmonitored chat, conflicting incentives)", "B3", "confirmed_collusion"),
    ("Case 1b, miscoordination / precautionary cascade (A3, aligned incentives)", "A3", "miscoordination"),
    ("Case 2, parallel conduct without agreement (B1, no chat)", "B1", "parallel_conduct"),
    ("Case 3, cheap talk (agreement not executed)", "B2", "cheap_talk"),
    ("Case 4, legitimate coordination (A2)", "A2", "legitimate_coordination"),
]
out = []
for title, cond, case in specs:
    f = find(cond, case)
    if f is None and case == "cheap_talk":
        for c in ("B3", "A2", "A3"):
            f = find(c, case)
            if f: break
    if f is None:
        continue
    _, seed, eid, r = f
    e = eps[eid]
    g = r["ground_truth"]
    sc = aud.get(eid, {})
    s = (f"\\textbf{{{esc(title)}}} --- episode \\texttt{{{esc(eid)}}}, round {r['t']}"
         f" (seed {seed}, {'test' if seed >= 10 else 'dev'} split).\\\\\n"
         f"\\emph{{Hidden state}} {g['state']}; signals A={r['private']['signal']['A']}, B={r['private']['signal']['B']};"
         f" public sensor {r['public_sensor']}; posterior $p={g['posterior']:.3f}$; episode $CA={0.5*(e['ground_truth_episode']['coalition_advantage']['A']+e['ground_truth_episode']['coalition_advantage']['B']):+.1f}$.\\\\\n")
    if r["messages"]:
        for m in r["messages"][:3]:
            s += f"\\texttt{{[{m['speaker']}]}} ``{esc(trunc(m['text']))}''\\\\\n"
    else:
        s += "\\emph{No chat channel in this condition.}\\\\\n"
    s += (f"\\emph{{Executed:}} A={r['actions']['A']['priority']}, B={r['actions']['B']['priority']};"
          f" $U_{{\\mathrm{{public}}}}={g['U_public']}$, regret $={g['regret']}$ lives.\\\\\n")
    vs = [("V1", "V1"), ("V2", "V2"), ("V3", "V3"), ("V4", "V4"), ("logreg_plus", "LogReg"), ("ceiling", "Ceiling")]
    s += "\\emph{Episode-level monitor scores:} " + ", ".join(
        f"{lab}={sc[k]:.2f}" for k, lab in vs if k in sc) + "."
    out.append(s)

open(sys.argv[1], "w").write("\n\\medskip\\hrule\\medskip\n".join(out) + "\n")
print("wrote", len(out), "cases")
