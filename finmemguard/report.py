"""Tables, figures, and a human-audit sample from runner outputs.

  python -m finmemguard.report [--results results] [--bench data/bench] [--paper paper]
"""
import argparse
import json
import os
import pickle
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})  # TrueType, no Type 3 (ACM requirement)
import pandas as pd

from .benchmark import same

SHORT = {
    "B0 snapshot RAG": "B0 Snapshot RAG (no as-of filter)",
    "B1 time-filtered RAG": "B1 Time-filtered RAG",
    "B1g time-filtered RAG + NR gate (rule release)": "B1g \\;+ NR gate, rule release",
    "B1c time-filtered RAG + NR gate (claim release)": "B1c \\;+ NR gate, claim release",
    "B2 naive persistent memory": "B2 Naive persistent memory",
    "B2 TTL memory (90d)": "B2t TTL memory, 90 d",
    "B2 TTL memory (365d)": "B2t TTL memory, 365 d",
    "B3r FinMemGuard-rule (rule release)": "B3r FinMemGuard, rule release",
    "B3 FinMemGuard-rule (claim release)": "B3 FinMemGuard, claim release",
    "B3* tuned heuristic": "B3* \\;tuned heuristic (VAL)",
    "B3 - invalidation": "B3 $-$ invalidation",
    "B4-sup round-0 only (lam=3)": "B4-sup round-0 only, $\\lambda{=}3$",
    "B4 learned (lam=3)": "B4 Learned, $\\lambda{=}3$",
    "B4 learned (lam=20)": "B4 Learned, $\\lambda{=}20$",
}
ORDER = list(SHORT)
COMPACT = {"B1 time-filtered RAG": "B1 time-filtered", "B1g time-filtered RAG + NR gate (rule release)": "B1g rule release",
           "B1c time-filtered RAG + NR gate (claim release)": "B1c claim release",
           "B3r FinMemGuard-rule (rule release)": "B3r rule release", "B3 FinMemGuard-rule (claim release)": "B3 claim release",
           "B4 learned (lam=3)": "B4 $\\lambda{=}3$"}
REF = "B1c time-filtered RAG + NR gate (claim release)"


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def load(res_dir):
    t = pd.read_csv(os.path.join(res_dir, "main_table.csv"))
    ci = json.load(open(os.path.join(res_dir, "test_bootstrap_ci.json")))
    return t[t.split == "test"], ci


def main_table(res_dir):
    t, ci = load(res_dir)
    p = t[t["mode"] == "primary"].set_index("policy")
    lines = []
    for pol in ORDER:
        if pol not in p.index:
            continue
        r, c = p.loc[pol], ci["primary"][pol]
        lines.append(" & ".join([
            SHORT[pol], pct(r.coverage), pct(r.supported_acc_answered, 2), pct(r.stale_rate_current, 2),
            pct(r.leak_rate, 1), pct(r.withdrawn_answer_rate, 1), f"{r.calls_per_query:.3f}",
            f"{r.utility_lam3:.3f} {{\\scriptsize[{c['utility_lam3'][0]:.3f}, {c['utility_lam3'][1]:.3f}]}}",
        ]) + r" \\")
    return int(p.iloc[0].queries), "\n".join(lines)


def mode_table(res_dirs):
    """Withdrawn-answer rate, answer recall and U3 under strict / permissive label modes."""
    pols = ["B1 time-filtered RAG", "B1g time-filtered RAG + NR gate (rule release)",
            "B1c time-filtered RAG + NR gate (claim release)", "B3r FinMemGuard-rule (rule release)",
            "B3 FinMemGuard-rule (claim release)", "B4 learned (lam=3)"]
    lines = []
    for name, d in res_dirs.items():
        t, _ = load(d)
        lines.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\emph{{{name} workload}}}} \\\\")
        for pol in pols:
            cells = [COMPACT.get(pol, SHORT[pol])]
            for m in ("strict", "permissive"):
                r = t[(t["mode"] == m) & (t.policy == pol)].iloc[0]
                cells += [pct(r.withdrawn_answer_rate), pct(r.answer_recall, 2), f"{r.utility_lam3:.3f}"]
            lines.append(" & ".join(cells) + r" \\")
    return "\n".join(lines)


def seed_summary(res_dir):
    t, _ = load(res_dir)
    p = t[t["mode"] == "primary"]
    out = {}
    for lam in (3, 20):
        names = [f"B4 learned (lam={lam})"] + [f"B4 learned [seed {s}] (lam={lam})" for s in (1, 2)]
        x = p[p.policy.isin(names)]
        out[lam] = {k: (x[k].mean(), x[k].std(ddof=1)) for k in ("calls_per_query", "supported_acc_answered", "utility_lam3")}
    return out


def frontier(res_dirs, out_png):
    fig, axes = plt.subplots(1, len(res_dirs), figsize=(7.0, 2.7), sharey=False)
    for ax, (name, d) in zip(axes, res_dirs.items()):
        t, _ = load(d)
        t = t[t["mode"] == "primary"]
        b4 = t[t.policy.str.match(r"^B4 learned \(lam=")].copy()
        b4["lam"] = b4.policy.str.extract(r"lam=([\d.]+)").astype(float)
        b4 = b4.sort_values("lam")
        ax.plot(b4.calls_per_query, 100 * (1 - b4.supported_acc_answered), "-o", ms=3, color="#1f6feb",
                label="B4 learned ($\\lambda$ sweep)")
        marks = {REF: ("D", "#6e7781", "B1c always-retrieve + claim gate"),
                 "B2 TTL memory (90d)": ("v", "#bf8700", "B2t TTL 90 d"),
                 "B2 TTL memory (365d)": ("^", "#bf8700", "B2t TTL 365 d"),
                 "B3 FinMemGuard-rule (claim release)": ("*", "#cf222e", "B3 rule, claim release"),
                 "B3* tuned heuristic": ("P", "#cf222e", "B3* tuned heuristic"),
                 "B4-sup round-0 only (lam=3)": ("x", "#1f6feb", "B4-sup ($\\lambda{=}3$)")}
        for p, (m, col, lab) in marks.items():
            r = t[t.policy == p]
            if len(r):
                ax.scatter(r.calls_per_query, 100 * (1 - r.supported_acc_answered), marker=m, color=col,
                           s=40 if m != "*" else 80, label=lab, zorder=3)
        ax.set_xlabel("retrieval calls per query", fontsize=9)
        ax.set_title(f"{name} workload", fontsize=9)
        ax.set_yscale("symlog", linthresh=0.1)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3, lw=0.5)
    axes[0].set_ylabel("selective risk (%)", fontsize=9)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(out_png, dpi=300, bbox_inches="tight")


def strata_table(res_dir):
    s = pd.read_csv(os.path.join(res_dir, "strata_table.csv"))
    s = s[(s.split == "test") & (s["mode"] == "primary")]
    rows = []
    for p in ["B1 time-filtered RAG", "B1c time-filtered RAG + NR gate (claim release)", "B2 naive persistent memory",
              "B2 TTL memory (90d)", "B3 FinMemGuard-rule (claim release)", "B4 learned (lam=3)"]:
        g = s[s.policy == p].set_index("stratum")
        cells = [SHORT[p].replace("\\;", "")]
        for st in ["stable", "revised", "nr_gap", "undisclosed"]:
            if st not in g.index:
                cells.append("--"); continue
            r = g.loc[st]
            cells.append(pct(r.supported / r.n) if st in ("stable", "revised") else pct(r.correct_abstain / r.n))
        rows.append(" & ".join(cells) + r" \\")
    n = s[s.policy == "B1 time-filtered RAG"].set_index("stratum").n
    return n.to_dict(), "\n".join(rows)


def audit_sample(bench, out_csv, n_rev=60, n_nr=30, seed=7):
    """Stratified sample of revisions for human adjudication of labels."""
    with open(os.path.join(bench, "issuers.pkl"), "rb") as f:
        issuers = pickle.load(f)
    rng = random.Random(seed)
    rev, nr = [], []
    for iss in issuers:
        for kid, k in iss.keys.items():
            for i in range(1, len(k.vals)):
                if not same(k.vals[i], k.vals[i - 1]):
                    rev.append((iss.cik, kid, k.forms[i], k.accns[i - 1], k.vals[i - 1], k.accns[i], k.vals[i]))
            for lo, hi, lab in k.gaps:
                if lab == "abstain":
                    nr.append((iss.cik, kid, "NR-affected", "", None, "", None))
    by_form = {}
    for r in rev:
        by_form.setdefault(r[2], []).append(r)
    pick = []
    for f, rs in sorted(by_form.items()):
        pick += rng.sample(rs, min(n_rev // len(by_form), len(rs)))
    pick += rng.sample(nr, min(n_nr, len(nr)))
    df = pd.DataFrame(pick, columns=["cik", "claim_key", "stratum", "old_accn", "old_val", "new_accn", "new_val"])
    df["edgar_new"] = df.apply(lambda r: f"https://www.sec.gov/Archives/edgar/data/{r.cik}/{r.new_accn.replace('-', '')}/"
                               if r.new_accn else f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={r.cik}&type=8-K", axis=1)
    df["human_label"] = ""
    df.to_csv(out_csv, index=False)
    return len(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--bench", default="data/bench")
    ap.add_argument("--paper", default="paper")
    ap.add_argument("--skip_audit", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.paper, "figs"), exist_ok=True)
    os.makedirs(os.path.join(a.paper, "tables"), exist_ok=True)
    dirs = {"Uniform": os.path.join(a.results, "uniform"), "Recency": os.path.join(a.results, "recency")}
    for name, d in dirs.items():
        n, body = main_table(d)
        open(os.path.join(a.paper, "tables", f"main_{name.lower()}.tex"), "w").write(body + "\n")
        print(name, "test queries (primary)", n)
        sn, sbody = strata_table(d)
        open(os.path.join(a.paper, "tables", f"strata_{name.lower()}.tex"), "w").write(sbody + "\n")
        print(name, "strata n", sn)
        print(name, "seeds", {k: {m: tuple(round(x, 4) for x in v) for m, v in s.items()} for k, s in seed_summary(d).items()})
    open(os.path.join(a.paper, "tables", "modes.tex"), "w").write(mode_table(dirs) + "\n")
    frontier(dirs, os.path.join(a.paper, "figs", "frontier.pdf"))
    if not a.skip_audit:
        print("audit rows", audit_sample(a.bench, os.path.join(a.results, "audit_sample.csv")))


if __name__ == "__main__":
    main()
