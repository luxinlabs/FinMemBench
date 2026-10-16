"""Run all FinMemBench-pilot experiments.

  python -m finmemguard.runner [--bench data/bench] [--out results] [--workers 6]

1. Train B4 (one-step learned controller) on TRAIN issuers, queries before 2022-01-01,
   for several seeds; keep the round-0 model as a supervised cost-sensitive baseline.
2. Tune a deterministic heuristic (B3 grid) on VAL issuers.
3. Evaluate all systems on VAL and TEST issuers (company-disjoint), scoring every decision
   under the three label modes (primary / strict / permissive).
4. Issuer-clustered bootstrap 95% CIs and paired differences on TEST.
"""
import argparse
import json
import os
import pickle
import random
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import date

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from . import simulate as S

TRAIN_CUTOFF = date(2022, 1, 1).toordinal()
LAMBDAS = (0.25, 0.5, 1, 2, 3, 5, 10, 20)
SEED_LAMBDAS = (3, 20)
SEEDS = (0, 1, 2)
GRID = [(inv, age) for inv in ("all", "form_aware", "annual_and_amend", "amend_only")
        for age in (None, 365, 180)]
REF = "B1c time-filtered RAG + NR gate (claim release)"
OUT_CODE = {o: i for i, o in enumerate(S.OUTCOMES)}


def ensure_shards(bench, n=48):
    """Split issuers.pkl into n shard files so each worker holds only one shard."""
    d = os.path.join(bench, "shards")
    if os.path.isdir(d) and len(os.listdir(d)) == n:
        return sorted(os.path.join(d, f) for f in os.listdir(d))
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(bench, "issuers.pkl"), "rb") as f:
        issuers = pickle.load(f)
    paths = []
    for i in range(n):
        p = os.path.join(d, f"shard_{i:03d}.pkl")
        with open(p, "wb") as f:
            pickle.dump(issuers[i::n], f)
        paths.append(p)
    counts = pd.Series([x.split for x in issuers]).value_counts().to_dict()
    del issuers
    print("shards written", counts, flush=True)
    return paths


def grid_name(inv, age):
    return f"B3 grid[inv={inv},age={age}]"


def make_policies(spec):
    kind, arg = spec
    if kind == "baselines":
        return [S.SnapshotRAG(), S.TimeFilteredRAG(None), S.TimeFilteredRAG("rule"), S.TimeFilteredRAG("claim"),
                S.NaiveMemory(), S.TTLMemory(90), S.TTLMemory(365),
                S.FinMemGuardRule(gate="rule", name="B3r FinMemGuard-rule (rule release)"),
                S.FinMemGuardRule(gate="claim", name="B3 FinMemGuard-rule (claim release)"),
                S.FinMemGuardRule(gate="claim", inv="form_aware", name="B3f form-aware invalidation"),
                S.FinMemGuardRule(gate=None, name="B3 - NR gate"),
                S.FinMemGuardRule(gate="claim", inv="none", name="B3 - invalidation")]
    if kind == "grid":
        return [S.FinMemGuardRule(gate="claim", inv=inv, max_age=age, name=grid_name(inv, age))
                for inv, age in arg]
    if kind == "tuned":
        inv, age = arg
        return [S.FinMemGuardRule(gate="claim", inv=inv, max_age=age, name="B3* tuned heuristic")]
    if kind == "learned":
        models, lams, tag, safety = arg
        return [S.LearnedPolicy(models, lam=l, safety=safety,
                                name=f"{tag} (lam={l:g})") for l in lams]
    if kind == "collect":
        models, explore, seed = arg
        if models is None:
            return [S.LearnedPolicy(None, explore=explore, seed=seed, name="collect",
                                    behaviour=S.FinMemGuardRule(gate="claim"))]
        return [S.LearnedPolicy(models, lam=3.0, explore=explore, seed=seed, name="collect")]
    raise ValueError(kind)


def _work(args):
    shard, split, spec, collect = args
    with open(shard, "rb") as f:
        issuers = [x for x in pickle.load(f) if x.split == split]
    out_agg, T, X, Y = {}, [], [], []
    for iss in issuers:
        pols = make_policies(spec)
        if collect:
            for p in pols:
                p.rng = random.Random(iss.cik * 1000 + spec[1][2])
        agg, logs = S.run_issuer(iss, pols, collect=collect)
        out_agg.update(agg)
        for lg in logs.values():
            for t, x, cf in lg:
                T.append(t); X.append(x)
                Y.append([OUT_CODE[cf[a]] if a in cf else -1 for a in (S.REUSE, S.REFRESH, S.ABSTAIN)])
    logs = (np.asarray(T, dtype=np.int32), np.asarray(X, dtype=np.float32).reshape(-1, len(S.FEATURES)),
            np.asarray(Y, dtype=np.int8).reshape(-1, 3))
    return out_agg, logs


class SerialPool:
    """Drop-in for ProcessPoolExecutor when memory is tight (workers=0)."""
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def map(self, f, it): return map(f, it)


def run(pool, shards, split, spec, collect=False):
    agg, logs = {}, []
    for a, l in pool.map(_work, [(s, split, spec, collect) for s in shards]):
        agg.update(a)
        if collect:
            logs.append(l)
    if collect:
        logs = tuple(np.concatenate([l[i] for l in logs]) for i in range(3))
    return agg, logs


def fit_models(logs, seed=0):
    S._PROBA_CACHE.clear()
    T, X, Y = logs
    keep = T < TRAIN_CUTOFF
    models = {}
    for a in (S.REUSE, S.REFRESH, S.ABSTAIN):
        m_ = keep & (Y[:, a] >= 0)
        y = np.array(S.OUTCOMES)[Y[m_, a]]
        # depth-limited tree: fast per-decision inference (traversed directly in simulate.py)
        m = DecisionTreeClassifier(max_depth=12, min_samples_leaf=50, random_state=seed)
        m.fit(X[m_].astype(float), y)
        models[a] = m
        print(f"  action {a}: n={len(y)} classes={dict(pd.Series(y).value_counts())}", flush=True)
    models["tag"] = os.urandom(8).hex()   # invalidates per-worker prediction caches
    return models


def to_frame(agg):
    rows = [(*k, *v) for k, v in agg.items()]
    return pd.DataFrame(rows, columns=["policy", "cik", "split", "qtype", "stratum", "mode", *S.COLS])


def summarize(df):
    s = df[list(S.COLS)].sum()
    n, ans = s.n, max(s.answered, 1)
    exp_ans = df[df.stratum.isin(["stable", "revised"])]
    exp_abs = df[df.stratum.isin(["nr_gap", "undisclosed"])]
    cur = df[df.qtype.isin(["current", "latest_annual"]) & df.stratum.isin(["stable", "revised"])]
    gap = df[df.stratum == "nr_gap"]
    return {
        "queries": int(n), "gap_queries": int(gap.n.sum()), "coverage": s.answered / n,
        "supported_acc_answered": s.supported / ans,
        "selective_risk": 1 - s.supported / ans,
        "stale_rate_current": cur.stale.sum() / max(cur.answered.sum(), 1),
        "leak_rate": s.leak / ans,
        "withdrawn_answer_rate": s.withdrawn / max(gap.n.sum(), 1),
        "abstain_recall": exp_abs.correct_abstain.sum() / max(exp_abs.n.sum(), 1),
        "answer_recall": exp_ans.supported.sum() / max(exp_ans.n.sum(), 1),
        "calls_per_query": s.calls / n,
        "calls_per_supported": s.calls / max(s.supported, 1),
        "utility_lam3": (s.supported + s.correct_abstain - 3 * (s.answered - s.correct) - 0.05 * s.calls) / n,
    }


def _issuer_sums(df):
    d = df.copy()
    ans_st = d.stratum.isin(["stable", "revised"])
    cur = d.qtype.isin(["current", "latest_annual"]) & ans_st
    d["cur_answered"] = np.where(cur, d.answered, 0)
    d["cur_stale"] = np.where(cur, d.stale, 0)
    d["gap_n"] = np.where(d.stratum == "nr_gap", d.n, 0)
    d["abs_n"] = np.where(~ans_st, d.n, 0)
    d["abs_ok"] = np.where(~ans_st, d.correct_abstain, 0)
    d["ans_n"] = np.where(ans_st, d.n, 0)
    d["ans_ok"] = np.where(ans_st, d.supported, 0)
    cols = list(S.COLS) + ["cur_answered", "cur_stale", "gap_n", "abs_n", "abs_ok", "ans_n", "ans_ok"]
    return d.groupby("cik")[cols].sum()


def _metrics_from_sums(M, cols):
    c = {k: M[:, i].astype(float) for i, k in enumerate(cols)}
    nz = lambda x: np.maximum(x, 1)
    return {
        "coverage": c["answered"] / nz(c["n"]),
        "supported_acc_answered": c["supported"] / nz(c["answered"]),
        "stale_rate_current": c["cur_stale"] / nz(c["cur_answered"]),
        "leak_rate": c["leak"] / nz(c["answered"]),
        "withdrawn_answer_rate": c["withdrawn"] / nz(c["gap_n"]),
        "abstain_recall": c["abs_ok"] / nz(c["abs_n"]),
        "answer_recall": c["ans_ok"] / nz(c["ans_n"]),
        "calls_per_query": c["calls"] / nz(c["n"]),
        "utility_lam3": (c["supported"] + c["correct_abstain"] - 3 * (c["answered"] - c["correct"])
                         - 0.05 * c["calls"]) / nz(c["n"]),
    }


def bootstrap(df, metric_keys, B=1000, seed=0, ref=None):
    """Issuer-clustered bootstrap; with `ref`, also CIs of paired differences (policy - ref)."""
    rng = np.random.default_rng(seed)
    A = _issuer_sums(df)
    W = rng.multinomial(len(A), np.full(len(A), 1 / len(A)), size=B)
    ma = _metrics_from_sums(W @ A.values, list(A.columns))
    out = {k: (float(np.percentile(ma[k], 2.5)), float(np.percentile(ma[k], 97.5))) for k in metric_keys}
    if ref is not None:
        R_ = _issuer_sums(ref).reindex(A.index, fill_value=0)
        mr = _metrics_from_sums(W @ R_.values, list(R_.columns))
        for k in metric_keys:
            dlt = ma[k] - mr[k]
            out["diff_" + k] = (float(np.percentile(dlt, 2.5)), float(np.percentile(dlt, 97.5)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="data/bench")
    ap.add_argument("--out", default="results")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    shards = ensure_shards(a.bench)
    t0 = time.time()
    frames = []
    with (ProcessPoolExecutor(a.workers) if a.workers > 0 else SerialPool()) as pool:
        # --- B4 training: 2 rounds per seed; round-0 model kept as supervised baseline
        trained = {}
        for seed in SEEDS:
            print(f"[seed {seed}] collect round 0 (rule behaviour, eps=0.3)", flush=True)
            _, logs0 = run(pool, shards, "train", ("collect", (None, 0.3, seed)), collect=True)
            m0 = fit_models(logs0, seed)
            print(f"[seed {seed}] collect round 1 (learned behaviour, eps=0.1) {time.time() - t0:.0f}s", flush=True)
            _, logs1 = run(pool, shards, "train", ("collect", (m0, 0.1, seed + 100)), collect=True)
            m1 = fit_models(tuple(np.concatenate([u, v]) for u, v in zip(logs0, logs1)), seed)
            trained[seed] = (m0, m1)
            del logs0, logs1
        with open(os.path.join(a.out, "b4_models.pkl"), "wb") as f:
            pickle.dump(trained, f)
        print(f"trained in {time.time() - t0:.0f}s", flush=True)

        # --- heuristic grid on VAL (claim-release gate), select by primary U3
        agg, _ = run(pool, shards, "val", ("grid", GRID))
        g = to_frame(agg)
        g = g[g["mode"] == "primary"]
        gscore = {p: summarize(x)["utility_lam3"] for p, x in g.groupby("policy")}
        best = max(GRID, key=lambda c: gscore[grid_name(*c)])
        pd.Series(gscore).sort_values(ascending=False).to_csv(os.path.join(a.out, "grid_val.csv"))
        print("tuned heuristic:", best, round(gscore[grid_name(*best)], 4), flush=True)

        specs = [("baselines", None), ("tuned", best),
                 ("learned", (trained[0][1], LAMBDAS, "B4 learned", "claim")),
                 ("learned", (trained[0][0], SEED_LAMBDAS, "B4-sup round-0 only", "claim"))]
        for seed in SEEDS[1:]:
            specs.append(("learned", (trained[seed][1], SEED_LAMBDAS, f"B4 learned [seed {seed}]", "claim")))
        for split in ("val", "test"):
            for spec in specs:
                agg, _ = run(pool, shards, split, spec)
                frames.append(to_frame(agg))
            print(f"evaluated {split} in {time.time() - t0:.0f}s", flush=True)
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(os.path.join(a.out, "per_issuer_counts.parquet"), index=False)
    json.dump({"tuned_heuristic": list(best)}, open(os.path.join(a.out, "config.json"), "w"))

    rows = []
    for (split, mode, pol), g in df.groupby(["split", "mode", "policy"]):
        rows.append({"split": split, "mode": mode, "policy": pol, **summarize(g)})
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "main_table.csv"), index=False)

    strat = []
    for (split, mode, pol, st), g in df.groupby(["split", "mode", "policy", "stratum"]):
        s = g[list(S.COLS)].sum()
        strat.append({"split": split, "mode": mode, "policy": pol, "stratum": st, "n": int(s.n),
                      "answered": int(s.answered), "supported": int(s.supported),
                      "correct_abstain": int(s.correct_abstain), "wrong": int(s.answered - s.correct),
                      "calls": int(s.calls)})
    pd.DataFrame(strat).to_csv(os.path.join(a.out, "strata_table.csv"), index=False)

    qt = []
    for (split, mode, pol, q), g in df.groupby(["split", "mode", "policy", "qtype"]):
        qt.append({"split": split, "mode": mode, "policy": pol, "qtype": q, **summarize(g)})
    pd.DataFrame(qt).to_csv(os.path.join(a.out, "qtype_table.csv"), index=False)

    keys = ["coverage", "supported_acc_answered", "stale_rate_current", "leak_rate",
            "withdrawn_answer_rate", "calls_per_query", "utility_lam3", "answer_recall", "abstain_recall"]
    ci = {}
    test = df[df.split == "test"]
    for mode in ("primary", "strict", "permissive"):
        tm = test[test["mode"] == mode]
        ref = tm[tm.policy == REF]
        ci[mode] = {pol: bootstrap(g, keys, B=a.boot, ref=ref) for pol, g in tm.groupby("policy")}
    with open(os.path.join(a.out, "test_bootstrap_ci.json"), "w") as f:
        json.dump(ci, f, indent=1)
    t = pd.DataFrame(rows)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(t[(t.split == "test") & (t["mode"] == "primary")].round(4).to_string(index=False))
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
