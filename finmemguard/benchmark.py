"""FinMemBench-pilot builder: SEC facts -> versioned claim ledger -> chronological episodes.

Definitions (all dates are integer day ordinals; see `available()`):
  claim key   (cik, concept, unit, start, end) with start=None for instants.
  version     one periodic filing (accession) reporting a value for a key.
  revision    a version whose value differs from the previous version of the same key.
              A revision is NOT necessarily a restatement (recasts, reclassifications).
  NR window   issuer filed 8-K Item 4.02 (non-reliance). For a key whose period ended
              before the notice, the first version filed after the notice (within
              NR_HORIZON days) decides the label:
                value changed  -> "affected": expected action is ABSTAIN during the gap
                value same / none -> "review_required": queries in the gap are NOT scored.
Availability: a filing with filingDate d is usable from day d+1 (conservative; SEC
acceptance timestamps are not proof of public dissemination).
"""
import argparse
import hashlib
import json
import os
import pickle
import random
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

NR_HORIZON = 540          # days after a non-reliance notice in which a restated version counts
REVIEW_HORIZON = 365      # unscored window when no restated version is observed
START_Q, END_Q = date(2016, 1, 1).toordinal(), date(2026, 6, 30).toordinal()
CORPUS_END = date(2026, 10, 1).toordinal()   # snapshot date of the bulk download
REL_TOL = 1e-9


def available(filing_date: str) -> int:
    return date.fromisoformat(filing_date).toordinal() + 1


def same(a: float, b: float) -> bool:
    return abs(a - b) <= REL_TOL * max(abs(a), abs(b))


def rerounded(old: float, new: float) -> bool:
    """True if `new` is just `old` re-rounded to thousands/millions (presentation change)."""
    return any(abs(old - new) < f and round(old / f) * f == new for f in (1e3, 1e6))


def split_of(cik: int) -> str:
    h = int(hashlib.sha256(str(cik).encode()).hexdigest(), 16) % 10
    return "train" if h < 6 else ("val" if h < 8 else "test")


@dataclass
class Key:
    concept: str
    unit: str
    start: str | None
    end: str
    kind: str                      # annual | quarter | instant
    avail: list = field(default_factory=list)   # sorted availability days
    vals: list = field(default_factory=list)
    accns: list = field(default_factory=list)
    forms: list = field(default_factory=list)
    gaps: list = field(default_factory=list)    # [(lo, hi, label)] label in {abstain, review}

    @property
    def end_day(self):
        return date.fromisoformat(self.end).toordinal()

    def version_at(self, t: int):
        """Index of the latest version available at day t, or -1."""
        return bisect_right(self.avail, t) - 1

    def gap_at(self, t: int):
        """Label of the gap containing t; a definite 'abstain' wins over uncertain labels."""
        hit = None
        for lo, hi, lab in self.gaps:
            if lo <= t < hi:
                if lab == "abstain":
                    return lab
                hit = hit or lab
        return hit


@dataclass
class Issuer:
    cik: int
    split: str
    keys: dict                     # key_id -> Key
    notices: list                  # [(avail_day, form, is_nr, accn)] sorted
    nr_days: list                  # availability days of 8-K 4.02 notices
    queries: list = field(default_factory=list)
    nr_cov: dict = field(default_factory=dict)       # notice day -> NoticeCoverage (parsed Item 4.02 text)
    merger_days: list = field(default_factory=list)  # days of 8-K Item 5.06 filings / registrant name changes


def covers(iss_cov, k, d):
    """Does the notice available on day d cover claim k? Falls back to 'any earlier period'."""
    cov = iss_cov.get(d) if iss_cov else None
    return (k.end_day < d - 1) if cov is None else cov.covers(k, d)


def period_kind(start, end):
    if start is None or (isinstance(start, float) and np.isnan(start)):
        return "instant"
    d = (date.fromisoformat(end) - date.fromisoformat(start)).days
    if 350 <= d <= 380:
        return "annual"
    if 80 <= d <= 100:
        return "quarter"
    return None


def build_all_keys(df: pd.DataFrame, stats: dict) -> dict:
    """Vectorised key construction for many issuers. Returns {cik: {kid: Key}}."""
    df = df.copy()
    df["start"] = df["start"].fillna("")
    s = pd.to_datetime(df.start.replace("", None), errors="coerce")
    e = pd.to_datetime(df.end)
    dur = (e - s).dt.days
    df["kind"] = np.select([df.start == "", dur.between(350, 380), dur.between(80, 100)],
                           ["instant", "annual", "quarter"], default="")
    df = df[df.kind != ""]
    kcols = ["cik", "concept", "unit", "start", "end"]
    df["rv"] = df.val.round(6)
    stats["keys_seen"] += int(df.groupby(kcols).ngroups)
    # ambiguity: one accession with several values, or one filing day with several values
    amb_acc = df.groupby(kcols + ["accn"]).rv.nunique()
    amb_acc = amb_acc[amb_acc > 1].reset_index()[kcols].drop_duplicates()
    d1 = df.drop_duplicates(kcols + ["accn"])
    amb_day = d1.groupby(kcols + ["filed"]).rv.nunique()
    amb_day = amb_day[amb_day > 1].reset_index()[kcols].drop_duplicates()
    stats["keys_ambiguous_accession"] += len(amb_acc)
    bad = pd.concat([amb_acc, amb_day]).drop_duplicates()
    stats["keys_ambiguous_sameday"] += len(bad) - len(amb_acc)
    d1 = d1.merge(bad.assign(_bad=1), on=kcols, how="left")
    d1 = d1[d1._bad.isna()].drop_duplicates(kcols + ["filed"])
    d1 = d1.sort_values(kcols + ["filed", "accn"])
    out = defaultdict(dict)
    cur_id, k = None, None
    for r in d1.itertuples(index=False):
        kid = (r.cik, r.concept, r.unit, r.start, r.end)
        if kid != cur_id:
            cur_id = kid
            k = Key(r.concept, r.unit, r.start or None, r.end, r.kind)
            out[r.cik][f"{r.concept}|{r.unit}|{r.start}|{r.end}"] = k
        k.avail.append(available(r.filed)); k.vals.append(r.val)
        k.accns.append(r.accn); k.forms.append(r.form)
    return out


def label_gaps(keys: dict, nr_days: list, stats: dict, nr_cov=None, merger_days=()):
    """Label withdrawal gaps. A claim is covered by a notice only if the parsed Item 4.02 text
    names its period (fallback: every earlier period). A value change across a reverse-merger
    event (8-K Item 5.06 or a registrant name change) is not treated as a restatement."""
    for nr in nr_days:
        for k in keys.values():
            if k.end_day >= nr - 1 or not covers(nr_cov, k, nr):
                if k.end_day < nr - 1:
                    stats["nr_not_covered_keys"] += 1
                continue
            # last version available strictly before the notice day; a filing on the notice
            # day itself (e.g. a concurrent 10-K/A) already counts as post-notice evidence
            i = k.version_at(nr - 1)
            if i < 0:
                continue
            j = i + 1
            if j < len(k.avail) and k.avail[j] <= nr + NR_HORIZON:
                if not same(k.vals[j], k.vals[i]) and not rerounded(k.vals[i], k.vals[j]):
                    merger = any(k.avail[i] < m <= k.avail[j] for m in merger_days)
                    lab = "review_merger" if merger else "abstain"
                    if k.avail[j] > nr:
                        k.gaps.append((nr, k.avail[j], lab))
                    stats["nr_merger_keys" if merger else "nr_affected_keys"] += 1
                    stats["nr_affected_sameday"] += int(k.avail[j] == nr and not merger)
                else:
                    if k.avail[j] > nr:
                        k.gaps.append((nr, k.avail[j], "review_same"))
                    stats["nr_unchanged_keys"] += 1
            else:
                k.gaps.append((nr, nr + REVIEW_HORIZON, "review_none")); stats["nr_unresolved_keys"] += 1


LABEL_MODES = ("primary", "strict", "permissive")


def truth(k: Key, t: int, mode: str = "primary"):
    """Reference for 'value of key k as known at day t'. Returns (kind, value, accns).

    Non-reliance gaps whose label is uncertain (value unchanged after the notice, or no
    post-notice version) are handled by `mode`:
      primary     -> unscored ("review"), the headline setting;
      strict      -> withdrawn (abstain) for every claim the notice covers (AhaSignals-style);
      permissive  -> only value-changing ("affected") claims are withdrawn.
    """
    g = k.gap_at(t)
    if g is not None and g.startswith("review"):
        if mode == "primary":
            return ("review", None, ())
        if mode == "strict":
            return ("abstain_withdrawn", None, ())
        g = None
    if g == "abstain":
        return ("abstain_withdrawn", None, ())
    i = k.version_at(t)
    if i < 0:
        return ("abstain_undisclosed", None, ())
    v = k.vals[i]
    # supporting accessions: admissible reports of the current value since its last change
    j = i
    while j > 0 and same(k.vals[j - 1], v):
        j -= 1
    return ("answer", v, tuple(k.accns[j:i + 1]))


def resolve_latest_annual(iss: Issuer, concept: str, unit: str, t: int):
    best = None
    for kid, k in iss.keys.items():
        if k.kind == "annual" and k.concept == concept and k.unit == unit and k.version_at(t) >= 0:
            if best is None or k.end > iss.keys[best].end:
                best = kid
    return best


def make_queries(iss: Issuer, rng: random.Random, per_month: int, p_hist: float, workload: str = "uniform"):
    keys = iss.keys
    if not keys:
        return
    concepts = sorted({(k.concept, k.unit) for k in keys.values()})
    # watchlist: an agent tracks a few metrics per issuer; NR issuers always watch
    # at least one concept with an affected key so non-reliance episodes get exercised.
    w = rng.sample(concepts, min(4, len(concepts)))
    affected = sorted({(k.concept, k.unit) for k in keys.values() if any(g[2] == "abstain" for g in k.gaps)})
    if affected and not set(affected) & set(w):
        w[-1] = rng.choice(affected)
    w = set(w)
    first = min(min(k.avail) for k in keys.values())
    last = max(max(k.avail) for k in keys.values()) + 120
    annual_concepts = sorted({(k.concept, k.unit) for k in keys.values() if k.kind == "annual" and (k.concept, k.unit) in w})
    t0 = max(START_Q, first)
    for t in range(t0, min(END_Q, last), 30):
        pool = [kid for kid, k in keys.items() if (k.concept, k.unit) in w and t - 730 <= k.end_day <= t]
        if workload == "recency" and pool:
            # analyst-style: most questions concern the latest periods (half-life ~120 days)
            wts = [0.5 ** ((t - keys[k].end_day) / 120.0) for k in pool]
        for _ in range(per_month):
            if not pool:
                break
            kid = rng.choices(pool, weights=wts)[0] if workload == "recency" else rng.choice(pool)
            if rng.random() < p_hist and t - 365 > keys[kid].end_day:
                iss.queries.append((t, "historical", kid, t - 365))
            else:
                iss.queries.append((t, "current", kid, t))
        if annual_concepts:
            c, u = rng.choice(annual_concepts)
            iss.queries.append((t, "latest_annual", f"{c}|{u}", t))


def main():
    global NR_HORIZON, REVIEW_HORIZON
    ap = argparse.ArgumentParser()
    ap.add_argument("--interim", default="data/interim")
    ap.add_argument("--out", default="data/bench")
    ap.add_argument("--n_random", type=int, default=3000)
    ap.add_argument("--per_month", type=int, default=3)
    ap.add_argument("--p_hist", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--workload", choices=("uniform", "recency"), default="uniform")
    ap.add_argument("--nr_horizon", type=int, default=NR_HORIZON, help="days after a notice in which a restated version counts")
    ap.add_argument("--review_horizon", type=int, default=REVIEW_HORIZON, help="unscored/withdrawn window when no restated version appears")
    a = ap.parse_args()
    NR_HORIZON, REVIEW_HORIZON = a.nr_horizon, a.review_horizon
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)

    facts = pd.read_parquet(os.path.join(a.interim, "facts.parquet"))
    filings = pd.read_parquet(os.path.join(a.interim, "filings.parquet"))
    filings = filings[filings.filing_date >= "2014-01-01"]
    nr = filings[filings.non_reliance & (filings.filing_date >= "2015-01-01") & (filings.filing_date <= "2026-06-30")]

    # issuer universe: >= 8 periodic filings with core facts; all NR issuers + random sample
    n_filings = facts.groupby("cik").accn.nunique()
    eligible = set(n_filings[n_filings >= 8].index)
    nr_ciks = sorted(set(nr.cik) & eligible)
    others = sorted(eligible - set(nr_ciks))
    sample = sorted(set(nr_ciks) | set(rng.sample(others, min(a.n_random, len(others)))))
    print("eligible", len(eligible), "nr issuers", len(nr_ciks), "sample", len(sample))

    from .notice_parse import parse_notice
    texts = {}
    tp = os.path.join(a.interim, "nr_texts.jsonl")
    if os.path.exists(tp):
        for line in open(tp):
            r = json.loads(line)
            if r["has_402"] and not r["text"].startswith("__ERROR__"):
                texts[r["accn"]] = r["text"]
    ev = pd.read_parquet(os.path.join(a.interim, "events.parquet")) if os.path.exists(os.path.join(a.interim, "events.parquet")) else None
    nm = pd.read_parquet(os.path.join(a.interim, "names.parquet")) if os.path.exists(os.path.join(a.interim, "names.parquet")) else None
    merger_by = defaultdict(list)
    if ev is not None:
        for r in ev[ev["items"].str.contains("5.06")].itertuples():
            merger_by[r.cik].append(available(r.filing_date))
    if nm is not None:
        # a rename counts only when the former name is a blank-check style name (de-SPAC);
        # ordinary renames hide genuine restatements (audit round 2: 6 of 10 flags were renames)
        spac = nm.name.str.contains(r"acquisition|blank check|merger corp", case=False, regex=True)
        for r in nm[spac].itertuples():
            if r.date_to:
                merger_by[r.cik].append(available(r.date_to))
    stats = defaultdict(int)
    stats.update(eligible_issuers=len(eligible), nr_issuers=len(nr_ciks), sampled_issuers=len(sample),
                 nr_notices=int(nr[nr.cik.isin(sample)].shape[0]))
    all_keys = build_all_keys(facts[facts.cik.isin(sample)], stats)
    ngrp = dict(tuple(filings[filings.cik.isin(sample)].groupby("cik")))
    issuers = []
    for cik in sample:
        keys = all_keys.get(cik, {})
        fl = ngrp.get(cik, pd.DataFrame(columns=filings.columns))
        notices = sorted((available(r.filing_date), r.form, bool(r.non_reliance), r.accn) for r in fl.itertuples())
        nr_cov = {}
        for d in sorted({d for d, f, isnr, _ in notices if isnr and START_Q - 365 <= d <= END_Q}):
            covs = [parse_notice(texts[acc], d) for dd, f, isnr, acc in notices if isnr and dd == d and acc in texts]
            if not covs:
                stats["nr_notice_days_no_text"] += 1
                nr_cov[d] = None
                continue
            if all(c.negated for c in covs):
                stats["nr_notice_days_negated"] += 1
                continue                      # the notice says statements may still be relied upon
            live = [c for c in covs if not c.negated]
            if any(not c.parsed for c in live):
                stats["nr_notice_days_fallback"] += 1
                nr_cov[d] = None              # unparsed text: conservative fallback
                continue
            merged = live[0]
            for c in live[1:]:
                merged.dates |= c.dates; merged.interim_years += c.interim_years; merged.fiscal_years |= c.fiscal_years
            nr_cov[d] = merged
            stats["nr_notice_days_parsed"] += 1
        nr_days = sorted(nr_cov)
        merger_days = sorted(merger_by.get(cik, []))
        label_gaps(keys, nr_days, stats, nr_cov, merger_days)
        for k in keys.values():
            stats["versions"] += len(k.avail)
            stats["revisions"] += sum(not same(k.vals[i], k.vals[i - 1]) for i in range(1, len(k.vals)))
            for i in range(1, len(k.vals)):
                if not same(k.vals[i], k.vals[i - 1]):
                    stats["revisions_by_" + k.forms[i]] += 1
        iss = Issuer(cik, split_of(cik), keys, notices, nr_days, nr_cov=nr_cov, merger_days=merger_days)
        make_queries(iss, rng, a.per_month, a.p_hist, a.workload)
        stats["keys_kept"] += len(keys)
        issuers.append(iss)

    with open(os.path.join(a.out, "issuers.pkl"), "wb") as f:
        pickle.dump(issuers, f)
    stats["queries"] = sum(len(i.queries) for i in issuers)
    pd.Series(dict(stats)).to_csv(os.path.join(a.out, "build_stats.csv"))
    print(pd.Series(dict(stats)).to_string(), flush=True)


if __name__ == "__main__":
    # import through the package so pickled classes resolve as finmemguard.benchmark.*
    from finmemguard.benchmark import main as _main
    _main()
