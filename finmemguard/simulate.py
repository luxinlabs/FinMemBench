"""Chronological replay of FinMemBench-pilot episodes against memory policies.

Every policy sees, per issuer, the same time-ordered stream of
  * filing notices (form type, 8-K Item 4.02 flag) -- free metadata, like an EDGAR feed
  * queries (current value of a claim key, latest annual value of a concept,
    or a historical as-of value)
and must choose: REUSE its memory (cost 0), REFRESH from the time-filtered fact
index (1 retrieval call), or ABSTAIN. Ground truth comes from benchmark.truth().
Policies act on every query; scoring is done under each label mode in LABEL_MODES.
"""
from bisect import bisect_right
from collections import defaultdict
import math
import random

import numpy as np

from .benchmark import Issuer, Key, truth, same, covers, CORPUS_END, LABEL_MODES

PERIODIC = ("10-K", "10-K/A", "10-Q", "10-Q/A")
FORM_IDX = {f: i for i, f in enumerate(PERIODIC)}
REUSE, REFRESH, ABSTAIN = 0, 1, 2
OUTCOMES = ("correct", "wrong", "correct_abstain", "unneeded_abstain")


# ---------------------------------------------------------------- environment
class Env:
    """Per-issuer read-only views: fact index, notice feed, ground truth."""

    def __init__(self, iss: Issuer):
        self.iss = iss
        self.keys = iss.keys
        self.accn_avail = {}
        for k in iss.keys.values():
            for a, d in zip(k.accns, k.avail):
                self.accn_avail.setdefault(a, d)
        self.form_days = [[] for _ in PERIODIC]
        for d, form, is_nr, _ in iss.notices:
            if form in FORM_IDX:
                self.form_days[FORM_IDX[form]].append(d)
        self.periodic_days = sorted(d for fd in self.form_days for d in fd)
        self.nr_days = iss.nr_days
        self.annual = defaultdict(list)       # (concept, unit) -> [(first_avail, end, kid)]
        for kid, k in iss.keys.items():
            if k.kind == "annual":
                self.annual[(k.concept, k.unit)].append((k.avail[0], k.end, kid))

    # fact index with an availability filter (as_of=None -> snapshot, no filter)
    def retrieve(self, kid, as_of):
        k = self.keys[kid]
        t = CORPUS_END if as_of is None else as_of
        i = k.version_at(t)
        return None if i < 0 else (k.vals[i], k.accns[i])

    def resolve_latest(self, cu, as_of):
        c, u = cu.split("|")
        t = CORPUS_END if as_of is None else as_of
        best = None
        for first, end, kid in self.annual.get((c, u), ()):
            if first <= t and (best is None or end > best[0]):
                best = (end, kid)
        return None if best is None else best[1]

    def n_forms_between(self, lo, hi):
        """Counts of periodic filings per form with availability in (lo, hi]."""
        return [bisect_right(fd, hi) - bisect_right(fd, lo) for fd in self.form_days]

    def any_periodic_between(self, lo, hi):
        return bisect_right(self.periodic_days, hi) - bisect_right(self.periodic_days, lo) > 0

    def covering_notice(self, kid, t):
        """Latest Item 4.02 notice available by t that covers the claim's period, or None."""
        k = self.keys[kid]
        cov = getattr(self.iss, "nr_cov", None)
        for d in reversed(self.nr_days[:bisect_right(self.nr_days, t)]):
            if k.end_day < d - 1 and covers(cov, k, d):
                return d
        return None

    def gate_open(self, kid, t):
        """RULE release (original heuristic): the quarantine lifts on any later 10-K/10-K/A
        (annual claims) or any later periodic filing (other claims), whether or not that
        filing re-reports this claim."""
        k = self.keys[kid]
        i = bisect_right(self.nr_days, t)
        cov = getattr(self.iss, "nr_cov", None)
        for d in self.nr_days[:i]:
            if k.end_day >= d - 1 or not covers(cov, k, d):
                continue
            lift_forms = ("10-K", "10-K/A") if k.kind == "annual" else PERIODIC
            lifted = any(bisect_right(self.form_days[FORM_IDX[f]], t) - bisect_right(self.form_days[FORM_IDX[f]], d) > 0
                         for f in lift_forms)
            if not lifted:
                return True
        return False

    def last_nr_age(self, t):
        i = bisect_right(self.nr_days, t)
        return 99999 if i == 0 else t - self.nr_days[i - 1]

    def query_truth(self, q, mode="primary"):
        t, qtype, ref, as_of = q
        if qtype == "latest_annual":
            kid = self.resolve_latest(ref, as_of)
            if kid is None:
                return ("abstain_undisclosed", None, ()), None
            return truth(self.keys[kid], as_of, mode), kid
        return truth(self.keys[ref], as_of, mode), ref

    def revised_before(self, kid, as_of):
        k = self.keys[kid]
        i = k.version_at(as_of)
        return any(not same(k.vals[j], k.vals[j - 1]) for j in range(1, i + 1))


def outcome(env, tr, ans, as_of):
    """Score an answer (value, accn) or None (abstain) against truth tr."""
    kind, val, support = tr
    if ans is None:
        return "correct_abstain" if kind.startswith("abstain") else "unneeded_abstain"
    if kind != "answer":
        return "wrong"
    return "correct" if same(ans[0], val) else "wrong"


# ---------------------------------------------------------------- policies
class Policy:
    name = "base"

    def reset(self, env):
        self.env = env
        self.mem = {}          # qkey -> (val, accn, read_t, kid)
        self.checked = {}      # qkey -> last day a quarantined claim was re-checked
        self.obs_refresh = 0
        self.obs_changed = 0

    def qkey(self, q):
        return q[2] if q[1] != "latest_annual" else "LA|" + q[2]

    def fetch(self, q):
        """Time-filtered retrieval for query q; returns (ans, kid)."""
        t, qtype, ref, as_of = q
        kid = self.env.resolve_latest(ref, as_of) if qtype == "latest_annual" else ref
        if kid is None:
            return None, None
        return self.env.retrieve(kid, as_of), kid

    def store(self, q, ans, kid):
        if q[1] == "historical" or ans is None:
            return
        qk = self.qkey(q)
        old = self.mem.get(qk)
        if old is not None:
            self.obs_refresh += 1
            self.obs_changed += int(not same(old[0], ans[0]) or old[3] != kid)
        self.mem[qk] = (ans[0], ans[1], q[0], kid)

    # ---- CLAIM-SPECIFIC quarantine: a covered claim is released only by evidence for
    # ---- that claim that became available on/after the notice day.
    def released(self, ans, d):
        return ans is not None and self.env.accn_avail.get(ans[1], -1) >= d

    def blocked_after_fetch(self, kid, ans, as_of):
        if kid is None:
            return False
        d = self.env.covering_notice(kid, as_of)
        return d is not None and not self.released(ans, d)

    def claim_precheck(self, q, m):
        """For current/latest queries. Returns None (claim not quarantined, proceed) or a
        final (ans, kid, cost) decision. Re-checks a quarantined claim (1 call) only when a
        periodic filing has arrived since the notice or since the last check."""
        t, qtype, ref, as_of = q
        if qtype == "latest_annual" and m is not None:
            # a new annual filing may change which claim "latest" resolves to; the gate must
            # check the claim the query resolves to now, so re-resolve before any reuse
            c = self.env.n_forms_between(m[2], t)
            if c[FORM_IDX["10-K"]] + c[FORM_IDX["10-K/A"]] > 0:
                ans, kid2 = self.fetch(q)
                if self.blocked_after_fetch(kid2, ans, t):
                    self.checked[self.qkey(q)] = t
                    return None, kid2, 1
                self.store(q, ans, kid2)
                return ans, kid2, 1
        kid = m[3] if m is not None else (ref if qtype == "current" else None)
        if kid is None:
            return None
        d = self.env.covering_notice(kid, t)
        if d is None or (m is not None and self.released((m[0], m[1]), d)):
            return None
        qk = self.qkey(q)
        since = max(d - 1, self.checked.get(qk, -1))
        if not self.env.any_periodic_between(since, t):
            return None, kid, 0
        ans, kid2 = self.fetch(q)
        if self.blocked_after_fetch(kid2, ans, t):
            self.checked[qk] = t
            return None, kid2, 1
        self.store(q, ans, kid2)
        return ans, kid2, 1

    def act(self, q):
        raise NotImplementedError


class SnapshotRAG(Policy):
    """B0: retrieves the latest version in the corpus snapshot, no as-of filter."""
    name = "B0 snapshot RAG"

    def act(self, q):
        t, qtype, ref, as_of = q
        kid = self.env.resolve_latest(ref, None) if qtype == "latest_annual" else ref
        if kid is None:
            return None, None, 1
        return self.env.retrieve(kid, None), kid, 1


class TimeFilteredRAG(Policy):
    """B1: always retrieves with the as-of filter. gate in {None, 'rule', 'claim'}."""

    def __init__(self, gate=None):
        self.gate = gate
        self.name = {None: "B1 time-filtered RAG", "rule": "B1g time-filtered RAG + NR gate (rule release)",
                     "claim": "B1c time-filtered RAG + NR gate (claim release)"}[gate]

    def act(self, q):
        ans, kid = self.fetch(q)
        if kid is None:
            return ans, kid, 1
        if self.gate == "rule" and self.env.gate_open(kid, q[3]):
            return None, kid, 1
        if self.gate == "claim" and self.blocked_after_fetch(kid, ans, q[3]):
            return None, kid, 1
        return ans, kid, 1


class NaiveMemory(Policy):
    """B2: persistent key->value memory; first answer is reused forever, no as-of notion."""
    name = "B2 naive persistent memory"

    def act(self, q):
        t, qtype, ref, as_of = q
        qk = ref if qtype != "latest_annual" else "LA|" + ref
        m = self.mem.get(qk)
        if m is not None:
            return (m[0], m[1]), m[3], 0
        nq = (t, "current" if qtype == "historical" else qtype, ref, t)   # no as-of awareness
        ans, kid = self.fetch(nq)
        if ans is not None:
            self.mem[qk] = (ans[0], ans[1], t, kid)
        return ans, kid, 1


class TTLMemory(Policy):
    """B2-TTL: memory refreshed when older than `ttl` days; historical via as-of retrieval."""

    def __init__(self, ttl):
        self.ttl = ttl
        self.name = f"B2 TTL memory ({ttl}d)"

    def act(self, q):
        if q[1] == "historical":
            ans, kid = self.fetch(q)
            return ans, kid, 1
        m = self.mem.get(self.qkey(q))
        if m is not None and q[0] - m[2] <= self.ttl:
            return (m[0], m[1]), m[3], 0
        ans, kid = self.fetch(q)
        self.store(q, ans, kid)
        return ans, kid, 1


INVALIDATION = {
    # which notice forms make an entry SUSPECT, per claim kind (annual, quarter, instant)
    "all":              (PERIODIC, PERIODIC, PERIODIC),
    "form_aware":       (("10-K", "10-K/A"), ("10-K/A", "10-Q", "10-Q/A"), PERIODIC),
    "annual_and_amend": (("10-K", "10-K/A", "10-Q/A"),) * 3,
    "amend_only":       (("10-K/A", "10-Q/A"),) * 3,
    "none":             ((), (), ()),
}
KINDS = ("annual", "quarter", "instant")


class FinMemGuardRule(Policy):
    """B3: event-sourced lifecycle. Entries become SUSPECT when a filing of an invalidating
    form arrives after they were read (refresh before reuse) and QUARANTINED (abstain)
    while an 8-K Item 4.02 notice covering their period is unresolved.
      gate   : 'claim' (released only by post-notice evidence for this claim),
               'rule' (original heuristic), or None;
      inv    : key of INVALIDATION;  max_age: optional TTL cap in days."""

    def __init__(self, gate="claim", inv="all", max_age=None, name=None):
        self.gate, self.inv, self.max_age = gate, inv, max_age
        self.name = name or f"B3 rule[gate={gate},inv={inv},age={max_age}]"

    def suspect(self, m, t):
        if self.max_age is not None and t - m[2] > self.max_age:
            return True
        forms = INVALIDATION[self.inv][KINDS.index(self.env.keys[m[3]].kind)]
        if not forms:
            return False
        c = self.env.n_forms_between(m[2], t)
        return any(c[FORM_IDX[f]] for f in forms)

    def act(self, q):
        t, qtype, ref, as_of = q
        if qtype == "historical":
            if self.gate == "rule" and self.env.gate_open(ref, as_of):
                return None, ref, 0
            ans, kid = self.fetch(q)
            if self.gate == "claim" and self.blocked_after_fetch(kid, ans, as_of):
                return None, kid, 1
            return ans, kid, 1
        m = self.mem.get(self.qkey(q))
        if self.gate == "claim":
            r = self.claim_precheck(q, m)
            if r is not None:
                return r
        if m is not None:
            if self.gate == "rule" and self.env.gate_open(m[3], t):
                return None, m[3], 0
            if not self.suspect(m, t):
                return (m[0], m[1]), m[3], 0
        if self.gate == "rule" and qtype == "current" and self.env.gate_open(ref, t):
            return None, ref, 0
        ans, kid = self.fetch(q)
        if self.gate == "rule" and kid is not None and self.env.gate_open(kid, t):
            return None, kid, 1
        if self.gate == "claim" and self.blocked_after_fetch(kid, ans, t):
            self.checked[self.qkey(q)] = t
            return None, kid, 1
        self.store(q, ans, kid)
        return ans, kid, 1


# ---------------------------------------------------------------- learned policy
FEATURES = ("has_entry", "n_10K", "n_10KA", "n_10Q", "n_10QA", "log_days_since_read",
            "gate_open", "log_days_since_nr", "obs_change_rate", "obs_refreshes",
            "kind_annual", "kind_quarter", "is_latest", "log_period_age")


def featurize(pol: Policy, q, kid_hint):
    t, qtype, ref, as_of = q
    m = pol.mem.get(pol.qkey(q))
    kid = m[3] if m is not None else kid_hint
    k: Key = pol.env.keys[kid] if kid is not None else None
    if m is not None:
        c = pol.env.n_forms_between(m[2], t)
        since = math.log1p(t - m[2])
    else:
        c, since = [-1, -1, -1, -1], -1.0
    return [int(m is not None), *c, since,
            int(kid is not None and pol.env.gate_open(kid, t)),
            math.log1p(min(pol.env.last_nr_age(t), 99999)),
            (pol.obs_changed + 1) / (pol.obs_refresh + 20), math.log1p(pol.obs_refresh),
            int(k is not None and k.kind == "annual"), int(k is not None and k.kind == "quarter"),
            int(qtype == "latest_annual"),
            math.log1p(max(0, t - k.end_day)) if k is not None else -1.0]


_PROBA_CACHE = {}


class LearnedPolicy(Policy):
    """B4: one-step (contextual-bandit) controller. A per-action outcome model
    P(outcome | state, action) is fit from replayed episodes on TRAIN issuers; at decision
    time it maximises expected immediate reward
        r = +1 correct answer / correct abstention, -lam wrong answer,
            0 unneeded abstention, -c per retrieval call.
    safety='claim' applies the claim-specific quarantine as a hard constraint before the
    learned decision, so the learner only trades retrieval cost against staleness risk.
    `explore` > 0 gives epsilon-greedy behaviour used during data collection."""

    def __init__(self, models=None, lam=3.0, c=0.05, explore=0.0, behaviour=None, seed=0,
                 safety="claim", name=None):
        self.models, self.lam, self.c, self.explore = models, lam, c, explore
        self.behaviour = behaviour          # rule policy used as behaviour when models is None
        self.safety = safety
        self.rng = random.Random(seed)
        self.name = name or f"B4 learned{'' if safety == 'claim' else ' (no hard gate)'} (lam={lam:g})"
        self.log = None

    def reset(self, env):
        super().reset(env)
        if self.behaviour is not None:
            self.behaviour.reset(env)
            self.behaviour.mem = self.mem    # share memory
            self.behaviour.checked = self.checked

    def proba(self, a, x):
        mdl = self.models[a]
        tr = getattr(mdl, "tree_", None)
        if tr is not None:              # direct traversal on the exact features (~microseconds)
            node, L, R, F, T = 0, tr.children_left, tr.children_right, tr.feature, tr.threshold
            while L[node] != -1:
                node = L[node] if x[F[node]] <= T[node] else R[node]
            v = tr.value[node][0]
            return dict(zip(mdl.classes_, v / v.sum()))
        # non-tree models: features rounded to 1 decimal so repeated states hit the cache
        key = (self.models.get("tag"), a, tuple(round(v, 1) for v in x))
        p = _PROBA_CACHE.get(key)
        if p is None:
            p = dict(zip(mdl.classes_, mdl.predict_proba(np.asarray([key[2]], dtype=float))[0]))
            _PROBA_CACHE[key] = p
        return p

    def expected_reward(self, x, feasible):
        R = {"correct": 1.0, "wrong": -self.lam, "correct_abstain": 1.0, "unneeded_abstain": 0.0}
        out = {}
        for a in feasible:
            p = self.proba(a, x)
            out[a] = sum(pi * R[cl] for cl, pi in p.items()) - (self.c if a == REFRESH else 0.0)
        return out

    def act(self, q):
        t, qtype, ref, as_of = q
        if qtype == "historical":       # not learned: as-of retrieval + gate
            ans, kid = self.fetch(q)
            if self.safety == "claim" and self.blocked_after_fetch(kid, ans, as_of):
                return None, kid, 1
            return ans, kid, 1
        m = self.mem.get(self.qkey(q))
        if self.safety == "claim":
            r = self.claim_precheck(q, m)
            if r is not None:
                return r
        kid_hint = ref if qtype == "current" else (m[3] if m else None)
        x = featurize(self, q, kid_hint)
        feasible = [REUSE, REFRESH, ABSTAIN] if m is not None else [REFRESH, ABSTAIN]
        if self.log is not None:            # counterfactual one-step outcomes for training
            tr, _ = self.env.query_truth(q)
            if tr[0] != "review":
                cf = {}
                if m is not None:
                    cf[REUSE] = outcome(self.env, tr, (m[0], m[1]), as_of)
                ans_f, kid_f = self.fetch(q)
                if self.safety == "claim" and self.blocked_after_fetch(kid_f, ans_f, t):
                    ans_f = None
                cf[REFRESH] = outcome(self.env, tr, ans_f, as_of)
                cf[ABSTAIN] = outcome(self.env, tr, None, as_of)
                self.log.append((q[0], x, cf))
        if self.models is None:
            a = self.behaviour_action(q, m)
        else:
            er = self.expected_reward(x, feasible)
            a = max(er, key=er.get)
        if self.explore and self.rng.random() < self.explore:
            a = self.rng.choice(feasible)
        if a == REUSE:
            return (m[0], m[1]), m[3], 0
        if a == ABSTAIN:
            return None, kid_hint, 0
        ans, kid = self.fetch(q)
        if self.safety == "claim" and self.blocked_after_fetch(kid, ans, t):
            self.checked[self.qkey(q)] = t
            return None, kid, 1
        self.store(q, ans, kid)
        return ans, kid, 1

    def behaviour_action(self, q, m):
        b = self.behaviour
        if m is not None and not b.suspect(m, q[0]):
            return REUSE
        return REFRESH


# ---------------------------------------------------------------- replay
def stratum(env, tr, kid, as_of):
    kind = tr[0]
    if kind == "abstain_withdrawn":
        return "nr_gap"
    if kind == "abstain_undisclosed":
        return "undisclosed"
    if kind == "review":
        return "review"
    return "revised" if env.revised_before(kid, as_of) else "stable"


COLS = ("n", "answered", "correct", "supported", "correct_abstain", "stale", "leak", "withdrawn", "calls")


def run_issuer(iss: Issuer, policies, collect=False, modes=LABEL_MODES):
    """Replay one issuer. Every policy acts on every query (so the query stream never
    depends on labels); each decision is scored under every label mode. Returns counts
    keyed by (policy, cik, split, qtype, stratum, mode)."""
    env = Env(iss)
    for p in policies:
        p.reset(env)
        if collect:
            p.log = []
    agg = defaultdict(lambda: np.zeros(len(COLS), dtype=np.int64))
    for q in sorted(iss.queries):
        trs = []
        for mode in modes:
            tr, kid_true = env.query_truth(q, mode)
            if tr[0] != "review":
                trs.append((mode, tr, stratum(env, tr, kid_true, q[3])))
        for p in policies:
            ans, kid, cost = p.act(q)
            leak = int(ans is not None and env.accn_avail.get(ans[1], 0) > q[3])
            for mode, tr, st in trs:
                o = outcome(env, tr, ans, q[3])
                supported = int(o == "correct" and ans[1] in tr[2])
                stale = int(o == "wrong" and tr[0] == "answer" and not leak)
                withdrawn = int(ans is not None and tr[0] == "abstain_withdrawn")
                v = agg[(p.name, iss.cik, iss.split, q[1], st, mode)]
                v += (1, int(ans is not None), int(o == "correct"), supported, int(o == "correct_abstain"),
                      stale, leak, withdrawn, cost)
    logs = {p.name: p.log for p in policies if collect and getattr(p, "log", None) is not None}
    return dict(agg), logs
