"""Synthetic end-to-end checks of the replay simulator (no SEC data)."""
from datetime import date

from finmemguard import simulate as S
from finmemguard.benchmark import Issuer, Key, label_gaps, truth


def d(y, m, dd):
    return date(y, m, dd).toordinal()


def make_issuer():
    # FY2020 revenue: 100 in the 10-K, withdrawn by an 8-K 4.02, restated to 90 in a 10-K/A.
    k = Key("Revenues", "USD", "2020-01-01", "2020-12-31", "annual")
    k.avail = [d(2021, 3, 1), d(2021, 8, 1)]
    k.vals = [100.0, 90.0]
    k.accns = ["10K-2020", "10KA-2020"]
    k.forms = ["10-K", "10-K/A"]
    keys = {"Revenues|USD|2020-01-01|2020-12-31": k}
    nr = [d(2021, 6, 1)]
    from collections import defaultdict
    stats = defaultdict(int)
    label_gaps(keys, nr, stats)
    notices = [(d(2021, 3, 1), "10-K", False, "10K-2020"), (d(2021, 5, 1), "10-Q", False, "q1"),
               (d(2021, 6, 1), "8-K", True, "nr"), (d(2021, 8, 1), "10-K/A", False, "10KA-2020")]
    return Issuer(1, "test", keys, notices, nr), stats


def test_gap_labels():
    iss, stats = make_issuer()
    k = next(iter(iss.keys.values()))
    assert stats["nr_affected_keys"] == 1
    assert truth(k, d(2021, 2, 1))[0] == "abstain_undisclosed"
    assert truth(k, d(2021, 4, 1))[1] == 100.0
    assert truth(k, d(2021, 7, 1))[0] == "abstain_withdrawn"
    assert truth(k, d(2021, 9, 1))[1] == 90.0


def run(policy, queries, iss=None, mode="primary"):
    if iss is None:
        iss, _ = make_issuer()
    kid = next(iter(iss.keys))
    iss.queries = [(t, "current", kid, t) for t in queries]
    agg, _ = S.run_issuer(iss, [policy])
    tot = sum(v for k, v in agg.items() if k[-1] == mode)
    return dict(zip(S.COLS, tot))


QS = [d(2021, 4, 1), d(2021, 7, 1), d(2021, 9, 1), d(2021, 10, 1)]


def test_time_filtered_rag_answers_withdrawn_value():
    r = run(S.TimeFilteredRAG(None), QS)
    assert r["withdrawn"] == 1 and r["supported"] == 3 and r["calls"] == 4


def test_gate_abstains_in_gap():
    r = run(S.TimeFilteredRAG("claim"), QS)
    assert r["withdrawn"] == 0 and r["correct_abstain"] == 1 and r["supported"] == 3


def test_naive_memory_is_stale_after_restatement():
    r = run(S.NaiveMemory(), QS)
    assert r["calls"] == 1 and r["stale"] == 2 and r["withdrawn"] == 1


def test_rule_controller_refreshes_only_after_new_filing():
    r = run(S.FinMemGuardRule(gate="rule"), QS)
    # Apr: fetch; Jul: quarantined -> abstain (free); Sep: suspect -> refresh 90; Oct: reuse
    assert r["supported"] == 3 and r["correct_abstain"] == 1 and r["calls"] == 2 and r["stale"] == 0


def test_snapshot_rag_leaks():
    r = run(S.SnapshotRAG(), QS)
    assert r["leak"] >= 1


def make_premature_lift_issuer():
    # a 10-K/A after the notice that does NOT re-report the withdrawn claim
    iss, _ = make_issuer()
    iss.notices = sorted(iss.notices + [(d(2021, 7, 1), "10-K/A", False, "amend-no-restate")])
    return iss


def test_rule_release_lifts_prematurely_but_claim_release_does_not():
    qs = [d(2021, 4, 1), d(2021, 6, 15), d(2021, 7, 15), d(2021, 9, 1)]
    rule = run(S.FinMemGuardRule(gate="rule"), qs, make_premature_lift_issuer())
    claim = run(S.FinMemGuardRule(gate="claim"), qs, make_premature_lift_issuer())
    assert rule["withdrawn"] == 1            # Jul 15: lifted by the unrelated 10-K/A
    assert claim["withdrawn"] == 0 and claim["correct_abstain"] == 2 and claim["supported"] == 2
    # claim release pays one re-check call on Jul 15 (a filing arrived) and one refresh in Sep
    assert claim["calls"] == 3


def test_label_modes_for_uncertain_gaps():
    k = Key("Assets", "USD", None, "2020-12-31", "instant")
    k.avail = [d(2021, 3, 1), d(2021, 8, 1)]
    k.vals = [50.0, 50.0]                    # value unchanged after the notice
    k.accns = ["a", "b"]; k.forms = ["10-K", "10-K/A"]
    from collections import defaultdict
    label_gaps({"x": k}, [d(2021, 6, 1)], defaultdict(int))
    t = d(2021, 7, 1)
    assert truth(k, t, "primary")[0] == "review"
    assert truth(k, t, "strict")[0] == "abstain_withdrawn"
    assert truth(k, t, "permissive")[1] == 50.0


def test_same_day_restatement_is_post_notice_evidence():
    k = Key("Revenues", "USD", "2020-01-01", "2020-12-31", "annual")
    k.avail = [d(2021, 3, 1), d(2021, 6, 1)]
    k.vals = [100.0, 90.0]
    k.accns = ["10K", "10KA-same-day"]; k.forms = ["10-K", "10-K/A"]
    from collections import defaultdict
    st = defaultdict(int)
    label_gaps({"x": k}, [d(2021, 6, 1)], st)
    assert st["nr_affected_sameday"] == 1 and k.gaps == []
    assert truth(k, d(2021, 6, 2))[1] == 90.0


def test_latest_annual_is_reresolved_before_reuse():
    # FY2020 is remembered; a new FY2021 10-K arrives whose value is then withdrawn.
    from collections import defaultdict
    k20 = Key("Revenues", "USD", "2020-01-01", "2020-12-31", "annual")
    k20.avail, k20.vals, k20.accns, k20.forms = [d(2021, 3, 1)], [100.0], ["k20"], ["10-K"]
    k21 = Key("Revenues", "USD", "2021-01-01", "2021-12-31", "annual")
    k21.avail, k21.vals, k21.accns, k21.forms = [d(2022, 3, 1), d(2022, 9, 1)], [120.0, 110.0], ["k21", "k21a"], ["10-K", "10-K/A"]
    keys = {"Revenues|USD|2020-01-01|2020-12-31": k20, "Revenues|USD|2021-01-01|2021-12-31": k21}
    label_gaps(keys, [d(2022, 5, 1)], defaultdict(int))
    notices = [(d(2021, 3, 1), "10-K", False, "k20"), (d(2022, 3, 1), "10-K", False, "k21"),
               (d(2022, 5, 1), "8-K", True, "nr"), (d(2022, 9, 1), "10-K/A", False, "k21a")]
    iss = Issuer(2, "test", keys, notices, [d(2022, 5, 1)])
    iss.queries = [(t, "latest_annual", "Revenues|USD", t) for t in (d(2021, 6, 1), d(2022, 6, 1), d(2022, 10, 1))]
    no_inv = S.FinMemGuardRule(gate="claim", inv="none")
    agg, _ = S.run_issuer(iss, [no_inv], modes=("primary",))
    tot = dict(zip(S.COLS, sum(agg.values())))
    assert tot["withdrawn"] == 0 and tot["correct_abstain"] == 1 and tot["supported"] == 2
