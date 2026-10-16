from datetime import datetime, timezone
from finmemguard.models import Evidence, Query
from finmemguard.controller import decide

def t(y,m,d): return datetime(y,m,d,tzinfo=timezone.utc)

def test_history_and_correction():
    ev=[Evidence("a","k","100",t(2025,1,1)), Evidence("b","k","90",t(2025,2,1),"supersede")]
    assert decide(ev,Query("k",t(2025,1,15))).value=="100"
    assert decide(ev,Query("k",t(2025,3,1))).value=="90"
    assert decide(ev,Query("k",t(2024,12,1))).action=="abstain"

def test_non_reliance():
    ev=[Evidence("a","k","100",t(2025,1,1)),Evidence("b","k","",t(2025,2,1),"non_reliance")]
    assert decide(ev,Query("k",t(2025,3,1))).action=="abstain"

import pytest
from finmemguard.evaluation import score, values_equal
from finmemguard.models import Decision

def test_boundary_is_inclusive():
    ev=[Evidence("a","k","100",t(2025,1,1))]
    assert decide(ev,Query("k",t(2025,1,1))).value=="100"

def test_simultaneous_conflict_abstains():
    ev=[Evidence("a","k","100",t(2025,1,1)),Evidence("b","k","90",t(2025,1,1))]
    d=decide(ev,Query("k",t(2025,2,1)))
    assert d.action=="abstain" and d.reason=="conflicting simultaneous evidence"

def test_simultaneous_agreement_answers():
    ev=[Evidence("b","k","100",t(2025,1,1)),Evidence("a","k","100",t(2025,1,1))]
    assert decide(ev,Query("k",t(2025,2,1))).value=="100"

def test_restatement_after_non_reliance_answers():
    ev=[Evidence("a","k","100",t(2025,1,1)),Evidence("b","k","",t(2025,2,1),"non_reliance"),
        Evidence("c","k","95",t(2025,3,1),"supersede")]
    assert decide(ev,Query("k",t(2025,2,15))).action=="abstain"
    assert decide(ev,Query("k",t(2025,3,2))).value=="95"

def test_naive_datetime_and_bad_status_rejected():
    with pytest.raises(ValueError):
        Evidence("a","k","1",datetime(2025,1,1))
    with pytest.raises(ValueError):
        Evidence("a","k","1",t(2025,1,1),"restated")

def test_score_numeric_and_support():
    d=Decision("answer","90.0","b","x")
    assert values_equal("90","90.0") and not values_equal("90","91")
    assert score(d,"90")["correct"]
    assert not score(d,"90",admissible_accessions=["a"])["supported"]
    assert score(Decision("abstain",None,None,"x"),None)["appropriate_abstention"]
    assert not score(Decision("abstain",None,None,"x"),"90")["correct"]
