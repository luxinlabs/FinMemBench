"""Parser and merger-filter checks on short synthetic Item 4.02 texts."""
from collections import defaultdict
from datetime import date

from finmemguard.benchmark import Key, label_gaps
from finmemguard.notice_parse import parse_notice


def d(y, m, dd):
    return date(y, m, dd).toordinal()


def key(start, end, kind):
    return Key("NetIncomeLoss", "USD", start, end, kind)


NOTICE_DAY = d(2022, 3, 1)


def test_named_quarters_and_year_lists():
    t = ("4.02 Non-Reliance. The Company concluded that its financial statements for the quarters ended "
         "June 30 and September 30, 2021 and the years ended December 31, 2020 and 2019 should no longer be relied upon.")
    c = parse_notice(t, NOTICE_DAY)
    assert c.covers(key("2021-07-01", "2021-09-30", "quarter"), NOTICE_DAY)
    assert c.covers(key("2019-01-01", "2019-12-31", "annual"), NOTICE_DAY)
    assert not c.covers(key("2021-01-01", "2021-03-31", "quarter"), NOTICE_DAY)


def test_quarters_within_fiscal_year_boundary():
    t = "4.02 ... for each of the quarters in the year ended December 31, 2018 should no longer be relied upon."
    c = parse_notice(t, d(2019, 6, 1))
    assert c.covers(key("2018-04-01", "2018-06-30", "quarter"), d(2019, 6, 1))
    assert not c.covers(Key("Assets", "USD", None, "2017-12-31", "instant"), d(2019, 6, 1))


def test_negated_notice_covers_nothing_and_unparsed_falls_back():
    neg = parse_notice("4.02 The Company is not aware of any basis to believe that statements should not be relied upon.")
    assert not neg.covers(key("2020-01-01", "2020-12-31", "annual"), NOTICE_DAY)
    unparsed = parse_notice("4.02 Previously issued statements should no longer be relied upon.", NOTICE_DAY)
    assert unparsed.covers(key("2015-01-01", "2015-12-31", "annual"), NOTICE_DAY)


def test_reverse_merger_change_is_not_affected():
    k = Key("StockholdersEquity", "USD", None, "2020-09-30", "instant")
    k.avail, k.vals, k.accns, k.forms = [d(2020, 11, 17), d(2021, 11, 16)], [5000003.0, -2.0e8], ["a", "b"], ["10-Q", "10-Q"]
    st = defaultdict(int)
    label_gaps({"x": k}, [d(2021, 7, 3)], st, nr_cov=None, merger_days=[d(2021, 9, 1)])
    assert k.gaps and k.gaps[0][2] == "review_merger" and st["nr_merger_keys"] == 1


def test_rerounding_is_not_a_restatement():
    from finmemguard.benchmark import rerounded
    assert rerounded(19128061.0, 19128000.0) and rerounded(78522311.0, 78522000.0)
    assert not rerounded(100.0, 90.0) and not rerounded(5000003.0, -50967603.0)


def test_named_fiscal_year_covers_its_interim_periods():
    t = ("4.02 ... the financial statements contained in the Company's Annual Report on Form 10-K for the year ended "
         "December 31, 2013 should no longer be relied upon.")
    nd = d(2015, 3, 16)
    c = parse_notice(t, nd)
    assert c.covers(key("2013-04-01", "2013-06-30", "quarter"), nd)          # interim of the named year
    assert c.covers(Key("Assets", "USD", None, "2013-09-30", "instant"), nd)
    assert not c.covers(key("2012-04-01", "2012-06-30", "quarter"), nd)      # prior year not named
