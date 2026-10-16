"""Parse which reporting periods an 8-K Item 4.02 notice names.

The parser only uses the notice text, which is public on the notice day, so a controller
may use it without look-ahead. It returns a NoticeCoverage that answers covers(key).
If the text cannot be parsed (no dates found), coverage falls back to "every period that
ended before the notice", the conservative rule used previously.
"""
import re
from dataclasses import dataclass, field
from datetime import date

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september",
     "october", "november", "december"], 1)}
MON_ABBR = {m[:3]: i for m, i in MONTHS.items()}
MD = r"(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\.?\s+(\d{1,2})"
NEGATION = re.compile(r"not aware of any basis|no basis to believe|is not required to (?:file|restate)|"
                      r"should continue to be relied upon", re.I)


def _mon(s):
    s = s.lower().rstrip(".")
    return MONTHS.get(s) or MON_ABBR.get(s[:3])


@dataclass
class NoticeCoverage:
    parsed: bool = False
    negated: bool = False
    dates: set = field(default_factory=set)          # ordinal days of named period ends / balance dates
    interim_years: list = field(default_factory=list)  # (fy_start_day, fy_end_day) whose quarters are covered
    fiscal_years: set = field(default_factory=set)   # bare 'fiscal 2019' style years

    def covers(self, k, notice_day):
        if self.negated:
            return False
        if not self.parsed:
            return k.end_day < notice_day - 1
        e = k.end_day
        if any(abs(e - d) <= 7 for d in self.dates):
            return True
        if any(lo < e <= hi + 7 for lo, hi in self.interim_years):
            return True
        if k.kind == "annual" and date.fromordinal(e).year in self.fiscal_years:
            return True
        return False


def parse_notice(text, notice_day=None):
    """Extract named period-end dates from Item 4.02 text."""
    cov = NoticeCoverage()
    t = re.sub(r"\s+", " ", text or "")
    # stop at the next item / signature block
    m = re.search(r"\b(item\s*(?:5|7|8|9)\.\d\d|signatures?)\b", t[200:], re.I)
    if m:
        t = t[: 200 + m.start()]
    cov.negated = bool(NEGATION.search(t))
    # month-day tokens followed (within ~60 chars) by a year; lists like "March 31, June 30 and
    # September 30, 2021" share the year; "December 31, 2020 and 2019" repeats the month-day.
    for mm in re.finditer(r"(?i)" + MD, t):
        mon, day = _mon(mm.group(1)), int(mm.group(2))
        tail = t[mm.end(): mm.end() + 60]
        ym = re.match(r"\s*,?\s*(\d{4})((?:\s*(?:,|and|&)\s*\d{4})*)", tail)
        if ym:
            years = [int(ym.group(1))] + [int(y) for y in re.findall(r"\d{4}", ym.group(2) or "")]
        else:   # list member: borrow the next year that appears soon after
            ny = re.search(r"\b(19|20)\d{2}\b", tail)
            if not ny:
                continue
            years = [int(ny.group(0))]
        for y in years:
            try:
                cov.dates.add(date(y, mon, min(day, 28) if mon == 2 and day > 28 else day).toordinal())
            except ValueError:
                continue
    # interim / quarterly language attached to a fiscal year: cover the quarters of that year
    # quarters explicitly tied to a fiscal year ("each of the quarters in the year ended ...",
    # "interim periods within fiscal year ended ...")
    for mm in re.finditer(r"(?i)(?:quarters|quarterly periods|interim periods)[^.;]{0,80}?(?:fiscal\s+)?years?\s+ended\s+"
                          + MD + r"\s*,?\s*(\d{4})", t):
        mon, day, y = _mon(mm.group(1)), int(mm.group(2)), int(mm.group(3))
        try:
            end = date(y, mon, day).toordinal()
            cov.interim_years.append((end - 358, end))     # strictly after the prior fiscal year end
        except ValueError:
            pass
    # a named fiscal YEAR also covers the interim periods inside it: restating a year restates its
    # quarters (recall audit: 7 of 15 uncovered-but-changed claims were such interim periods).
    # Matches "year(s) ended December 31, 2020 (and 2019)" and "10-K for the (fiscal) year ended ...".
    for mm in re.finditer(r"(?i)(?:fiscal\s+)?(?:years?|periods?)\s+(?:then\s+)?end(?:ed|ing)\s+" + MD +
                          r"\s*,?\s*(\d{4})((?:\s*(?:,|and|&)\s*\d{4})*)", t):
        mon, day = _mon(mm.group(1)), int(mm.group(2))
        years = [int(mm.group(3))] + [int(y) for y in re.findall(r"\d{4}", mm.group(4) or "")]
        ctx = t[max(0, mm.start() - 80): mm.start()].lower()
        if "quarter" in ctx and "year" not in mm.group(0).lower():
            continue
        if "year" not in mm.group(0).lower() and "10-k" not in ctx and "annual" not in ctx:
            continue
        ends = [(mon, day, y) for y in years]
        # a list of full dates after "years ended": "October 3, 2020, October 2, 2021 and October 1, 2022"
        tail = t[mm.end(): mm.end() + 120]
        for m2 in re.finditer(r"(?i)^\s*(?:,|and|&)?\s*(?:and\s+)?" + MD + r"\s*,?\s*(\d{4})", tail):
            ends.append((_mon(m2.group(1)), int(m2.group(2)), int(m2.group(3))))
        rest = tail
        while True:
            m2 = re.match(r"(?i)\s*(?:,|and|&)\s*(?:and\s+)?" + MD + r"\s*,?\s*(\d{4})", rest)
            if not m2:
                break
            ends.append((_mon(m2.group(1)), int(m2.group(2)), int(m2.group(3))))
            rest = rest[m2.end():]
        for mo, dy, y in set(ends):
            try:
                end = date(y, mo, min(dy, 28) if mo == 2 and dy > 28 else dy).toordinal()
                cov.interim_years.append((end - 358, end))
            except ValueError:
                pass
    # "each of the quarterly periods in 2022 and 2023" / "quarters of 2019": calendar-year quarters
    for mm in re.finditer(r"(?i)(?:quarters|quarterly periods|interim periods)\s+(?:in|of|during)\s+((?:19|20)\d{2})((?:\s*(?:,|and|&)\s*(?:19|20)\d{2})*)", t):
        for y in [int(mm.group(1))] + [int(v) for v in re.findall(r"(?:19|20)\d{2}", mm.group(2) or "")]:
            cov.interim_years.append((date(y, 1, 1).toordinal(), date(y, 12, 31).toordinal()))
    for mm in re.finditer(r"(?i)\bfiscal(?:\s+years?)?\s+((?:19|20)\d{2})((?:\s*(?:,|and|&|through|-)\s*(?:19|20)\d{2})*)", t):
        ys = [int(mm.group(1))] + [int(y) for y in re.findall(r"(?:19|20)\d{2}", mm.group(2) or "")]
        if "through" in (mm.group(2) or "") and len(ys) == 2:
            ys = list(range(ys[0], ys[1] + 1))
        cov.fiscal_years.update(ys)
    # drop the notice's own filing/event dates (within 60 days of the notice) — they are not periods
    if notice_day is not None:
        cov.dates = {d for d in cov.dates if d < notice_day - 60}
    cov.parsed = bool(cov.dates or cov.interim_years or cov.fiscal_years)
    return cov
