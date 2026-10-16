"""Parse official SEC bulk archives into two flat tables (Parquet).

Inputs (download separately, see data_manifest/README.md):
  data/raw/companyfacts.zip   https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip
  data/raw/submissions.zip    https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip

Outputs:
  data/interim/facts.parquet    one row per (issuer, concept, unit, period, accession) report
  data/interim/filings.parquet  periodic filings and 8-K Item 4.02 non-reliance notices

Run: python -m finmemguard.ingest_sec [--raw data/raw] [--out data/interim]
"""
import argparse
import os
import zipfile
from concurrent.futures import ProcessPoolExecutor

import orjson
import pandas as pd

# Core primary-statement concepts. Kept small on purpose: these are the facts an
# analyst agent is most likely to memorise and re-use.
CONCEPTS = (
    "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "NetIncomeLoss",
    "OperatingIncomeLoss", "GrossProfit", "IncomeTaxExpenseBenefit",
    "EarningsPerShareBasic", "EarningsPerShareDiluted",
    "Assets", "AssetsCurrent", "Liabilities", "LiabilitiesCurrent",
    "StockholdersEquity", "LiabilitiesAndStockholdersEquity",
    "RetainedEarningsAccumulatedDeficit", "CashAndCashEquivalentsAtCarryingValue",
    "NetCashProvidedByUsedInOperatingActivities",
)
PERIODIC_FORMS = ("10-K", "10-K/A", "10-Q", "10-Q/A")
MIN_FILED = "2014-01-01"
MIN_END = "2013-01-01"


def _facts_one(args):
    path, name = args
    with zipfile.ZipFile(path) as z:
        d = orjson.loads(z.read(name))
    gaap = d.get("facts", {}).get("us-gaap", {})
    cik = int(d.get("cik", 0) or name[3:13])
    rows = []
    for concept in CONCEPTS:
        units = gaap.get(concept, {}).get("units", {})
        for unit, xs in units.items():
            for x in xs:
                if x.get("form") not in PERIODIC_FORMS:
                    continue
                if x.get("filed", "") < MIN_FILED or x.get("end", "") < MIN_END:
                    continue
                rows.append((cik, concept, unit, x.get("start"), x["end"], float(x["val"]),
                             x["accn"], x["form"], x["filed"], x.get("fy"), x.get("fp")))
    return rows


def _subs_one(args):
    path, names = args
    out = []
    with zipfile.ZipFile(path) as z:
        for name in names:
            try:
                d = orjson.loads(z.read(name))
            except KeyError:
                continue
            block = d["filings"]["recent"] if "filings" in d else d
            cik = int(name[3:13])
            n = len(block.get("accessionNumber", []))
            items = block.get("items") or [""] * n
            for i in range(n):
                form = block["form"][i]
                it = items[i] or ""
                is_nr = form in ("8-K", "8-K/A") and "4.02" in it
                if form in PERIODIC_FORMS or is_nr:
                    out.append((cik, block["accessionNumber"][i], form, block["filingDate"][i],
                                block.get("acceptanceDateTime", [""] * n)[i], it, is_nr))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/interim")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    cf = os.path.join(a.raw, "companyfacts.zip")
    names = [n for n in zipfile.ZipFile(cf).namelist() if n.endswith(".json")]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = [r for chunk in ex.map(_facts_one, [(cf, n) for n in names], chunksize=64) for r in chunk]
    facts = pd.DataFrame(rows, columns=["cik", "concept", "unit", "start", "end", "val", "accn",
                                        "form", "filed", "fy", "fp"])
    facts.to_parquet(os.path.join(a.out, "facts.parquet"), index=False)
    print("facts", len(facts), "issuers", facts.cik.nunique())

    sb = os.path.join(a.raw, "submissions.zip")
    ciks = set(facts.cik.unique())
    all_names = zipfile.ZipFile(sb).namelist()
    wanted = [n for n in all_names if n.startswith("CIK") and int(n[3:13]) in ciks]
    batches = [(sb, wanted[i:i + 500]) for i in range(0, len(wanted), 500)]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = [r for chunk in ex.map(_subs_one, batches) for r in chunk]
    filings = pd.DataFrame(rows, columns=["cik", "accn", "form", "filing_date", "acceptance",
                                          "items", "non_reliance"]).drop_duplicates(["cik", "accn"])
    filings.to_parquet(os.path.join(a.out, "filings.parquet"), index=False)
    print("filings", len(filings), "non-reliance notices", int(filings.non_reliance.sum()))


if __name__ == "__main__":
    main()
