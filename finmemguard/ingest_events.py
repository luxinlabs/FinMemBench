"""Corporate-event metadata for the label fixes (reverse mergers) and the Item 4.02 notice texts.

  python -m finmemguard.ingest_events events   # 8-K Item 2.01/5.06 filings + name history, from submissions.zip
  python -m finmemguard.ingest_events notices --ua "Name email"   # download Item 4.02 text of in-window notices

Outputs (data/interim/):
  events.parquet      cik, accn, filing_date, items  (8-K/8-K/A with Item 2.01 or 5.06)
  names.parquet       cik, name, date_from, date_to  (formerNames history)
  nr_texts.jsonl      one JSON per Item 4.02 notice: cik, accn, filing_date, text (Item 4.02 section)
"""
import argparse
import json
import os
import pickle
import zipfile

import orjson
import pandas as pd

from .audit_nr import get, text_of
from . import audit_nr


def events(raw, interim, bench):
    issuers = pickle.load(open(os.path.join(bench, "issuers.pkl"), "rb"))
    ciks = {i.cik for i in issuers}
    z = zipfile.ZipFile(os.path.join(raw, "submissions.zip"))
    ev, names = [], []
    for cik in sorted(ciks):
        try:
            d = orjson.loads(z.read(f"CIK{cik:010d}.json"))
        except KeyError:
            continue
        for n in d.get("formerNames", []) or []:
            names.append((cik, n.get("name"), (n.get("from") or "")[:10], (n.get("to") or "")[:10]))
        blocks = [d["filings"]["recent"]] + [orjson.loads(z.read(f["name"])) for f in d["filings"].get("files", [])]
        for b in blocks:
            items = b.get("items") or [""] * len(b["form"])
            for a, f, fd, it in zip(b["accessionNumber"], b["form"], b["filingDate"], items):
                if f in ("8-K", "8-K/A") and it and ("2.01" in it or "5.06" in it):
                    ev.append((cik, a, fd, it))
    pd.DataFrame(ev, columns=["cik", "accn", "filing_date", "items"]).drop_duplicates(["cik", "accn"]).to_parquet(
        os.path.join(interim, "events.parquet"), index=False)
    pd.DataFrame(names, columns=["cik", "name", "date_from", "date_to"]).to_parquet(
        os.path.join(interim, "names.parquet"), index=False)
    print("events", len(ev), "name changes", len(names))


def notices(raw, interim, ua, workers=4):
    """Download Item 4.02 sections; resumable; <=5 requests/s across workers (SEC allows 10)."""
    import threading
    from concurrent.futures import ThreadPoolExecutor
    import time as _t
    audit_nr.UA = ua
    f = pd.read_parquet(os.path.join(interim, "filings.parquet"))
    nr = f[f.non_reliance & (f.filing_date >= "2015-01-01") & (f.filing_date <= "2026-06-30")]
    issuers = pickle.load(open("data/bench/issuers.pkl", "rb"))
    nr = nr[nr.cik.isin({i.cik for i in issuers})]
    out_path = os.path.join(interim, "nr_texts.jsonl")
    done = {json.loads(l)["accn"] for l in open(out_path)} if os.path.exists(out_path) else set()
    z = zipfile.ZipFile(os.path.join(raw, "submissions.zip"))
    jobs = []
    for cik, g in nr.groupby("cik"):
        todo = [r for r in g.itertuples() if r.accn not in done]
        if not todo:
            continue
        d = orjson.loads(z.read(f"CIK{cik:010d}.json"))
        blocks = [d["filings"]["recent"]] + [orjson.loads(z.read(x["name"])) for x in d["filings"].get("files", [])]
        prim = {a: p for b in blocks for a, p in zip(b["accessionNumber"], b["primaryDocument"])}
        jobs += [(int(cik), r.accn, r.filing_date, prim.get(r.accn, "")) for r in todo]
    print("to fetch", len(jobs), flush=True)
    lock, gate = threading.Lock(), threading.Lock()
    last = [0.0]

    def fetch(job):
        cik, accn, fdate, doc = job
        with gate:                                   # global rate limit
            w = 0.2 - (_t.time() - last[0])
            if w > 0:
                _t.sleep(w)
            last[0] = _t.time()
        url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accn.replace('-', '')}/{doc}"
        try:
            req = audit_nr.urllib.request.Request(url, headers={"User-Agent": ua})
            with audit_nr.urllib.request.urlopen(req, timeout=60) as r:
                t = text_of(r.read().decode("utf-8", errors="replace"))
        except Exception as ex:  # noqa: BLE001
            t = "__ERROR__ " + repr(ex)[:100]
        i = t.find("4.02")
        rec = {"cik": cik, "accn": accn, "filing_date": fdate, "has_402": i >= 0,
               "text": t[i: i + 5000] if i >= 0 else t[:5000]}
        with lock:
            with open(out_path, "a") as out:
                out.write(json.dumps(rec) + "\n")

    with ThreadPoolExecutor(workers) as ex:
        for n, _ in enumerate(ex.map(fetch, jobs)):
            if n % 200 == 0:
                print(n, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("events", "notices"))
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--interim", default="data/interim")
    ap.add_argument("--bench", default="data/bench")
    ap.add_argument("--ua")
    a = ap.parse_args()
    if a.what == "events":
        events(a.raw, a.interim, a.bench)
    else:
        notices(a.raw, a.interim, a.ua)


if __name__ == "__main__":
    main()
