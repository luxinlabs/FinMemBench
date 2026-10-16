"""Independent audit of automatically labelled non-reliance episodes against EDGAR documents.

For a stratified sample of (issuer, Item 4.02 notice, claim key) episodes it downloads
  (1) the 8-K/8-K/A that carried Item 4.02, and
  (2) the first post-notice filing that re-reported the claim (if any),
and extracts the Item 4.02 text plus "restat*" passages for human/LLM-free review.

  python -m finmemguard.audit_nr --ua "Name email@domain" [--n_affected 30 --n_same 10 --n_none 10]

Respects SEC fair access: declared User-Agent, <= 4 requests/second.
"""
import argparse
import html as htmlmod
import json
import os
import pickle
import random
import re
import time
import urllib.request
import zipfile
from datetime import date

UA = None
_last = [0.0]


def get(url):
    wait = 0.25 - (time.time() - _last[0])
    if wait > 0:
        time.sleep(wait)
    _last[0] = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "identity"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", errors="replace")


def text_of(h):
    h = re.sub(r"(?is)<ix:header>.*?</ix:header>", " ", h)   # hidden inline-XBRL contexts
    h = re.sub(r"(?is)<(script|style).*?</\1>", " ", h)
    h = re.sub(r"(?s)<[^>]+>", " ", h)
    return re.sub(r"\s+", " ", htmlmod.unescape(h))


def primary_docs(sub_zip, cik, accns):
    """accession -> primaryDocument from the submissions archive (recent + paged files)."""
    out = {}
    z = zipfile.ZipFile(sub_zip)
    names = [f"CIK{cik:010d}.json"]
    d = json.loads(z.read(names[0]))
    blocks = [d["filings"]["recent"]] + [json.loads(z.read(f["name"])) for f in d["filings"].get("files", [])]
    for b in blocks:
        for a, p in zip(b["accessionNumber"], b["primaryDocument"]):
            if a in accns:
                out[a] = p
    return out


def snippet(text, pattern, width=350, k=3):
    out = []
    for m in re.finditer(pattern, text, flags=re.I):
        out.append(text[max(0, m.start() - width): m.end() + width])
        if len(out) >= k:
            break
    return out


def refetch_restating(path):
    """Re-download only the restating documents of an existing audit file (improved parsing)."""
    d = json.load(open(path))
    for e in d:
        url = e.get("restating_url")
        if not url:
            continue
        raw = get(url)
        t = text_of(raw)
        e["restating_snippets"] = snippet(t, r"as restated|restatement|restated|as previously reported")
        e["restating_len"] = len(t)
        e["xbrl_restatement_markers"] = len(re.findall(r"RestatementAdjustmentMember|ScenarioPreviouslyReportedMember", raw))
        nm = re.search(r'name="dei:EntityRegistrantName"[^>]*>(.*?)</ix:nonNumeric>', raw, re.S)
        e["restating_registrant"] = text_of(nm.group(1)).strip() if nm else None
        if e.get("notice_url"):
            t2 = text_of(get(e["notice_url"]))
            nm2 = re.search(r"\(Exact name of registrant[^)]*\)", t2, re.I)
            e["notice_registrant_ctx"] = t2[max(0, nm2.start() - 200): nm2.start()] if nm2 else t2[:300]
    json.dump(d, open(path, "w"), indent=1)


def main():
    global UA
    ap = argparse.ArgumentParser()
    ap.add_argument("--ua", required=True)
    ap.add_argument("--bench", default="data/bench")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="results/nr_audit")
    ap.add_argument("--name", default="nr_audit_raw.json")
    ap.add_argument("--n_affected", type=int, default=30)
    ap.add_argument("--n_same", type=int, default=10)
    ap.add_argument("--n_none", type=int, default=10)
    ap.add_argument("--n_merger", type=int, default=0)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--exclude", default=None, help="audit JSON whose issuers must not be re-sampled")
    ap.add_argument("--labels", default="abstain,review_same,review_none,review_merger")
    a = ap.parse_args()
    UA = a.ua
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)
    issuers = pickle.load(open(os.path.join(a.bench, "issuers.pkl"), "rb"))
    pref = ("NetIncomeLoss", "StockholdersEquity", "Liabilities", "Assets", "Revenues",
            "RetainedEarningsAccumulatedDeficit", "OperatingIncomeLoss")
    pools = {lab: [] for lab in a.labels.split(",")}
    excl = {e["cik"] for f in (a.exclude.split(",") if a.exclude else []) for e in json.load(open(f))}
    for iss in issuers:
        if iss.cik in excl:
            continue
        nr_acc = {d: acc for d, f, isnr, acc in iss.notices if isnr}
        per = {}
        for kid, k in iss.keys.items():
            for lo, hi, lab in k.gaps:
                if lo in nr_acc and lab in pools:
                    score = (k.concept in pref, k.kind == "annual")
                    if (lab, lo) not in per or score > per[(lab, lo)][0]:
                        per[(lab, lo)] = (score, kid, k, lo, hi)
        for (lab, lo), (_, kid, k, lo_, hi) in per.items():
            i = k.version_at(lo - 1)
            j = i + 1 if i + 1 < len(k.avail) and k.avail[i + 1] <= hi else None
            pools[lab].append(dict(cik=iss.cik, split=iss.split, label=lab, claim_key=kid,
                                   notice_day=date.fromordinal(lo - 1).isoformat(), notice_accn=nr_acc[lo],
                                   old_val=k.vals[i], old_accn=k.accns[i],
                                   new_val=k.vals[j] if j is not None else None,
                                   new_accn=k.accns[j] if j is not None else None,
                                   new_form=k.forms[j] if j is not None else None))
    sample = []
    quota = {"abstain": a.n_affected, "review_same": a.n_same, "review_none": a.n_none, "review_merger": a.n_merger}
    for lab, n in quota.items():
        if lab not in pools:
            continue
        # one episode per issuer
        seen, pool = set(), pools[lab][:]
        rng.shuffle(pool)
        for e in pool:
            if e["cik"] not in seen:
                seen.add(e["cik"]); sample.append(e)
            if len([x for x in sample if x["label"] == lab]) >= n:
                break
    print({k: len(v) for k, v in pools.items()}, "sampled", len(sample), flush=True)
    for n_, e in enumerate(sample):
        accns = {e["notice_accn"]} | ({e["new_accn"]} if e["new_accn"] else set())
        docs = primary_docs(os.path.join(a.raw, "submissions.zip"), e["cik"], accns)
        for role, acc in (("notice", e["notice_accn"]), ("restating", e["new_accn"])):
            if not acc:
                continue
            url = f"https://www.sec.gov/Archives/edgar/data/{e['cik']}/{acc.replace('-', '')}/{docs.get(acc, '')}"
            e[f"{role}_url"] = url
            try:
                raw = get(url)
                t = text_of(raw)
            except Exception as ex:          # noqa: BLE001 - record and continue
                e[f"{role}_error"] = repr(ex)[:200]
                continue
            if role == "notice":
                i = t.lower().find("4.02")
                e["notice_text"] = t[max(0, i - 200): i + 2500] if i >= 0 else t[:1500]
                e["notice_has_402"] = i >= 0
            else:
                e["restating_snippets"] = snippet(t, r"as restated|restatement|restated|as previously reported")
                e["restating_len"] = len(t)
                e["xbrl_restatement_markers"] = len(re.findall(r"RestatementAdjustmentMember|ScenarioPreviouslyReportedMember", raw))
                nm = re.search(r'name="dei:EntityRegistrantName"[^>]*>(.*?)</ix:nonNumeric>', raw, re.S)
                e["restating_registrant"] = text_of(nm.group(1)).strip() if nm else None
        print(n_ + 1, e["cik"], e["label"], e["claim_key"], flush=True)
    json.dump(sample, open(os.path.join(a.out, a.name), "w"), indent=1)


if __name__ == "__main__":
    main()
