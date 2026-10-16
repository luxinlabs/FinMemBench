"""LLM-in-the-loop check: a frozen local LLM performs the REFRESH step.

For selected TEST issuers, every (claim, accession) that any refresh could return is
extracted by a frozen LLM (ollama) from the filing's rendered financial-statement page
(EDGAR R-pages listed in FilingSummary.xml). The replay is then repeated with an
environment whose fact index returns the LLM's extracted value instead of the ledger
value, so memory reuse propagates the LLM's own extraction errors. Scoring is against
the ledger, which separates controller errors from extraction errors.

  python -m finmemguard.llm_loop --ua "Name email" --ciks 1818212,... [--model qwen2.5:7b-instruct]
"""
import argparse
import glob
import json
import os
import pickle
import re
import time
import urllib.request
from datetime import date

import pandas as pd

from . import simulate as S
from .audit_nr import text_of
from .benchmark import same

LABEL = {
    "Revenues": "Total revenues", "RevenueFromContractWithCustomerExcludingAssessedTax": "Revenue (net sales)",
    "NetIncomeLoss": "Net income (loss)", "OperatingIncomeLoss": "Operating income (loss)",
    "GrossProfit": "Gross profit", "IncomeTaxExpenseBenefit": "Income tax expense (benefit)",
    "EarningsPerShareBasic": "Basic earnings (loss) per share", "EarningsPerShareDiluted": "Diluted earnings (loss) per share",
    "Assets": "Total assets", "AssetsCurrent": "Total current assets", "Liabilities": "Total liabilities",
    "LiabilitiesCurrent": "Total current liabilities", "StockholdersEquity": "Total stockholders' equity (deficit)",
    "LiabilitiesAndStockholdersEquity": "Total liabilities and stockholders' equity",
    "RetainedEarningsAccumulatedDeficit": "Retained earnings (accumulated deficit)",
    "CashAndCashEquivalentsAtCarryingValue": "Cash and cash equivalents",
    "NetCashProvidedByUsedInOperatingActivities": "Net cash provided by (used in) operating activities",
}
BS = {"Assets", "AssetsCurrent", "Liabilities", "LiabilitiesCurrent", "StockholdersEquity",
      "LiabilitiesAndStockholdersEquity", "RetainedEarningsAccumulatedDeficit", "CashAndCashEquivalentsAtCarryingValue"}
STMT_RE = {"BS": r"balance sheet|financial condition|financial position",
           "IS": r"operations|income|earnings|comprehensive (loss|income)",
           "CF": r"cash flow"}


def stmt_of(concept):
    return "BS" if concept in BS else ("CF" if concept.startswith("NetCash") else "IS")


UA, _last = None, [0.0]


def get(url):
    wait = 0.25 - (time.time() - _last[0])
    if wait > 0:
        time.sleep(wait)
    _last[0] = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", errors="replace")


def statement_pages(cik, accn, cache_dir):
    """{stmt: text} for the primary BS/IS/CF R-pages of a filing (cached on disk)."""
    p = os.path.join(cache_dir, f"{accn}.json")
    if os.path.exists(p):
        return json.load(open(p))
    base = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accn.replace('-', '')}/"
    out = {}
    try:
        fs = get(base + "FilingSummary.xml")
        reports = re.findall(r"<Report[^>]*>(.*?)</Report>", fs, flags=re.S)
        cands = []
        for r in reports:
            sn = re.search(r"<ShortName>(.*?)</ShortName>", r, re.S)
            fn = re.search(r"<HtmlFileName>(.*?)</HtmlFileName>", r, re.S)
            cat = re.search(r"<MenuCategory>(.*?)</MenuCategory>", r, re.S)
            if sn and fn and (cat is None or "Statement" in cat.group(1)):
                cands.append((sn.group(1).strip(), fn.group(1).strip()))
        for st, rx in STMT_RE.items():
            for sn, fn in cands:
                low = sn.lower()
                if re.search(rx, low) and "parenthetical" not in low and not (st == "IS" and "cash" in low):
                    out[st] = text_of(get(base + fn))[:6000]
                    break
    except Exception as ex:  # noqa: BLE001
        out["_error"] = repr(ex)[:200]
    json.dump(out, open(p, "w"))
    return out


def period_text(k):
    if k.start is None:
        return f"as of {k.end}"
    months = "twelve" if k.kind == "annual" else "three"
    return f"for the {months} months ended {k.end} (period starting {k.start})"


SCALE_RULE = {
    "llm": ("report each value in FULL units (apply any 'in thousands'/'in millions' scale stated in the "
            "header; per-share amounts stay as stated)"),
    "code": "report each value exactly as printed in the table, WITHOUT applying any scale",
}


def header_scale(page):
    m = re.search(r"in (thousands|millions|billions)", page[:600], flags=re.I)
    return {"thousands": 1e3, "millions": 1e6, "billions": 1e9}[m.group(1).lower()] if m else 1.0


def ask(model, page, items, scale_mode="llm"):
    lines = "\n".join(f'{i + 1}. {LABEL[k.concept]} {period_text(k)}' for i, k in enumerate(items))
    prompt = (
        "You are extracting numbers from an SEC financial statement table.\n"
        f"Rules: {SCALE_RULE[scale_mode]}; parentheses mean negative; if a value is not in the table "
        "use null. Return ONLY a JSON object mapping item number to number.\n\n"
        f"TABLE:\n{page}\n\nITEMS:\n{lines}\n\nJSON:")
    body = json.dumps({"model": model, "prompt": prompt, "stream": False, "format": "json",
                       "options": {"temperature": 0, "num_ctx": 8192}}).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    try:
        ans = json.loads(d["response"])
    except Exception:  # noqa: BLE001
        ans = {}
    vals = []
    f = header_scale(page) if scale_mode == "code" else 1.0
    for i, k in enumerate(items):
        v = ans.get(str(i + 1))
        try:
            v = None if v is None else float(str(v).replace(",", ""))
        except ValueError:
            v = None
        if v is not None and not k.concept.startswith("EarningsPerShare"):
            v *= f
        vals.append(v)
    return vals, d.get("prompt_eval_count", 0), d.get("eval_count", 0)


class LLMEnv(S.Env):
    """Fact index whose values come from LLM extraction (None -> failed extraction)."""

    def __init__(self, iss, extracted):
        super().__init__(iss)
        self.extracted = extracted

    def retrieve(self, kid, as_of):
        r = super().retrieve(kid, as_of)
        if r is None:
            return None
        v = self.extracted.get((kid, r[1]))
        return None if v is None else (v, r[1])


def replay(iss, policies, env_cls, **kw):
    """S.run_issuer with a custom environment class (primary labels only)."""
    orig = S.Env
    S.Env = (lambda i: env_cls(i, **kw)) if kw else env_cls
    try:
        return S.run_issuer(iss, policies, modes=("primary",))[0]
    finally:
        S.Env = orig


def main():
    global UA
    ap = argparse.ArgumentParser()
    ap.add_argument("--ua", required=True)
    ap.add_argument("--ciks", required=True)
    ap.add_argument("--bench", default="data/bench")
    ap.add_argument("--results", default="results/uniform")
    ap.add_argument("--model", default="qwen2.5:7b-instruct")
    ap.add_argument("--out", default="results/llm_loop")
    ap.add_argument("--scale_mode", choices=("llm", "code"), default="llm")
    ap.add_argument("--pages_dir", default=None, help="reuse a page cache (default: <out>/pages)")
    a = ap.parse_args()
    UA = a.ua
    pages_dir = a.pages_dir or os.path.join(a.out, "pages")
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(pages_dir, exist_ok=True)
    ciks = {int(c) for c in a.ciks.split(",")}
    issuers = [i for f in sorted(glob.glob(os.path.join(a.bench, "shards", "*.pkl")))
               for i in pickle.load(open(f, "rb")) if i.cik in ciks]
    ext_path = os.path.join(a.out, "extractions.json")
    done = {tuple(k.split("##")): v for k, v in json.load(open(ext_path)).items()} if os.path.exists(ext_path) else {}
    log = []
    t0 = time.time()
    for iss in issuers:
        env = S.Env(iss)
        pairs = set()
        for q in iss.queries:
            kid = env.resolve_latest(q[2], q[3]) if q[1] == "latest_annual" else q[2]
            if kid is None:
                continue
            r = env.retrieve(kid, q[3])
            if r:
                pairs.add((kid, r[1]))
        by_page = {}
        for kid, accn in pairs:
            by_page.setdefault((accn, stmt_of(kid.split("|")[0])), []).append(kid)
        print(iss.cik, "pairs", len(pairs), "pages", len(by_page), flush=True)
        for (accn, st), kids in sorted(by_page.items()):
            todo = [k for k in kids if (k, accn) not in done]
            if not todo:
                continue
            pages = statement_pages(iss.cik, accn, pages_dir)
            page = pages.get(st)
            if not page:
                for k in todo:
                    done[(k, accn)] = None
                log.append(dict(cik=iss.cik, accn=accn, stmt=st, n=len(todo), page=False, ptok=0, etok=0))
                continue
            for i in range(0, len(todo), 8):
                chunk = todo[i:i + 8]
                vals, ptok, etok = ask(a.model, page, [iss.keys[k] for k in chunk], a.scale_mode)
                for k, v in zip(chunk, vals):
                    done[(k, accn)] = v
                log.append(dict(cik=iss.cik, accn=accn, stmt=st, n=len(chunk), page=True, ptok=ptok, etok=etok))
            json.dump({"##".join(k): v for k, v in done.items()}, open(ext_path, "w"))
        print(f"  extracted, {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(log).to_csv(os.path.join(a.out, "llm_calls.csv"), index=False)

    # ---- extraction accuracy at the (claim, accession) level
    def on_page(kid, accn):
        pj = os.path.join(pages_dir, f"{accn}.json")
        if not os.path.exists(pj):
            return False
        pg = json.load(open(pj)).get(stmt_of(kid.split("|")[0]), "")
        return f"us-gaap_{kid.split('|')[0]}" in pg

    rows = []
    for iss in issuers:
        for (kid, accn), v in done.items():
            if kid in iss.keys and accn in iss.keys[kid].accns:
                k = iss.keys[kid]
                truthv = k.vals[k.accns.index(accn)]
                ok = v is not None and (same(v, truthv) or abs(v - truthv) <= 1e-6 * max(1, abs(truthv)))
                scale = v is not None and not ok and any(abs(v * f - truthv) <= 1e-6 * max(1, abs(truthv))
                                                        for f in (1e3, 1e6, 1e-3, 1e-6))
                sign = v is not None and not ok and same(-v, truthv)
                rows.append(dict(cik=iss.cik, kid=kid, accn=accn, stmt=stmt_of(k.concept), kind=k.kind,
                                 on_page=on_page(kid, accn),
                                 ledger=truthv, llm=v, correct=ok, missing=v is None, scale_err=scale, sign_err=sign))
    ex = pd.DataFrame(rows)
    ex.to_csv(os.path.join(a.out, "extraction_accuracy.csv"), index=False)
    print(ex.groupby("stmt")[["correct", "missing", "scale_err", "sign_err", "on_page"]].mean().round(3), flush=True)
    print("accuracy when the claim's tag is on the page:", round(ex[ex.on_page].correct.mean(), 4),
          "n", int(ex.on_page.sum()), flush=True)
    print("overall extraction accuracy", round(ex.correct.mean(), 4), "n", len(ex), flush=True)

    # ---- end-to-end replay: ledger env vs LLM env, same controllers
    trained = pickle.load(open(os.path.join(a.results, "b4_models.pkl"), "rb"))
    tuned = json.load(open(os.path.join(a.results, "config.json")))["tuned_heuristic"]

    def pols():
        return [S.TimeFilteredRAG("claim"), S.FinMemGuardRule(gate="claim", name="B3 FinMemGuard-rule (claim release)"),
                S.FinMemGuardRule(gate="claim", inv=tuned[0], max_age=tuned[1], name="B3* tuned heuristic"),
                S.LearnedPolicy(trained[0][1], lam=3.0, name="B4 learned (lam=3)"),
                S.TTLMemory(90), S.NaiveMemory()]
    out = []
    for iss in issuers:
        for envname, cls, kw in (("ledger", S.Env, {}), ("llm", LLMEnv, {"extracted": done})):
            agg = replay(iss, pols(), cls, **kw)
            for k, v in agg.items():
                out.append((envname, *k, *v))
    df = pd.DataFrame(out, columns=["env", "policy", "cik", "split", "qtype", "stratum", "mode", *S.COLS])
    df.to_csv(os.path.join(a.out, "end_to_end_counts.csv"), index=False)
    tab = []
    for (envname, pol), g in df.groupby(["env", "policy"]):
        s = g[list(S.COLS)].sum()
        tab.append(dict(env=envname, policy=pol, queries=int(s.n), coverage=s.answered / s.n,
                        supported_acc=s.supported / max(s.answered, 1), wrong=int(s.answered - s.correct),
                        withdrawn=int(s.withdrawn), calls=int(s.calls)))
    tab = pd.DataFrame(tab)
    calls = pd.read_csv(os.path.join(a.out, "llm_calls.csv"))
    tok = calls[calls.page].eval("ptok + etok").sum() / max(calls[calls.page].n.sum(), 1)
    tab["est_llm_tokens_per_refresh"] = round(tok)
    tab.to_csv(os.path.join(a.out, "end_to_end_table.csv"), index=False)
    print(tab.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
