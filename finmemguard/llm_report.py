"""Post-hoc analysis of the frozen-LLM check: separates harness coverage (is the ledger
value printed on the statement page we gave the model?) from LLM extraction errors.

  python -m finmemguard.llm_report results/llm_loop results/llm_loop_code
"""
import json
import os
import sys

import pandas as pd

from .llm_loop import stmt_of


def printed_forms(v):
    out = set()
    for f in (1, 1e3, 1e6):
        x = abs(v) / f
        if abs(x - round(x)) < 1e-9 and x >= 1:
            out.add(f"{int(round(x)):,}")
    if abs(v) < 100:                       # per-share style amounts
        out.update({f"{abs(v):.2f}", f"{abs(v):.3f}".rstrip("0")})
    return out


def value_on_page(row, pages_dir):
    pj = os.path.join(pages_dir, f"{row.accn}.json")
    if not os.path.exists(pj):
        return False
    pg = json.load(open(pj)).get(stmt_of(row.kid.split("|")[0]), "")
    return any(s in pg for s in printed_forms(row.ledger))


def main():
    pages = os.path.join(sys.argv[1], "pages")
    for d in sys.argv[1:]:
        ex = pd.read_csv(os.path.join(d, "extraction_accuracy.csv"))
        ex["on_page"] = [value_on_page(r, pages) for r in ex.itertuples()]
        ex.to_csv(os.path.join(d, "extraction_accuracy.csv"), index=False)
        op = ex[ex.on_page]
        print(f"== {d}: n={len(ex)} value-on-page={len(op)} ({len(op) / len(ex):.1%})")
        print(f"   all: correct {ex.correct.mean():.3f}")
        print(f"   on-page: correct {op.correct.mean():.3f} scale_err {op.scale_err.mean():.3f} "
              f"missing {op.missing.mean():.3f} other_wrong {(~op.correct & ~op.scale_err & ~op.missing).mean():.3f}")
        print(op.groupby("stmt")[["correct", "scale_err", "missing"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
