# FinMemBench-pilot and FinMemGuard

Code, label audits, and result tables for the paper
**"FinMemGuard: Trustworthy Lifecycle Control of Financial Agent Memory over Evolving Disclosures"** (Luxin Zhang, 2026).

FinMemBench-pilot measures whether an agent's persistent memory stays valid as SEC disclosures are revised, withdrawn by 8-K Item 4.02 non-reliance notices, and restated. It is built from the complete SEC `companyfacts` and `submissions` bulk archives: 4,790 issuers, 1.91M versioned claims, and 2,584 Item 4.02 notices whose texts are parsed for the periods they name, replayed as 1.4–2.5M as-of-stamped queries with company-disjoint splits.
FinMemGuard is the companion controller: an immutable evidence ledger plus active / suspect / quarantined memory states, claim-specific quarantine release, and a learned one-step (contextual-bandit) variant that tunes the re-verification rate inside the hard gate.

All main-experiment accuracies are *ledger-consistent* (agreement with SEC-reported values); the main experiments contain no LLM. See the paper for scope and limitations.

## Data policy
This repository ships **no SEC records**. Download the bulk archives yourself under the SEC fair-access policy (declare a contact User-Agent) and verify them against `data_manifest/SHA256SUMS`; see `data_manifest/README.md`. SEC refreshes the archives nightly, so the hashes identify the exact snapshot used in the paper.

## Quick start
```bash
python -m pip install -r requirements.txt
python -m pytest -q          # 24 synthetic tests, no SEC data needed
python -m finmemguard.demo   # smoke test
# full pipeline (needs data/raw/{companyfacts,submissions}.zip):
SEC_UA="FinMemBench research you@example.org" ./run_all.sh
```
`run_all.sh` runs ingestion → benchmark construction (Uniform and Recency workloads) → training and evaluation with issuer-clustered bootstrap CIs → report tables. With `SEC_UA` set it also downloads the Item 4.02 notice texts, runs the document-audit sampler, and the frozen-LLM refresh check (requires a local ollama server with `qwen2.5:7b-instruct`). Set `OMP_NUM_THREADS=1` on memory-constrained machines.

## Reproduction levels
| Level | Command | Needs | Reproduces |
|---|---|---|---|
| 1. Tables from released results | `python -m finmemguard.report --skip_audit --paper out` | this repo only | Tables II, III, V and Fig. 2 of the paper, byte-identical (`out/tables/*.tex`, `out/figs/frontier.pdf`) |
| 2. Unit tests | `python -m pytest -q` | this repo only | 24 synthetic tests |
| 3. Full benchmark and results | `SEC_UA=... ./run_all.sh` | SEC bulk archives matching `data_manifest/SHA256SUMS` (about 45 min with 6 workers) | `results/uniform`, `results/recency`, `results/sens` |
| 4. Notice texts, audit sampler, LLM check | included in `run_all.sh` when `SEC_UA` is set | EDGAR access; local ollama with `qwen2.5:7b-instruct` | `results/llm_loop*`; audit judgements themselves are manual |

Dependencies are pinned in `requirements.txt`.

## Layout
| Path | Contents |
|---|---|
| `finmemguard/ingest_sec.py`, `ingest_events.py` | bulk archives → Parquet; filing events and Item 4.02 notice texts |
| `finmemguard/notice_parse.py` | parses which reporting periods an Item 4.02 notice names |
| `finmemguard/benchmark.py` | ledger, non-reliance labels and the three label modes, query workloads, splits |
| `finmemguard/simulate.py`, `controller.py` | systems B0–B4, claim-specific quarantine, replay |
| `finmemguard/runner.py`, `evaluation.py`, `report.py` | training (3 seeds), validation grid for the tuned heuristic, evaluation, bootstrap, tables and figure |
| `finmemguard/audit_nr.py` | samples non-reliance episodes for document-level audit against EDGAR |
| `finmemguard/llm_loop.py`, `llm_report.py` | frozen-LLM refresh check |
| `results/uniform`, `results/recency` | main tables, per-stratum tables, bootstrap CIs, validation grids |
| `results/sens` | restatement-horizon and review-window sensitivity |
| `results/nr_audit` | adjudicated audit sheets (rounds 1–4) with EDGAR document references |
| `results/llm_loop*` | LLM extractions, accuracy, and end-to-end tables |
| `docs/` | benchmark, architecture, and metric notes |

## Audit annotation
All audit judgements in `results/nr_audit/` were made by a single LLM-based annotator (Anthropic's Claude) and were not double-annotated; see the paper's acknowledgment and limitations.

## Citation
```bibtex
@misc{zhang2026finmemguard,
  title  = {FinMemGuard: Trustworthy Lifecycle Control of Financial Agent Memory over Evolving Disclosures},
  author = {Zhang, Luxin},
  year   = {2026},
  note   = {Code: https://github.com/luxinlabs/FinMemBench}
}
```

## License
Code: MIT (see `LICENSE`). Derived result tables and audit sheets: CC BY 4.0. SEC source data are public records obtained from sec.gov and are not redistributed here. Nothing in this repository is investment advice.
