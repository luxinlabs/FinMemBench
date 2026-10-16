#!/usr/bin/env bash
# Full FinMemBench-pilot pipeline. Requires data/raw/{companyfacts,submissions}.zip (see data_manifest/README.md).
# Set SEC_UA to a User-Agent with your contact e-mail (SEC fair-access policy), e.g.
#   SEC_UA="FinMemBench research you@example.org" ./run_all.sh
set -euo pipefail
export OMP_NUM_THREADS=1
python -m finmemguard.ingest_sec
python -m finmemguard.benchmark --workload uniform --per_month 3 --out data/bench   # first pass: sample issuers
python -m finmemguard.ingest_events events
[ -n "${SEC_UA:-}" ] && python -m finmemguard.ingest_events notices --ua "$SEC_UA"   # Item 4.02 texts
python -m finmemguard.benchmark --workload uniform --per_month 3 --out data/bench
python -m finmemguard.benchmark --workload recency --per_month 6 --out data/bench_recency
rm -rf data/bench/shards data/bench_recency/shards
python -m finmemguard.runner --bench data/bench --out results/uniform --boot 1000 --workers 6
python -m finmemguard.runner --bench data/bench_recency --out results/recency --boot 1000 --workers 6
python -m finmemguard.report
if [ -n "${SEC_UA:-}" ]; then
  # independent audit: downloads 8-K Item 4.02 notices and restating filings (human adjudication follows)
  python -m finmemguard.audit_nr --ua "$SEC_UA"
  # frozen-LLM refresh check (requires a local ollama server with qwen2.5:7b-instruct)
  python -m finmemguard.llm_loop --ua "$SEC_UA" --ciks 19612,1818212,1658432 --scale_mode llm
  python -m finmemguard.llm_loop --ua "$SEC_UA" --ciks 19612,1818212,1658432,813640,1823854,1047170,92679,1732406,1922331 --scale_mode code --out results/llm_loop_code --pages_dir results/llm_loop/pages
  python -m finmemguard.llm_report results/llm_loop results/llm_loop_code
fi
