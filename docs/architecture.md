# Software architecture and code design

```text
Official SEC bulk archives -> immutable raw store -> normalized event ledger
                                                    -> candidate correction labeling
                                                    -> chronological episode builder
Question + as-of -> retrieval (availability filter) -> memory controller
                -> evidence verifier -> answer / abstain + trace
Experiment runner -> metrics -> per-issuer bootstrap -> tables and error analysis
```

## Modules
- `finmemguard/models.py`: immutable event, query, memory state types.
- `finmemguard/controller.py`: deterministic temporal-memory baseline and audit log.
- `finmemguard/evaluation.py`: strict as-of and answer correctness metrics.
- `finmemguard/demo.py`: synthetic end-to-end test (not empirical SEC evidence).
- Future: `ingest_sec.py`, `normalize_xbrl.py`, `episode_builder.py`, `retrieval.py`, `policies.py`, `runner.py`, `report.py`.

## Invariants
Never overwrite source evidence. No event may influence a query before `available_at`. Supersession affects active/latest answers but never erases historical availability. Abstain on unresolved conflicting eligible evidence. Log every retrieval and transition with accession IDs. Avoid calling an update a restatement without adjudication.

## Storage
Prototype: JSONL + SQLite/DuckDB; production: partitioned Parquet + DuckDB for event analytics, optional PostgreSQL for memory transitions. Vector retrieval is optional; structured financial facts should be keyed, not only embedded.
