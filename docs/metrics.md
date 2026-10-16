# Evaluation and metrics

- **Stale-answer rate:** stale/superseded answers divided by answered current-state queries where controlling evidence is known.
- **Future-evidence leakage:** answers supported by any accession available after query cutoff / answered queries.
- **Supported accuracy:** correct answers with valid evidence / answered queries.
- **Coverage:** answered / total queries.
- **Selective risk:** incorrect or unsupported / answered at a specified coverage.
- **Transition precision/recall:** correctness of retire/retain/conflict decisions against adjudicated labels.
- **Cost:** model tokens, SEC/retrieval calls, latency, estimated USD per query and per supported correct answer.
- **Historical fidelity:** accuracy of queries about previous knowledge states after later corrections.

## Reporting
Always report denominators, dataset size, uncertainty intervals, and class prevalence. Break out original vs amended vs non-reliance cases; current-state vs historical queries. An abstention is neither an answered success nor an answered hallucination. Report both accuracy and coverage to prevent gaming.
