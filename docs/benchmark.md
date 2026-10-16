# FinMemBench specification

## Evaluation unit
A chronological **episode** contains an issuer, ordered disclosure events, queries at as-of cutoffs, reference answers/abstention labels, and evidence accession IDs. Measure both per-question and per-episode outcomes.

## Episode families
A. Valid original claim, then corrected amendment.
B. Original claim then non-reliance notice and later restatement: abstention during gap.
C. New fiscal period: previous fact remains valid historically but not as latest.
D. Two competing sources: uncertain or incompatible contexts; require verification.
E. Missing disclosure: appropriate abstention.

## JSONL fields
`episode_id, cik, events:[{accession, available_at, claim_key, value, unit, status}], queries:[{as_of, question, expected_value, expected_action, evidence_accessions}]`.

## Labeling
Only high-confidence matched facts form automatically scored answers. Require independent sample audits for amendment/non-reliance labels. Preserve uncertain cases with `review_required`, not fabricated ground truth. Track inter-annotator agreement if humans label cases.

## Scale gates
Gate 1: >= 1000 verified source events; Gate 2: >= 100 verified transitions; Gate 3: >= 1000 verified questions; long-term 5000+ questions. Report observed counts and confidence intervals, never aspirational numbers as achieved.
