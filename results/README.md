# Historical summaries

The three summary JSON files are historical records. Their saved counts match the archived predictions under `evidence/evaluation/`, as checked by `python3 scripts/audit_evidence.py`.

They do not establish an independently verified model score, an improvement over the base model, or a match to the recovered checkpoint-250 adapter. D1 and D2 are disjoint archived question sets, not repeated runs on the same questions. New evaluation runs must use a new output directory under `outputs/` and the separate strict scoring protocol.
