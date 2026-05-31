# Curation Eval Report

- Tasks processed: **2**
- Tasks scored against ground truth: **2**
- Verdict accuracy, exact 4-way category (on original answers): **100%**
- Verdict accuracy, binary supported-vs-flagged: **100%**
- Refinements applied: **1**
- Mean confidence when correct: **0.89 (n=2)**
- Mean confidence when incorrect: **n/a (no such cases)**

## Rubric (mean axis scores across all tasks)

- Faithfulness: **0.95**
- Completeness: **0.88**
- Specificity: **0.75**

## Confusion (gold -> predicted)

| gold label | predicted | count |
|---|---|---|
| contradicted | contradicted | 1 |
| supported | supported | 1 |

## Per-task

| task | gold | predicted | match | conf | refined |
|---|---|---|---|---|---|
| pm_001 | contradicted | contradicted | ✓ | 0.92 | yes |
| pm_002 | supported | supported | ✓ | 0.88 | no |
