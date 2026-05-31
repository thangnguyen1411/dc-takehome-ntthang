# Curation Eval Report

- Tasks processed: **2**
- Tasks scored against ground truth: **2**
- Verdict accuracy, exact 4-way category (on original answers): **100%**
- Verdict accuracy, binary supported-vs-flagged: **100%**
- Refinements applied: **1**
- Mean confidence when correct: **0.95**
- Mean confidence when incorrect: **0.00**

## Rubric (mean axis scores across all tasks)

- Faithfulness: **1.00**
- Completeness: **0.97**
- Specificity: **0.94**

## Confusion (gold -> predicted)

| gold label | predicted | count |
|---|---|---|
| contradicted | contradicted | 1 |
| supported | supported | 1 |

## Per-task

| task | gold | predicted | match | conf | refined |
|---|---|---|---|---|---|
| ret_001 | contradicted | contradicted | ✓ | 0.98 | yes |
| ret_002 | supported | supported | ✓ | 0.90 | no |
