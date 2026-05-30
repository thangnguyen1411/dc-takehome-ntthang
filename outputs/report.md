# Curation Eval Report

- Tasks processed: **10**
- Tasks scored against ground truth: **10**
- Verdict accuracy, exact 4-way category (on original answers): **20%**
- Verdict accuracy, binary supported-vs-flagged: **50%**
- Refinements applied: **10**
- Mean confidence when correct: **0.97**
- Mean confidence when incorrect: **0.91**

## Rubric (mean axis scores across all tasks)

- Faithfulness: **0.98**
- Completeness: **0.93**
- Specificity: **0.89**

## Confusion (gold -> predicted)

| gold label | predicted | count |
|---|---|---|
| contradicted | contradicted | 2 |
| contradicted | hallucinated  ⚠ | 1 |
| hallucinated | contradicted  ⚠ | 1 |
| supported | hallucinated  ⚠ | 1 |
| supported | unsupported  ⚠ | 2 |
| supported | weak_reasoning  ⚠ | 2 |
| weak_reasoning | contradicted  ⚠ | 1 |

## Per-task

| task | gold | predicted | match | conf | refined |
|---|---|---|---|---|---|
| task_001 | supported | hallucinated | ✗ | 0.93 | yes |
| task_002 | supported | unsupported | ✗ | 0.89 | yes |
| task_003 | contradicted | hallucinated | ✗ | 0.99 | yes |
| task_004 | contradicted | contradicted | ✓ | 0.94 | yes |
| task_005 | supported | weak_reasoning | ✗ | 0.90 | yes |
| task_006 | contradicted | contradicted | ✓ | 0.98 | yes |
| task_007 | supported | weak_reasoning | ✗ | 0.96 | yes |
| task_008 | hallucinated | contradicted | ✗ | 0.96 | yes |
| task_009 | weak_reasoning | contradicted | ✗ | 0.95 | yes |
| task_010 | supported | unsupported | ✗ | 0.91 | yes |
