# Curation Eval Report

- Tasks processed: **10**
- Tasks scored against ground truth: **10**
- Verdict accuracy, exact 4-way category (on original answers): **30%**
- Verdict accuracy, binary supported-vs-flagged: **60%**
- Refinements applied: **9**
- Mean confidence when correct: **0.95**
- Mean confidence when incorrect: **0.92**

## Rubric (mean axis scores across all tasks)

- Faithfulness: **0.93**
- Completeness: **0.87**
- Specificity: **0.82**

## Confusion (gold -> predicted)

| gold label | predicted | count |
|---|---|---|
| contradicted | contradicted | 2 |
| contradicted | hallucinated  ⚠ | 1 |
| hallucinated | contradicted  ⚠ | 1 |
| supported | supported | 1 |
| supported | unsupported  ⚠ | 2 |
| supported | weak_reasoning  ⚠ | 2 |
| weak_reasoning | contradicted  ⚠ | 1 |

## Per-task

| task | gold | predicted | match | conf | refined |
|---|---|---|---|---|---|
| task_001 | supported | weak_reasoning | ✗ | 0.86 | yes |
| task_002 | supported | unsupported | ✗ | 0.86 | yes |
| task_003 | contradicted | hallucinated | ✗ | 0.84 | yes |
| task_004 | contradicted | contradicted | ✓ | 0.98 | yes |
| task_005 | supported | weak_reasoning | ✗ | 0.90 | yes |
| task_006 | contradicted | contradicted | ✓ | 0.97 | yes |
| task_007 | supported | supported | ✓ | 0.89 | no |
| task_008 | hallucinated | contradicted | ✗ | 0.96 | yes |
| task_009 | weak_reasoning | contradicted | ✗ | 0.99 | yes |
| task_010 | supported | unsupported | ✗ | 0.90 | yes |
