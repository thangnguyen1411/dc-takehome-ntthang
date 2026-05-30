# Curation Eval Report

- Tasks processed: **10**
- Tasks scored against ground truth: **10**
- Verdict accuracy, exact 4-way category (on original answers): **50%**
- Verdict accuracy, binary supported-vs-flagged: **70%**
- Refinements applied: **8**
- Mean confidence when correct: **0.94**
- Mean confidence when incorrect: **0.84**

## Confusion (gold -> predicted)

| gold label | predicted | count |
|---|---|---|
| contradicted | contradicted | 3 |
| hallucinated | contradicted  ⚠ | 1 |
| supported | supported | 2 |
| supported | unsupported  ⚠ | 1 |
| supported | weak_reasoning  ⚠ | 2 |
| weak_reasoning | contradicted  ⚠ | 1 |

## Per-task

| task | gold | predicted | match | conf | refined |
|---|---|---|---|---|---|
| task_001 | supported | supported | ✓ | 0.90 | no |
| task_002 | supported | weak_reasoning | ✗ | 0.83 | yes |
| task_003 | contradicted | contradicted | ✓ | 0.86 | yes |
| task_004 | contradicted | contradicted | ✓ | 0.97 | yes |
| task_005 | supported | weak_reasoning | ✗ | 0.90 | yes |
| task_006 | contradicted | contradicted | ✓ | 0.97 | yes |
| task_007 | supported | supported | ✓ | 0.90 | no |
| task_008 | hallucinated | contradicted | ✗ | 0.97 | yes |
| task_009 | weak_reasoning | contradicted | ✗ | 0.97 | yes |
| task_010 | supported | unsupported | ✗ | 0.90 | yes |
