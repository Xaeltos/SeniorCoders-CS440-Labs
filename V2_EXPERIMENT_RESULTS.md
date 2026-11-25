# Corrected SkillScope Dataset v2 — Experimental Results

## Scope and dataset correction

These results were generated using the corrected `dataset_v2/skillscope_data_v2.db`, containing **3,070 labeled issue records**, **217 skill labels**, and **11 repositories**. The earlier experiments used an obsolete dataset with 7,154 labeled issue records. The original database, checkpoints, and experimental results are retained for historical reproducibility but **must not be reported as results on v2**.

## Experimental partition and interpretation

The v2 experiments shown here used a deterministic local **random** train/validation/test partition of **2,149 / 307 / 614 records**, respectively (as reported by `run_v2_rebuild.py`). This is **not demonstrated to be identical to the competition's provided splits**. The comparator is our **leakage-controlled reproduction of a SkillScope-style TF-IDF + Random Forest approach**, *not* the official organizers' reported or evaluation-server baseline. Accordingly, these results **do not establish competition eligibility or ranking**.

The held-out test subset was first evaluated for the weighted-BCE model, and subsequently reused in a follow-up evaluation for focal RoBERTa. We disclose that reuse; do not describe the focal evaluation as a wholly untouched, independent final test.

## Models

- **Weighted-BCE RoBERTa (v2):** `runs/v2_roberta_recovery/best_model`; validation-selected global threshold `0.320`.
- **Focal-loss RoBERTa (v2):** `runs/v2_roberta_focal/best_model`; gamma `2.0`, no positive weighting, 4 epochs. Its precision-oriented global threshold `0.545` was selected from the v2 validation partition. The validation-best F1 operating point was approximately `0.390`; the test figures below are **only** for the frozen `0.545` threshold.

## Validation comparison (focal RoBERTa, global threshold 0.545)

| Model | Precision | Recall | Micro-F1 |
|---|---:|---:|---:|
| Reproduced SkillScope-style RF | 0.726917 | 0.308551 | 0.433217 |
| Focal-loss RoBERTa | **0.759230** | **0.337055** | **0.466853** |
| Difference, RoBERTa − RF | +0.032313 | +0.028504 | +0.033636 |

Focal-loss RoBERTa attained a higher validation micro-F1 of **0.588346** at a different threshold (`0.390`), but did **not** outperform the reproduced RF baseline on precision at that operating point. The `0.545` setting was selected to improve all three validation metrics simultaneously.

At its epoch-selected validation operating point, focal RoBERTa yielded micro-F1 **0.585784** with original issue text, **0.411069** with shuffled issue text, and **0.467748** for a frequency-only baseline. This sensitivity check is evidence of input-text dependence, not conclusive evidence of generalization.

## Follow-up test comparison (focal RoBERTa, frozen threshold 0.545)

| Model | Precision | Recall | Micro-F1 |
|---|---:|---:|---:|
| Reproduced SkillScope-style RF | 0.728987 | **0.339809** | 0.463543 |
| Focal-loss RoBERTa | **0.764720** | 0.335468 | **0.466355** |
| Difference, RoBERTa − RF | +0.035733 | −0.004341 | +0.002812 |

**Interpretation:** Focal RoBERTa improves precision and micro-F1 but slightly reduces recall relative to our reproduced baseline on the reused v2 test partition. **It does not demonstrate a three-metric clean sweep on this test partition.**

For context, the first weighted-BCE v2 model at threshold `0.320` attained test precision **0.456699**, recall **0.481333**, and micro-F1 **0.468692**. That model prioritizes recall at a substantial precision cost.

## Reproducibility

- Training and validation scripts: `run_v2_rebuild.py`, `roberta_v2_focal.py`, `v2_precision_rescue.py`, `evaluate_v2_focal_fixed.py`.
- Corrected data: `dataset_v2/skillscope_data_v2.db` (Git LFS).
- Focal checkpoint: `runs/v2_roberta_focal/best_model/` (weights in Git LFS).
- Detailed results: `runs/v2_roberta_focal/recovery_metrics.json`, `runs/v2_roberta_focal/v2_precision_audit.json`, `runs/v2_roberta_focal/v2_fixed_threshold_followup.json`.
- Earlier v2 weighted-BCE metrics: `runs/v2_roberta_recovery/recovery_metrics.json`, `runs/v2_roberta_recovery/v2_locked_comparison.json`.

## Competition submission caveat

The competition README states that submissions are evaluated on **provided splits**, relative to its **best baseline**, and must not degrade precision, recall, or micro-F1. Before asserting eligibility or reporting competition-comparable metrics, verify the official splits, benchmark implementation, and approved submission mechanism. Do not substitute the local RF reproduction for the official baseline without saying so explicitly.
