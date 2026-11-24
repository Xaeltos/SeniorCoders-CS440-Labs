# Senior Coders — NLBSE 2027 Technical Results and Methodology Notes

> Technical experiment record for project reproducibility and team handoff. Not a research-paper draft.

## Data and evaluation

- Source: `skillscope_data.db`, issue-level table `nlbse_tool_competition_data_by_issue`.
- Labeled competition table: **7,154** issue records from **12** repositories; **217** skill-label columns.
- Source database also contains **7,245** raw rows in `pull_requests`; **91** raw PR identifiers are absent from the labeled issue-level table. Of those 91, **20** have empty descriptions. The reason for exclusion is not established; they were **not** added to training.
- Fixed split (`random_state=42`): **5,007** training, **716** validation, **1,431** held-out test records.
- Inputs to the final model: issue title and description only. Targets: positive skill-label counts converted to binary (`>0`).
- No completed PR patches, extracted API-name fields, or post-solution content entered final-model training.
- Final classification threshold **0.54**, chosen on validation only; not readjusted on test.

## Model and experiments

- Base encoder: `FacebookAI/roberta-base` with **217** multi-label classification outputs.
- Original training experiments: binary cross-entropy (BCE), focal loss, and weighted focal loss. A subsequent recovery experiment used weighted BCE, distinct learning rates, and additional epochs.
- Final saved checkpoint: `runs/roberta_recovery/best_model`.
- Inference CLI: `nlbse_predict.py`, producing a 217-position binary vector; single-issue and two-row JSONL batch inference verified.
- Python environment recorded in `requirements-lock.txt` and `requirements.txt`.

## Held-out test: same-split comparison

Both methods were evaluated on the same **1,431** held-out test issues. The Random Forest was trained using train-only TF-IDF features and **did not use the original baseline's pre-split oversampling**. This is a **leakage-controlled reproduction of the SkillScope Random Forest + TF-IDF method**, **not** the published SkillScope score or the official competition evaluation.

| Metric | SkillScope-method RF | RoBERTa (threshold 0.54) | RoBERTa minus RF |
|---|---:|---:|---:|
| Micro-precision | 67.7279% | 69.3560% | +1.6282 percentage points |
| Micro-recall | 48.2726% | 50.3163% | +2.0436 percentage points |
| Micro-F1 | 56.3688% | 58.3215% | +1.9528 percentage points |

Reference file: `runs/skillscope_same_split/locked_test.json`.

## Validation threshold selection

At threshold **0.54**, RoBERTa met or exceeded all three scores of the leakage-controlled SkillScope-method RF on the **716** validation issues:

| Metric | SkillScope-method RF | RoBERTa (threshold 0.54) |
|---|---:|---:|
| Micro-precision | 69.5014% | 71.0844% |
| Micro-recall | 47.0177% | 50.3011% |
| Micro-F1 | 56.0903% | 58.9135% |

Reference files: `runs/skillscope_same_split/comparison.json` and `runs/skillscope_same_split/clean_sweep_thresholds.json`. This threshold was selected using validation results and then frozen before the held-out test.

## Important limitations and diagnostic findings

- **Common-label dominance:** Shuffling validation issue texts had almost no effect on RoBERTa's predictions in multiple experiments. The final model's text sensitivity remains weak; beating the reproduced RF does **not** establish strong issue-specific skill recognition.
- **Frequency-only reference:** A predictor using training-set label prevalence alone reached approximately **63.68% validation micro-F1** with a tuned cutoff in an earlier test, exceeding the tuned RoBERTa validation micro-F1. This is a key limitation that should be reported alongside the same-split RF comparison; the scoring thresholds and prediction strategies differ.
- **Rare labels:** **86 of 217** skills had at most **20** positive training examples. The evaluated models failed to predict positives for the rare-label subset at their selected thresholds.
- **Hierarchy:** **186** direct parent–child relationships inferred from hyphen-prefixed labels; no parent-missing instances in the audited ground truth or model predictions. Hierarchy-closing postprocessing had no effect on F1.
- **Additional training:** Extending recovery training to four epochs, pairwise fine-tuning, and linear probing did not substantially improve validation F1 or text sensitivity.
- **Baseline caveat:** The original SkillScope procedure fitted TF-IDF and augmented/duplicated records before splitting, which can risk evaluation leakage. The comparison here deliberately moves splitting ahead of feature fitting and excludes pre-split MLSMOTE. It is not an exact reproduction of the published procedure.
- **Eligibility caveat:** These local results are **not evidence of official NLBSE qualification**. Official submission acceptance and evaluation criteria need separate confirmation.
- **Reproducibility caveat:** GitHub branch `final-nlbse-2027`, commit `5a10972`, was cloned successfully; Git LFS database and final model downloaded, and inference succeeded using the original project virtual environment. A fully clean environment installation and independent reproduction of training metrics have **not** been tested.

## Verification and code entry points

- `skill_classifier.py`: original BCE/focal training.
- `skill_classifier_weighted.py`: weighted focal training.
- `roberta_recovery.py`: final model's training configuration.
- `compare_skillscope_roberta.py`: same-split validation comparison.
- `find_clean_sweep.py`: validation threshold selection.
- `verify_clean_sweep_test.py`: held-out test comparison.
- `nlbse_predict.py`: single and JSONL batch inference (217-position binary vectors).

GitHub project branch: https://github.com/Xaeltos/SeniorCoders-CS440-Labs/tree/final-nlbse-2027
