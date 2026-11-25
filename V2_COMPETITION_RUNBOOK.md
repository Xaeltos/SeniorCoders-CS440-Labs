# NLBSE 2027 Skill Classification — corrected-v2 execution guide

## Status and scope

- Git branch: `final-nlbse-2027` (keep independent from all other branches).
- Dataset: `dataset_v2/skillscope_data_v2.db`; 3,070 labeled issues, 217 binary label columns.
- Final inference candidate: `runs/v2_roberta_focal/best_model` with global threshold `0.545`, chosen solely using v2 validation for the three-metric constraint against a leakage-controlled reproduced SkillScope Random Forest.
- Retain original `skillscope_data.db`, `runs/roberta_recovery/best_model`, and v1 reports unchanged.
- This is not certification of the organizers' evaluation. As of 2026-11-25, the organizers' upstream `competition_baseline/TODO.md` still calls the official split, baseline policy, and reference scores outstanding decisions.

## Install and quick check (Windows PowerShell, project root)

Existing project environment:

```powershell
.\.venv\Scripts\python.exe nlbse_predict_v2.py --self-test
```

One input example, prints a JSON object whose `vector` contains 217 binary entries:

```powershell
.\.venv\Scripts\python.exe nlbse_predict_v2.py --title "Fix connection timeout" --description "Implement retry logic" --check-label-order dataset_v2/skillscope_data_v2.db
```

Batch input: create a UTF-8 JSON Lines file with one object per line, containing `title` and `description` (alternatively `issue text` and `issue description`). Run:

```powershell
.\.venv\Scripts\python.exe nlbse_predict_v2.py --input issues.jsonl --output predictions_v2.jsonl --check-label-order dataset_v2/skillscope_data_v2.db
```

Do not reuse the old `nlbse_predict.py` for the corrected submission: it defaults to the v1 checkpoint and threshold.

## Result reporting

**Local validation, reproduced RF vs focal RoBERTa at 0.545:** precision 0.72692 vs 0.75923; recall 0.30855 vs 0.33705; micro-F1 0.43322 vs 0.46685. All three improve on validation.

**Follow-up test, reproduced RF vs focal RoBERTa at 0.545:** precision 0.72899 vs 0.76472; recall 0.33981 vs 0.33547; micro-F1 0.46354 vs 0.46635. Recall is 0.00434 lower; there is **no test clean sweep**. The same test split had been inspected in prior experiments; this is not a newly untouched final evaluation.

The reproduced RF is a train-only, leakage-controlled approximation, **not** the historical SkillScope pipeline or a confirmed organizer reference result. Do not claim official eligibility from these comparisons.

## Competition handoff

- Share `V2_EXPERIMENT_RESULTS.md` with the paper author and retain v1 results as superseded history.
- Paper must name corrected dataset, data split, focal loss, validation-selected threshold, reproducibility instructions, and honest metric tradeoffs. Do not call local RF results official scores.
- Check the current organizers' instructions for any published split/scoring clarification before final submission. If still unspecified, ask the organizers for exact row assignments, v2 reference baseline scores, and evaluation interface.
- Do not use any API-derived skill labels as inputs. Predictions use issue title/description only.
- Git LFS is required to clone the database and model. Run `git lfs pull` after cloning if assets are missing.
