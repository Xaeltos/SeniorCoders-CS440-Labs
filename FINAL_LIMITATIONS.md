# Senior Coders — Final Limitations (Corrected v2 Ensemble)

**Companion to:** `FINAL_EXPERIMENT_AND_RESULTS.md`  
**Final model:** Three-model focal-loss RoBERTa ensemble, threshold **0.535**  
**Scope:** Technical notes for the limitations section and video; not a manuscript draft.

## Summary of the measured result

On the team's corrected-v2 **614-issue follow-up test** partition, the ensemble improved all three micro-averaged metrics against the **leakage-controlled reproduced** SkillScope-method Random Forest:

| Metric | Reproduced RF | Ensemble | Difference |
|---|---:|---:|---:|
| Precision | 72.90% | **77.67%** | +4.77 pp |
| Recall | 33.98% | **35.29%** | +1.31 pp |
| Micro-F1 | 46.35% | **48.53%** | +2.18 pp |

This is the correct final comparison. The older 46.64% F1 and slight recall regression describe the **single focal model**, not the final ensemble.

## Limitation 1 — Baseline comparability

The benchmark in the team's experiments is a **reproduced** TF-IDF (1,000 features) + Random Forest (100 trees) classifier, with TF-IDF fitted on training records only and no pre-split MLSMOTE. It is not identical to the historical SkillScope baseline procedure and is not the competition organizers' independently verified reference. Therefore the observed gains refer to **this local comparison**, not an official competition scoring result.

**Evidence:** `v2_ensemble_rescue.py` (`baseline_validation`) and `v2_ensemble_followup.py`.

## Limitation 2 — Test partition reuse

The 614-issue evaluation partition was examined for earlier v2 model experiments before the ensemble follow-up. Although the ensemble's **0.535** threshold was selected on validation, this partition is **not a fresh independent holdout** for the overall development process. The final results should be called a **follow-up test comparison**.

**Evidence:** `runs/v2_ensemble_rescue/fixed_0535_followup.json`.

## Limitation 3 — Team-defined data split

The experiment used **2,149 training**, **307 validation**, and **614 test** issues, with a deterministic random split (seed 42). These assignments have not been verified against any official organizer-provided split. Scores on a different evaluation partition could differ.

**Evidence:** `skill_classifier.py` (`split_indices`), `v2_ensemble_rescue.py`.

## Limitation 4 — Rare skills and uneven label frequency

The task contains **217 possible skill labels** with an uneven frequency distribution. Micro-averaged scores emphasize labels occurring often, so improvements in overall micro-F1 do not establish reliable predictions for every rare label. Additional label-level or rare-label analysis would be needed for stronger conclusions.

**Important accuracy note:** The statement **“86 labels had at most 20 training positives”** came from earlier experiments and has **not** been reverified on corrected v2. Do not label that number as a v2 finding.

## Limitation 5 — Reproducibility boundaries

The final three checkpoints and the 217-label prediction interface were successfully exercised in the established Windows/Python project environment. Git LFS holds the model weights. A wholly fresh-environment dependency installation and independent full-training reproduction were not yet verified.

**Evidence:** `nlbse_predict_ensemble_v2.py`, `requirements.txt`, `requirements-lock.txt`, and the three `runs/.../best_model/` directories.

## What the text-sensitivity diagnostic does and does not show

An **individual focal model** on corrected-v2 validation achieved **58.58% micro-F1** with original issue text and **41.11%** with shuffled text; the frequency-only reference achieved **46.77%**. This supports meaningful input-text dependence in that model, but does **not** show that the ensemble solves rare-label classification or that the same diagnostic was run on the final ensemble.

## Research-ready future directions

- Independently evaluate on an untouched set of GitHub issues.
- Report per-label and rare-label precision/recall in addition to micro-averages.
- Evaluate training/data-augmentation methods for infrequent skills without leaking labels or test information.
- Recheck the final model under any officially specified competition split and benchmark.

## Files to show in a limitations video

- **Results:** `runs/v2_ensemble_rescue/fixed_0535_followup.json`
- **Validation:** `runs/v2_ensemble_rescue/validation_report.json`
- **Full context:** `FINAL_EXPERIMENT_AND_RESULTS.md`

**Short distinction to preserve:** The ensemble passed all three metrics **against the reproduced RF in the local follow-up comparison**. That is a concrete measured finding, **not** an organizer-certified eligibility outcome.
