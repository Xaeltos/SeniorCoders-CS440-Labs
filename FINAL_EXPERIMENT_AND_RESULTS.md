# Senior Coders — Final Experiment and Results

**NLBSE 2027 Skill Classification · Corrected SkillScope v2 · Technical experiment record**

> **Use this file for the final-model statistics and baseline explanation.** The earlier `EXPERIMENT_RESULTS.md` documents the superseded v1 dataset, while `V2_EXPERIMENT_RESULTS.md` documents intermediate single-model v2 results. Neither contains the final ensemble comparison. This is a technical reference for the team, not a manuscript draft.

## 1. Experiment overview

**Objective:** Predict which of 217 domain/subdomain software-development skills apply to a GitHub issue, using the issue's title and description. This is multi-label classification: multiple skills can be relevant to a single issue.

**Final system:** Equal-weight ensemble of three independently fine-tuned `FacebookAI/roberta-base` classifiers. Each outputs 217 logits. For each model, logits are converted to probabilities with sigmoid; the three probability vectors are averaged by label; a probability of **at least 0.535** produces a positive prediction.

**Chosen model:** The ensemble, not the historical v1 RoBERTa checkpoint, the weighted-BCE v2 recovery model, or the single focal-loss v2 checkpoint by itself.

## 2. Dataset and experimental split

| Property | Corrected v2 experiment |
|---|---|
| Dataset | `dataset_v2/skillscope_data_v2.db` |
| Issue-level table | `nlbse_tool_competition_data_by_issue` |
| Labeled issues | **3,070** |
| Source repositories | **11** |
| Skill-label columns | **217** |
| Positive-label rule | Source label count greater than zero becomes 1 |
| Training issues | **2,149 (70%)** |
| Validation issues | **307 (10%)** |
| Follow-up test issues | **614 (20%)** |
| Split | Deterministic local random partition; seed **42**, as implemented by `split_indices` in `skill_classifier.py` |
| Model input | GitHub issue title and issue description |

The earlier `skillscope_data.db` experiment had **7,154 labeled issues** and a different distribution. Its numerical results must not be mixed with corrected-v2 results. Both databases and the earlier experiments remain in the repository for historical reproducibility.

**Scope of the comparison:** Both final approaches were evaluated on the same local v2 evaluation records. The local split has not been established as identical to any organizer-provided split.

## 3. Reproduced SkillScope-method baseline

The comparison baseline is a **leakage-controlled reproduction** of the SkillScope text-classification approach. It is **not** the competition organizers' official reference score and is not a byte-for-byte run of the historically extracted MLSMOTE pipeline.

| Component | Reproduced baseline |
|---|---|
| Algorithm | Random Forest multi-label classifier |
| Text features | TF-IDF of cleaned issue title and description |
| Maximum TF-IDF features | **1,000** |
| Random Forest | `RandomForestClassifier(random_state=42)`; default tree count **100** |
| TF-IDF fitting | **Training records only** |
| Evaluation records | Transformed with the training-fitted vectorizer; not used to fit features or classifier |
| Pre-split MLSMOTE / duplicated rows | **Not used** |
| Scoring | Micro-averaged precision, recall, F1 |
| Source | `v2_ensemble_rescue.py` (`baseline_validation`), reused by `v2_ensemble_followup.py`; cleaning/metrics from `compare_skillscope_roberta.py` |

**Why it differs from the historical baseline:** The extracted historical SkillScope path fits TF-IDF and performs MLSMOTE before selecting a holdout. This experiment instead separates original records before fitting features and avoids pre-split synthetic oversampling, preventing that specific source of train/evaluation contamination. These procedural differences limit direct comparison with published SkillScope scores.

## 4. Model training and selection

| Setting | Final experiment |
|---|---|
| Pretrained backbone | `FacebookAI/roberta-base` |
| Number of independently trained models | **3** |
| Random seeds | **42, 2028, 2029** |
| Output dimensionality | **217** logits/model |
| Loss | Symmetric sigmoid focal loss |
| Focal gamma | **2.0** |
| Additional positive-label weighting | **None** in final focal models |
| Epochs | **4** for each model |
| Maximum input sequence length | **256** tokens |
| Batch size | **2** |
| Gradient accumulation | **8** |
| Encoder learning rate | **2e-5** |
| Classification-head learning rate | **1e-4** |
| Optimizer | AdamW, weight decay **0.01** |
| Model checkpoint selection | Validation performance |
| Ensemble combination | Equal-weight mean of sigmoid probabilities across the three models |
| Final global threshold | **0.535**, selected on validation |

Focal loss gives less relative weight to easy, confidently predicted label outcomes. It does **not** mean that labels were automatically weighted according to their training frequency; there was no separate positive-class weighting in the final focal runs.

**Model checkpoints:**

1. `runs/v2_roberta_focal/best_model/` — original corrected-v2 focal checkpoint (seed 42).
2. `runs/v2_focal_seed_2028/best_model/` — independent seed 2028.
3. `runs/v2_focal_seed_2029/best_model/` — independent seed 2029.

All three saved checkpoints include model configuration and tokenizer files; weight files are tracked with Git LFS.

## 5. Final validation results — frozen ensemble at 0.535

**Evaluation partition:** 307 validation issues. The ensemble's threshold was chosen on these records, not on the test labels.

| Micro-averaged metric | Reproduced SkillScope-method RF | Final RoBERTa ensemble | Difference |
|---|---:|---:|---:|
| Precision | 72.69% | **75.13%** | **+2.44 pp** |
| Recall | 30.86% | **34.01%** | **+3.16 pp** |
| F1 | 43.32% | **46.83%** | **+3.51 pp** |

More precisely, the ensemble reached precision **0.7513116474**, recall **0.3401425178**, and micro-F1 **0.4682799215**; the reproduced RF achieved **0.7269166200**, **0.3085510689**, and **0.4332166083**. The validation sweep found **seven** candidate global thresholds that improved all three metrics over the reproduced RF. The selected threshold **0.535** maximized the smallest absolute margin across the three metrics within the sweep.

Reference: `runs/v2_ensemble_rescue/validation_report.json`.

## 6. Final follow-up test — 614 issues

**These are the final figures for the teammate's results table and video.** Threshold **0.535** was selected from validation and held fixed for this comparison.

| Micro-averaged metric | Reproduced SkillScope-method RF | Final RoBERTa ensemble | Improvement |
|---|---:|---:|---:|
| **Precision** | **72.90%** | **77.67%** | **+4.77 percentage points** |
| **Recall** | **33.98%** | **35.29%** | **+1.31 percentage points** |
| **Micro-F1** | **46.35%** | **48.53%** | **+2.18 percentage points** |

Exact saved values:

| Metric | RF | Ensemble | Ensemble minus RF |
|---|---:|---:|---:|
| Precision | 0.7289871944 | 0.7766897540 | +0.0477025596 |
| Recall | 0.3398089863 | 0.3529411765 | +0.0131321901 |
| Micro-F1 | 0.4635428233 | 0.4853369152 | +0.0217940919 |

**Finding:** In this local **follow-up test** against the reproduced baseline, the ensemble improves **all three** specified metrics. Its largest absolute gain is in precision, followed by micro-F1 and recall.

**Important description:** This is a follow-up analysis on a test partition that had already been inspected during development of earlier v2 checkpoints. Do not call it a wholly untouched independent confirmation. It also does not establish the organizers' final eligibility verdict.

Reference: `runs/v2_ensemble_rescue/fixed_0535_followup.json`.

## 7. Development comparisons (not final submission results)

These provide research context and explain why the ensemble was selected. **Do not replace the final results table above with these intermediate scores.**

| Candidate / setting | Partition | Precision | Recall | Micro-F1 | Interpretation |
|---|---|---:|---:|---:|---|
| Weighted-BCE v2, threshold 0.320 | Follow-up/test | 45.67% | 48.13% | 46.87% | High recall, much lower precision |
| Original focal v2, threshold 0.545 | Follow-up/test | 76.47% | 33.55% | 46.64% | Precision and F1 above RF, recall just below RF |
| **Final three-model ensemble, threshold 0.535** | **Follow-up/test** | **77.67%** | **35.29%** | **48.53%** | **All three above reproduced RF** |

Relative to the original focal v2 model, the ensemble improved follow-up test **precision by 1.20 pp, recall by 1.75 pp, and micro-F1 by 1.90 pp** (based on displayed rounded percentages). The earlier weighted-BCE model was retained as an exploratory comparison, not the selected entry.

The new seed 2029 achieved approximately **59.80% validation micro-F1** at its own F1-optimal threshold, but that configuration traded substantial precision for recall; it is not interchangeable with the final ensemble's threshold and results.

## 8. Diagnostics and ablations

**Text sensitivity (individual focal model, not ensemble):**

| Diagnostic on corrected-v2 validation | Micro-F1 |
|---|---:|
| Focal RoBERTa, original issue text | **58.58%** |
| Same focal RoBERTa, shuffled issue text | **41.11%** |
| Frequency-only prediction using training label prevalence | **46.77%** |

The ~**17.47 pp** difference between original and shuffled text supports some dependence on issue-text content. It does not prove semantic understanding for every skill, nor does it measure the final ensemble at threshold 0.535. These diagnostic figures correspond to the individual focal checkpoint's validation-selected F1 operating point.

**Rejected validation-only calibration:** An exploratory regularized per-label threshold calibration increased precision but reduced recall and F1; none of the evaluated configurations improved all three metrics over the frozen original focal model. It was **not** incorporated into the final ensemble. Source: `runs/v2_roberta_focal/v2_calibration_probe.json` (local exploratory report).

**Long-tail distribution:** Some labels are much less frequent than others. The often-cited count of **86 labels with at most 20 positives** was measured in an older dataset analysis and **has not been verified for v2**; it should not be reported as a corrected-v2 statistic.

## 9. Limitations and interpretation

1. **Baseline scope:** The comparator is the team's leakage-controlled RF reproduction, not the original SkillScope published scores or an official competition benchmark.
2. **Follow-up rather than fresh holdout:** The 614 test records had been examined for earlier checkpoints before the final ensemble follow-up. The reported numbers are useful local evidence but are not a fresh independent confirmation.
3. **Local splitting protocol:** The 2,149/307/614 deterministic split is the team's split; equivalence to an official competition split has not been established.
4. **Long-tail label imbalance:** Overall micro-averaged scores can be dominated by common labels. Per-label and rare-label generalization are not established by the reported overall metrics.
5. **Independent environment reproducibility:** Model inference and label-order checking were verified in the established project environment; a full fresh dependency installation was not independently verified.
6. **Competition status:** Passing the three-metric rule against the local reproduced RF does not itself certify acceptance, ranking, or eligibility against an organizer-defined reference.

For a standalone limitations handoff, see `FINAL_LIMITATIONS.md`.

## 10. Reproduce and locate the final evidence

**Repository branch:** `final-nlbse-2027`  
**Ensemble implementation commit:** `5012b3e` (later README documentation commits may exist)  
**Repository:** https://github.com/Xaeltos/SeniorCoders-CS440-Labs/tree/final-nlbse-2027

| Purpose | File / directory |
|---|---|
| Corrected dataset | `dataset_v2/skillscope_data_v2.db` |
| Final training | `roberta_v2_focal.py`, `v2_ensemble_rescue.py` |
| Final prediction interface | **`nlbse_predict_ensemble_v2.py`** |
| Final follow-up evaluator | `v2_ensemble_followup.py` |
| Final validation report | `runs/v2_ensemble_rescue/validation_report.json` |
| Final test report | `runs/v2_ensemble_rescue/fixed_0535_followup.json` |
| Original focal model | `runs/v2_roberta_focal/best_model/` |
| Seed 2028 model | `runs/v2_focal_seed_2028/best_model/` |
| Seed 2029 model | `runs/v2_focal_seed_2029/best_model/` |
| Historical single-model v2 results | `V2_EXPERIMENT_RESULTS.md` |
| Historical v1 results | `EXPERIMENT_RESULTS.md` |

**Existing-environment prediction command (Windows PowerShell):**

```powershell
.\.venv\Scripts\python.exe nlbse_predict_ensemble_v2.py --title "Fix connection timeout" --description "Implement retry logic" --check-label-order dataset_v2/skillscope_data_v2.db
```

**Output:** JSON with a `row` index and a 217-element binary `vector` of skill predictions. All three checkpoints were exercised successfully with the corrected-v2 label-order check.

**Do not run the training/evaluation scripts over existing reports unless directing outputs to new locations.** Several scripts intentionally refuse to overwrite saved experiment results. For the video, show the already-saved final JSON reports rather than retraining the models.
