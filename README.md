# Tool Competition - Skill Classification

## Competition Overview

The 2027 competition consists of building and assessing multi‑label classifiers that predict, for each issue, the set of domains and sub‑domains representing the skills required to solve it.


## Dataset

We release a dataset mined from 7,245 merged pull requests across 11 popular Java repositories (57,206 source files; 59,644 methods; 13,097 classes) annotated with 217 skill labels composed by domain/sub‑domains You may find the dataset in a SQLite database, named `skillscope_data.db`, [here](https://huggingface.co/datasets/NLBSE/SkillCompetition). This dataset is an updated version of the one submitted in the original [SkillScope](#citing-relevant-work) paper.

### Ready-made Table
Inside the database you will find a table named `nlbse_tool_competition_data_by_issue` that joins each pull request’s textual and code‑context features with its canonical domain/sub‑domain labels per issue. There is also a view `vw_nlbse_tool_competition_data_by_file` that labels each filename/function associated with each issue. 

- The `nlbse_tool_competition_data_by_issue` table contains a column for each domain and subdomain with an integer count of the number of matching APIs found for that issue. A value greater than zero indicates that domain/subdomain is present in the issue.

- The `vw_nlbse_tool_competition_data_by_file` is present for convenience purposes only and will not be used in evaluation of the model.

### Additional Metadata/Input Data is Allowed
Your model is allowed to use input data that is outside what is given in the database. You may use additional GitHub APIs to fetch more metadata, or download relevant files in an issue for further analysis. However, you must not use any third-party classification engine or the outputs present in `skillscope_data.db` as direct inputs to the model. 

### Baselines
Your models will be compared against the SkillScope Random‑Forest + TF‑IDF baselines reported in the paper by evaluating the overall prediction metrics against the issue classifications as recorded in the `nlbse_tool_competition_data_by_issue` table. The models you create should return a multi-label classification encoded in an one-hot encoded vector. However, it is the metrics of your models which will be the subject to the evaluation instead of the specific model output. Random Forest Baseline is provided in the folder `competition_baseline` with README and requirements.txt.

### Goal
Train, tune and evaluate your models on the provided splits and improve at least one of precision, recall, or micro‑F1 while not decreasing the remaining metrics relative to the best baseline.

## Organizers

The skill classification competition is organized by: Fabio Santos (fabio.deabreusantos@colostate.edu), and Jacob Penney (jmp458@nau.edu)

## Submission Acceptance & Competition

Submissions will first be filtered to ensure they do not lower any of the three core metrics (precision, recall, micro‑F1) compared with the baseline. Among qualifying submissions, ranking is determined by the largest positive improvement in micro‑F1. Ties will be broken by (1) precision, then (2) runtime.

Submissions will be judged on:
- Clarity and completeness of the paper.
- Availability and reproducibility of code/tool (open‑source required).
- Correct reporting of metrics.
- Quality of documentation.

Accepted papers will appear in the **NLBSE ’27 proceedings**.

### Ranking Details

Participants submit a single multi‑label classifier. Rankings proceed in two stages:

1. Eligibility check: The submission must *not* reduce any of precision, recall, or micro‑F1 compared with the baseline.
2. Ordering: Qualifying submissions are ordered by the greatest positive ∆micro‑F1.

## Citing Relevant Work

### General Competition Citation:

```tex
@inproceedings{nlbse2027,
  author = {Mock, Moritz and Borsani, Thomas and Russo, Barbara and Santos, Fabio and and Penney, Jacob and Valenzuela{-}Toledo, Pablo and Kehrer, Timo and Panichella, Sebastiano },
  title={The NLBSE'27 Tool Competition and Challenge},
  booktitle={Proceedings of The 6th International Workshop on Natural Language-based Software Engineering (NLBSE'27)},
  year={2027}
}
```

### Skill Competition Citation:
```tex
@inproceedings{carter2025skillscope,
    title        = {SkillScope: A Tool to Predict Fine-Grained Skills Needed to Solve Issues on GitHub},
    author       = {Carter, Benjamin C. and Contreras, Jonathan Rivas and Llanes Villegas, Carlos A. and Acharya, Pawan and Utzerath, Jack and Farner, Adonijah O. and Jenkins, Hunter and Johnson, Dylan and Penney, Jacob and Steinmacher, Igor and Gerosa, Marco A. and Santos, Fabio},
    year         = 2025,
    booktitle    = {2025 IEEE/ACM International Workshop on Natural Language-Based Software Engineering (NLBSE)},
    volume       = {},
    number       = {},
    pages        = {9--12},
    doi          = {10.1109/NLBSE66842.2025.00007},
    keywords     = {Random Forest;Training;Java;Large language models;Semantics;Retrieval augmented generation;Machine learning;Open source software;Software engineering;Software development management;software engineering;skill categorization;open source software (OSS);machine learning;large language models}
}
```
```tex
@article{santos2023tag,
    title        = {Tag that issue: applying API-domain labels in issue tracking systems},
    author       = {Santos, Fabio and Vargovich, Joseph and Trinkenreich, Bianca and Santos, Italo and Penney, Jacob and Britto, Ricardo and Pimentel, Jo{\~a}o Felipe and Wiese, Igor and Steinmacher, Igor and Sarma, Anita and others},
    year         = 2023,
    journal      = {Empirical Software Engineering},
    publisher    = {Springer},
    volume       = 28,
    number       = 5,
    pages        = 116
}
```
```tex
@inproceedings{vargovich2023givemelabeledissues,
    title        = {Givemelabeledissues: An open source issue recommendation system},
    author       = {Vargovich, Joseph and Santos, Fabio and Penney, Jacob and Gerosa, Marco A and Steinmacher, Igor},
    year         = 2023,
    booktitle    = {2023 IEEE/ACM 20th International Conference on Mining Software Repositories (MSR)},
    pages        = {402--406},
    organization = {IEEE}
}
```

---

## Senior Coders: RoBERTa skill classifier (CS440)

This section documents the team's implementation. The competition overview and organizer instructions above remain unchanged.

### Implemented system

- **Task:** Predict 217 domain/subdomain skill labels for a GitHub issue (multi-label classification).
- **Inputs:** Issue title and description. API-derived label columns and completed pull-request code are **not** model inputs.
- **Model:** `FacebookAI/roberta-base`, fine-tuned for 217 binary outputs.
- **Final checkpoint:** `runs/roberta_recovery/best_model/` (weights stored using Git LFS).
- **Decision rule:** Apply sigmoid to each output logit and predict 1 if the probability is at least **0.54**; this threshold was selected on validation data.

### Setup (Windows PowerShell)

From the repository root, with Python installed and Git LFS enabled:

```powershell
git lfs pull
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The tested environment used Python 3.13.2 and a CUDA-enabled PyTorch 2.14.1+cu130 installation. Depending on platform and CUDA availability, installing that PyTorch build can require the appropriate PyTorch package index; `requirements-lock.txt` records the tested dependency versions. GPU acceleration is optional for inference but recommended for training. Installation in a newly created environment has not yet been independently tested.

### Predict one issue

```powershell
.\.venv\Scripts\python.exe nlbse_predict.py --model runs/roberta_recovery/best_model --check-label-order skillscope_data.db --title "Fix Kafka consumer timeout" --description "Improve retry handling after broker disconnects."
```

### Predict a batch of issues

Provide a UTF-8 JSON Lines file with one issue per line, for example:

```jsonl
{"title":"Fix Kafka consumer timeout","description":"Improve retry handling."}
{"title":"Correct texture rendering","description":"Fix framebuffer coordinates."}
```

Then run (choose an output file that does not already exist):

```powershell
.\.venv\Scripts\python.exe nlbse_predict.py --input issues.jsonl --output predictions.jsonl --check-label-order skillscope_data.db
```

Each output row has a `row` index and a 217-element binary `vector`. The optional `--include-labels` flag includes the corresponding ordered label names. For an adapter logic check without loading weights:

```powershell
.\.venv\Scripts\python.exe nlbse_predict.py --self-test
```

### Train and evaluate

The full issue-level competition table in the local SQLite database has **7,154 labeled records**. The separate raw `pull_requests` table contains **7,245 records**; the 91-record difference was identified by repository and PR number, but the exclusion mechanism is not yet established. Do not add unmatched raw rows to the labeled training data.

Training and validation use the deterministic split implemented in `skill_classifier.py` (seed 42): **5,007 train**, **716 validation**, and **1,431 held-out test** records. The original BCE and focal-loss experiments are in `skill_classifier.py`; a weighted focal-loss implementation is in `skill_classifier_weighted.py`. The selected checkpoint was generated by the weighted-BCE training experiment in `roberta_recovery.py`.

To train a *new* experiment without overwriting the selected checkpoint:

```powershell
.\.venv\Scripts\python.exe roberta_recovery.py --epochs 4 --output runs/new_recovery_experiment
```

For the existing comparison results, see:

- `runs/skillscope_same_split/comparison.json` — validation comparison.
- `runs/skillscope_same_split/clean_sweep_thresholds.json` — validation-only threshold search.
- `runs/skillscope_same_split/locked_test.json` — final held-out comparison.
- `runs/roberta_recovery/recovery_metrics.json` — recovery training/validation history.

The verification code is in `compare_skillscope_roberta.py`, `find_clean_sweep.py`, and `verify_clean_sweep_test.py`. The held-out test evaluation was already performed; **do not retune the model or decision threshold using the test results**.

### Held-out test comparison

Both methods were evaluated on the **same 1,431 held-out issues**. The SkillScope-method reproduction used its text cleaning, 1,000-feature TF-IDF, and 100-tree Random Forest, but deliberately **did not** apply pre-split MLSMOTE or pre-split duplication.

| Metric | Leakage-controlled SkillScope-method RF | RoBERTa, threshold 0.54 | Difference (percentage points) |
|---|---:|---:|---:|
| Micro precision | 67.73% | 69.36% | +1.63 |
| Micro recall | 48.27% | 50.32% | +2.04 |
| Micro-F1 | 56.37% | 58.32% | +1.95 |

**Limitations:** This is a leakage-controlled reproduction of the SkillScope **method**, not the original published baseline result or proof of official NLBSE competition qualification. The final RoBERTa model's predictions were found to depend heavily on frequent label distributions, and rare-label performance remains weak. The experiment should not be described as demonstrating robust semantic skill recognition. The competition's official evaluation protocol and submission acceptance remain separate from this local held-out comparison.

### Reproducibility notes

- The checkpoint, tokenizer, and database were retrieved successfully from a fresh GitHub checkout through Git LFS.
- A prediction from that checkout successfully loaded the model and verified its 217-label order using the existing Python environment.
- A fresh dependency installation in an independent environment has **not** yet been verified.
- Keep the original competition files and this repository's `final-nlbse-2027` branch separate from other coursework branches.
