"""NLBSE 2027 -- same-split SkillScope RF+TF-IDF vs RoBERTa comparison.

Reproduces the OFFICIAL SkillScope RandomForestClassifier(random_state=42)
with TfidfVectorizer(max_features=1000) and SkillScope's text cleaning,
except it intentionally prevents the original baseline's pre-split leakage.
No MLSMOTE is performed (its original implementation generates hundreds of
thousands of rows prior to splitting and is unsuitable for honest evaluation).
This is an honest same-split SkillScope *method* benchmark, NOT the published
SkillScope score nor a reproduction of the original oversampled evaluation.

Only the train split is used to fit preprocessing / RF. Only validation
is used for scores; test rows are NEVER used in training or inference.

Usage:
 .\\.venv\\Scripts\\python.exe compare_skillscope_roberta.py
 .\\.venv\\Scripts\\python.exe compare_skillscope_roberta.py --rf-only
"""
from __future__ import annotations
import argparse
import json
import re
import string
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support

from skill_classifier import load_dataset, split_indices, SkillDataset


def skill_scope_clean(value, demojize):
    """Same transformations, same order as official competition_baseline/main.py."""
    if not isinstance(value, str):
        return ''
    value = value.replace('"', '')
    value = re.sub(r'DevTools.*?\(automated\)', '', value)
    value = value.lower()
    value = demojize(value)
    value = re.sub(r'https?://\S+|www\.\S+', '', value)
    value = re.sub(r'<.*?>', '', value)
    value = re.sub(f'[{re.escape(string.punctuation)}]', '', value)
    value = value.replace('#', '')
    value = re.sub(r'\s+', ' ', value)
    return ' '.join(word for word in value.split() if len(word) <= 20)


def metrics(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0)
    return dict(precision=float(p), recall=float(r), micro_f1=float(f),
                predicted_labels_per_issue=float(y_pred.sum(axis=1).mean()))


def roberta_scores(texts, labels, val, names, checkpoint, threshold, batch_size, max_length):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    if not checkpoint.is_dir():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=True)
    if model.config.num_labels != len(names):
        raise ValueError(f'Wrong label count in RoBERTa checkpoint: {model.config.num_labels}')
    if model.config.id2label:
        got = [model.config.id2label[i] for i in range(len(names))]
        if got != names:
            raise ValueError('RoBERTa label ORDER differs from the SQLite dataset. Stopping.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    dl = DataLoader(SkillDataset(texts, labels, val, tokenizer, max_length),
                    batch_size=batch_size, shuffle=False)
    results = []
    with torch.inference_mode():
        for k, batch in enumerate(dl, start=1):
            batch.pop('labels')
            logit = model(**{key: tensor.to(device) for key, tensor in batch.items()}).logits
            results.append(torch.sigmoid(logit.float()).cpu().numpy())
            if k % 100 == 0:
                print(f'RoBERTa validation batches: {k}/{len(dl)}', flush=True)
    probs = np.concatenate(results, axis=0)
    if probs.shape != labels[val].shape:
        raise ValueError(f'Prediction shape mismatch: {probs.shape} vs {labels[val].shape}')
    return metrics(labels[val].astype(np.uint8), (probs >= threshold).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--db', type=Path, default=Path('skillscope_data.db'))
    ap.add_argument('--checkpoint', type=Path, default=Path('runs/roberta_recovery/best_model'))
    ap.add_argument('--threshold', type=float, default=0.375, help='Previously chosen validation threshold')
    ap.add_argument('--max-length', type=int, default=256)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--output', type=Path, default=Path('runs/skillscope_same_split/comparison.json'))
    ap.add_argument('--rf-only', action='store_true', help='Only train/evaluate SkillScope method; no GPU')
    args = ap.parse_args()
    if not (0 < args.threshold < 1):
        ap.error('threshold must be strictly between zero and one')
    try:
        import emoji
    except ImportError:
        ap.error('Official SkillScope cleaning requires emoji: .\\.venv\\Scripts\\python.exe -m pip install emoji')

    # Labels and split indices match the user's ACTUAL existing RoBERTa pipeline.
    texts, labels, names = load_dataset(args.db)
    train, val, test = split_indices(len(texts))
    if len(texts) != 7154 or len(names) != 217:
        raise ValueError('Unexpected dataset shape; refusing to compare different data')
    if len(train) != 5007 or len(val) != 716 or len(test) != 1431:
        raise ValueError('Unexpected train/val/test sizes; stopping')
    if (np.intersect1d(train,val).size or np.intersect1d(train,test).size or
            np.intersect1d(val,test).size):
        raise AssertionError('Split overlap detected')
    print('COMPARISON: SkillScope official RF+TF-IDF METHOD vs saved RoBERTa', flush=True)
    print('Train=5007, validation=716, test=1431 (TEST NEVER ACCESSED)', flush=True)
    print('Official method: TF-IDF max_features=1000; RandomForestClassifier(random_state=42)', flush=True)
    print('Safety adjustment: split BEFORE TF-IDF; NO pre-split MLSMOTE or duplication', flush=True)

    # Read the original text columns, keeping exact row order. No answer-derived metadata.
    import sqlite3
    with sqlite3.connect(args.db) as conn:
        frame = pd.read_sql_query(
            'SELECT "issue text", "issue description" FROM "nlbse_tool_competition_data_by_issue"', conn)
    if len(frame) != len(texts):
        raise ValueError('SQL row alignment failed')
    cleaned_title = frame['issue text'].map(lambda x: skill_scope_clean(x, emoji.demojize))
    cleaned_desc = frame['issue description'].map(lambda x: skill_scope_clean(x, emoji.demojize))
    combined = (cleaned_title + ' ' + cleaned_desc).tolist()

    tfidf = TfidfVectorizer(max_features=1000)
    X_train = tfidf.fit_transform([combined[int(i)] for i in train])
    X_val = tfidf.transform([combined[int(i)] for i in val])
    y_train = labels[train].astype(np.uint8)
    y_val = labels[val].astype(np.uint8)
    print(f'SkillScope TF-IDF features: {X_train.shape[1]} (fit TRAIN only)', flush=True)
    print('Fitting official 100-tree multi-output Random Forest on TRAIN...', flush=True)
    # Exact RF estimator from the official extraction; sparse-to-dense makes
    # it comparable to its 1000-column DataFrame.
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train.toarray(), y_train)
    pred = clf.predict(X_val.toarray()).astype(np.uint8)
    baseline = metrics(y_val, pred)
    print('SKILLSCOPE RF SAME-SPLIT VALIDATION:', json.dumps(baseline), flush=True)
    report = {
        'dataset': str(args.db), 'seed': 42, 'train_rows':len(train),
        'validation_rows':len(val), 'test_rows_untouched':len(test),
        'labels':len(names), 'tfidf_features':int(X_train.shape[1]),
        'baseline_source':'nlbse2027/skill-classification/competition_baseline/main.py',
        'baseline_method':'SkillScope cleaning + 1000-feature TF-IDF + default 100-tree RandomForest',
        'changes_from_official_implementation':[
            'train/validation/test split before fitting TF-IDF',
            'no MLSMOTE (original synthesizes hundreds of thousands of samples before split)',
            'no duplicate concatenation',
            'evaluate on same 716 validation examples as RoBERTa',
            'retain null text as empty so alignment with RoBERTa is unchanged'],
        'NOT_published_or_official_competition_score':True,
        'skillscope_rf_validation':baseline,
    }
    if not args.rf_only:
        print('Evaluating existing RoBERTa checkpoint on SAME validation rows...', flush=True)
        rb = roberta_scores(texts, labels, val, names, args.checkpoint,
                            args.threshold, args.batch_size, args.max_length)
        report['roberta_checkpoint'] = str(args.checkpoint)
        report['roberta_threshold_previously_selected_on_validation'] = args.threshold
        report['roberta_validation'] = rb
        report['roberta_minus_skillscope'] = {k:float(rb[k]-baseline[k]) for k in
                                             ('precision','recall','micro_f1')}
        print('ROBERTA SAME-SPLIT VALIDATION:', json.dumps(rb), flush=True)
        print('ROBERTA MINUS SKILLSCOPE RF:', json.dumps(report['roberta_minus_skillscope']), flush=True)
    else:
        print('Skipped RoBERTa inference (--rf-only).', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Saved:', args.output, flush=True)
    print('CAUTION: this is a leakage-controlled SkillScope-method comparison,', flush=True)
    print('NOT the published 88.9% score or official competition qualification.', flush=True)

if __name__ == '__main__':
    main()
