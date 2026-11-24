"""One-time locked held-out comparison of SkillScope-method RF and RoBERTa.

Uses exactly the existing random_state=42 train/validation/test split, trains
SkillScope's leakage-controlled TF-IDF+RF only on train, and evaluates BOTH
models on the previously untouched test split. Threshold 0.54 was selected
on validation; DO NOT tune it after seeing these test results.

Run once:
  .\\.venv\\Scripts\\python.exe verify_clean_sweep_test.py
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from compare_skillscope_roberta import metrics, skill_scope_clean
from skill_classifier import load_dataset, split_indices, SkillDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path, default=Path('skillscope_data.db'))
    parser.add_argument('--checkpoint', type=Path, default=Path('runs/roberta_recovery/best_model'))
    parser.add_argument('--output', type=Path, default=Path('runs/skillscope_same_split/locked_test.json'))
    args = parser.parse_args()
    import emoji
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from torch.utils.data import DataLoader

    texts, labels, names = load_dataset(args.db)
    train, val, test = split_indices(len(texts))
    if (len(texts), len(names), len(train), len(val), len(test)) != (7154,217,5007,716,1431):
        raise ValueError('Dataset or split does not match prior experiments')
    if any(np.intersect1d(a,b).size for a,b in ((train,val),(train,test),(val,test))):
        raise ValueError('Split overlap!')
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(args.checkpoint)
    with sqlite3.connect(args.db) as conn:
        frame = pd.read_sql_query('SELECT "issue text", "issue description" FROM "nlbse_tool_competition_data_by_issue"', conn)
    if len(frame) != len(texts):
        raise ValueError('Text row count mismatch')
    cleaned_title = frame['issue text'].map(lambda s: skill_scope_clean(s, emoji.demojize))
    cleaned_desc = frame['issue description'].map(lambda s: skill_scope_clean(s, emoji.demojize))
    combined = (cleaned_title + ' ' + cleaned_desc).tolist()

    print('LOCKED TEST: train=5007, validation=716 (threshold chosen here), test=1431', flush=True)
    print('Threshold FIXED at 0.54; baseline RF default hard predictions; no tuning on test.', flush=True)
    vectorizer = TfidfVectorizer(max_features=1000)
    x_train = vectorizer.fit_transform([combined[int(i)] for i in train])
    x_test = vectorizer.transform([combined[int(i)] for i in test])
    rf = RandomForestClassifier(random_state=42)
    print('Training SkillScope-method 100-tree RF on TRAIN ONLY...', flush=True)
    rf.fit(x_train.toarray(), labels[train].astype(np.uint8))
    rf_pred = rf.predict(x_test.toarray()).astype(np.uint8)
    rf_stats = metrics(labels[test].astype(np.uint8), rf_pred)

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint, local_files_only=True)
    if model.config.num_labels != 217:
        raise ValueError('Checkpoint does not have 217 labels')
    if model.config.id2label:
        checkpoint_names = [model.config.id2label[i] for i in range(217)]
        if checkpoint_names != names:
            raise ValueError('Checkpoint label ordering does not match database')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    loader = DataLoader(SkillDataset(texts, labels, test, tokenizer, 256), batch_size=4, shuffle=False)
    probs = []
    with torch.inference_mode():
        for k,batch in enumerate(loader, 1):
            batch.pop('labels')
            logits = model(**{key:value.to(device) for key,value in batch.items()}).logits
            probs.append(torch.sigmoid(logits.float()).cpu().numpy())
            if k % 100 == 0:
                print(f'RoBERTa test inference: {k}/{len(loader)} batches', flush=True)
    scores = np.concatenate(probs)
    assert scores.shape == labels[test].shape
    rb_pred = (scores >= 0.54).astype(np.uint8)
    rb_stats = metrics(labels[test].astype(np.uint8), rb_pred)
    margins = {key:float(rb_stats[key] - rf_stats[key]) for key in ('precision','recall','micro_f1')}
    clean = (all(x >= 0 for x in margins.values()) and any(x > 0 for x in margins.values()))
    report = {
        'comparison':'leakage-controlled SkillScope RF method vs RoBERTa on identical held-out test',
        'not_official_competition_qualification':True,
        'threshold_locked_from_validation':0.54,
        'model_checkpoint':str(args.checkpoint),
        'train_count':len(train),'validation_count':len(val),'heldout_test_count':len(test),
        'baseline_method':'SkillScope cleaning, 1000-feature TF-IDF, 100-tree RF; no pre-split MLSMOTE or duplicate records',
        'skillscope_rf_test':rf_stats,
        'roberta_test':rb_stats,
        'roberta_minus_skillscope':margins,
        'clean_sweep_on_locked_test':bool(clean),
    }
    print('SKILLSCOPE RF LOCKED TEST:', json.dumps(rf_stats), flush=True)
    print('ROBERTA LOCKED TEST:', json.dumps(rb_stats), flush=True)
    print('ROBERTA MINUS SKILLSCOPE:', json.dumps(margins), flush=True)
    print('CLEAN SWEEP ON LOCKED TEST:', 'YES' if clean else 'NO', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Saved:', args.output, flush=True)
    print('Do not tune threshold after examining test. This does not establish official competition qualification.', flush=True)


if __name__ == '__main__':
    main()
