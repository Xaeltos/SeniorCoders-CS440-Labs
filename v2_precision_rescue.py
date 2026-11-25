"""Validation-only precision/recall feasibility audit for existing NLBSE v2 checkpoint.
No training, no test predictions, no changes to saved models.
"""
import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from compare_skillscope_roberta import skill_scope_clean, metrics
from skill_classifier import load_dataset, split_indices, SkillDataset
import emoji


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', type=Path, default=Path('dataset_v2/skillscope_data_v2.db'))
    ap.add_argument('--model', type=Path, default=Path('runs/v2_roberta_recovery/best_model'))
    ap.add_argument('--out', type=Path, default=Path('runs/v2_roberta_recovery/v2_precision_audit.json'))
    a = ap.parse_args()
    if not a.db.exists() or not a.model.exists():
        ap.error('v2 database or saved model missing. Run from original project folder.')
    if a.out.exists():
        ap.error(f'Refusing to overwrite existing report: {a.out}')

    texts, y, names = load_dataset(a.db)
    train, val, _test = split_indices(len(texts))
    print(f'V2 validation only: train={len(train)} val={len(val)}. TEST IS NOT READ.', flush=True)
    with sqlite3.connect(a.db) as conn:
        frame = pd.read_sql_query('SELECT "issue text", "issue description" FROM "nlbse_tool_competition_data_by_issue"',conn)
    clean = (frame['issue text'].map(lambda z: skill_scope_clean(z,emoji.demojize)) + ' ' +
             frame['issue description'].map(lambda z: skill_scope_clean(z,emoji.demojize))).tolist()
    tfidf = TfidfVectorizer(max_features=1000)
    xtr = tfidf.fit_transform([clean[int(i)] for i in train])
    xval = tfidf.transform([clean[int(i)] for i in val])
    rf = RandomForestClassifier(random_state=42)
    rf.fit(xtr.toarray(),y[train].astype(np.uint8))
    rf_m = metrics(y[val].astype(np.uint8),rf.predict(xval.toarray()).astype(np.uint8))
    print('SKILLSCOPE VALIDATION:',json.dumps(rf_m),flush=True)

    tk = AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(a.model,local_files_only=True)
    if model.config.num_labels != 217 or [model.config.id2label[i] for i in range(217)] != names:
        raise RuntimeError('Checkpoint skill labels do not match v2 database')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    loader = DataLoader(SkillDataset(texts,y,val,tk,256),batch_size=4,shuffle=False)
    probs=[]
    with torch.inference_mode():
        for batch in loader:
            batch.pop('labels')
            z=model(**{k:v.to(device) for k,v in batch.items()}).logits
            probs.append(torch.sigmoid(z.float()).cpu().numpy())
    probs=np.concatenate(probs)
    truth=y[val].astype(np.uint8)
    candidates=[]
    for threshold in np.round(np.arange(0.05,0.951,0.005),3):
        mm=metrics(truth,(probs>=threshold).astype(np.uint8))
        margin={k: mm[k]-rf_m[k] for k in ('precision','recall','micro_f1')}
        candidates.append({'threshold':float(threshold),'metrics':mm,'margins':margin,'clean_sweep':all(v>=0 for v in margin.values()) and any(v>0 for v in margin.values())})
    winners=[x for x in candidates if x['clean_sweep']]
    best_f1=max(candidates,key=lambda x:x['metrics']['micro_f1'])
    best_precision_with_recall=max((x for x in candidates if x['metrics']['recall']>=rf_m['recall']),key=lambda x:x['metrics']['precision'],default=None)
    print('VALIDATION CLEAN SWEEP THRESHOLDS:',len(winners),flush=True)
    print('BEST F1:',json.dumps(best_f1),flush=True)
    print('BEST PRECISION WHILE MEETING BASELINE RECALL:',json.dumps(best_precision_with_recall),flush=True)
    if winners:
        winner=max(winners,key=lambda x:(min(x['margins'].values()),x['metrics']['micro_f1']))
        print('BEST VALIDATION CLEAN SWEEP:',json.dumps(winner),flush=True)
    else:
        print('NO VALIDATION CLEAN SWEEP from global thresholds. Do not attempt to tune on TEST.',flush=True)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps({'split':'validation only','note':'Exploratory analysis AFTER v2 locked test was previously observed. Any changed setting requires a NEW untouched evaluation dataset for confirmatory claims.','baseline':rf_m,'best_f1':best_f1,'best_precision_with_recall':best_precision_with_recall,'sweep_count':len(winners),'winners':winners,'all_thresholds':candidates},indent=2),encoding='utf-8')
    print('Saved',a.out,flush=True)

if __name__=='__main__':
    main()
