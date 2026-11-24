"""Validation-only RoBERTa threshold sweep vs saved same-split SkillScope RF baseline.

Run: .\\.venv\\Scripts\\python.exe find_clean_sweep.py

Uses the *existing* RF comparison JSON and existing RoBERTa checkpoint.
No retraining, test-set inference, downloading, or ground-truth leakage.
A validation clean sweep is NOT official competition qualification.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from skill_classifier import load_dataset, split_indices, SkillDataset


def score(y, pred):
    tp = int(np.logical_and(y, pred).sum())
    fp = int(np.logical_and(~y, pred).sum())
    fn = int(np.logical_and(y, ~pred).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {'precision':precision, 'recall':recall, 'micro_f1':f1, 'predicted_labels_per_issue':float(pred.sum(axis=1).mean())}


def get_probs(texts, labels, val, names, checkpoint, max_length, batch_size):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=True)
    if model.config.num_labels != len(names):
        raise ValueError('Checkpoint label count mismatch')
    if model.config.id2label:
        ordering = [model.config.id2label[i] for i in range(len(names))]
        if ordering != list(names):
            raise ValueError('Checkpoint label ordering mismatch')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    loader = DataLoader(SkillDataset(texts, labels, val, tokenizer, max_length),
                        batch_size=batch_size, shuffle=False)
    output = []
    with torch.inference_mode():
        for i, batch in enumerate(loader, 1):
            batch.pop('labels')
            logits = model(**{k:v.to(device) for k,v in batch.items()}).logits
            output.append(torch.sigmoid(logits.float()).cpu().numpy())
            if i % 100 == 0: print(f'Inference: {i}/{len(loader)} batches', flush=True)
    result = np.concatenate(output, axis=0)
    if result.shape != labels[val].shape:
        raise ValueError(f'Wrong prediction dimensions {result.shape}')
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--db', type=Path, default=Path('skillscope_data.db'))
    ap.add_argument('--checkpoint', type=Path, default=Path('runs/roberta_recovery/best_model'))
    ap.add_argument('--baseline', type=Path, default=Path('runs/skillscope_same_split/comparison.json'))
    ap.add_argument('--max-length', type=int, default=256)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--output', type=Path, default=Path('runs/skillscope_same_split/clean_sweep_thresholds.json'))
    args = ap.parse_args()

    if not args.baseline.is_file():
        ap.error(f'Cannot find comparison JSON: {args.baseline}')
    baseline_report = json.loads(args.baseline.read_text(encoding='utf-8'))
    baseline = baseline_report['skillscope_rf_validation']
    texts, labels, names = load_dataset(args.db)
    train, val, test = split_indices(len(texts))
    assert (len(texts),len(train),len(val),len(test),len(names)) == (7154,5007,716,1431,217)
    if (baseline_report.get('train_rows'),baseline_report.get('validation_rows'),baseline_report.get('labels')) != (5007,716,217):
        raise ValueError('Comparison metadata does not match this dataset/split')
    print('Threshold tuning: 716 VALIDATION issues only; 1431 held-out TEST issues untouched.',flush=True)
    print('REFERENCE SkillScope-method RF:',json.dumps(baseline),flush=True)
    probs = get_probs(texts, labels, val, names, args.checkpoint, args.max_length, args.batch_size)
    y = labels[val].astype(bool)
    candidates=[]
    # Grid spans useful low/high regions at finer-than-1% increments.
    for threshold in np.round(np.arange(0.05,0.951,0.005),3):
        m = score(y, probs >= threshold)
        margins = {key:m[key]-baseline[key] for key in ('precision','recall','micro_f1')}
        candidate={'threshold':float(threshold), **m, 'margins':margins,
                   'meets_all_three':all(d >= 0 for d in margins.values()),
                   'strictly_improves_at_least_one':any(d > 1e-10 for d in margins.values())}
        candidates.append(candidate)
    qualifying=[c for c in candidates if c['meets_all_three'] and c['strictly_improves_at_least_one']]
    selected=max(qualifying,key=lambda c:(c['micro_f1'],c['recall'],c['precision'])) if qualifying else None
    best_f1=max(candidates,key=lambda c:c['micro_f1'])
    # If not feasible, show threshold that maximizes the worst margin in percentage points.
    closest=max(candidates,key=lambda c:min(c['margins'].values()))
    print('BEST VALIDATION F1:',json.dumps(best_f1),flush=True)
    print('CLEAN SWEEP THRESHOLDS FOUND:',len(qualifying),flush=True)
    if selected:
        print('BEST CLEAN SWEEP:',json.dumps(selected),flush=True)
    else:
        print('NO CLEAN SWEEP ON VALIDATION. CLOSEST:',json.dumps(closest),flush=True)
    report={
        'description':'Validation-only threshold selection vs leakage-controlled SkillScope RF method',
        'not_official_competition_qualification':True,
        'test_rows_untouched':len(test), 'validation_rows':len(val),
        'checkpoint':str(args.checkpoint), 'same_split_baseline':baseline,
        'best_f1':best_f1, 'clean_sweep_count':len(qualifying),
        'best_clean_sweep':selected,'closest_if_no_sweep':None if selected else closest,
        'all_thresholds':candidates,
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Saved:',args.output,flush=True)
    print('NOTE: This is SAME-SPLIT VALIDATION versus a leakage-controlled RF METHOD, not a competition qualification score.',flush=True)

if __name__=='__main__':
    main()
