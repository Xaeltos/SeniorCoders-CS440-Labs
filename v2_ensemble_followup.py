"""Follow-up comparison for the frozen v2 three-model ensemble.

Threshold 0.535 was selected on validation before this follow-up.
The 614-row test partition has been viewed in earlier experiments; this is
NOT a fresh independent test or proof of official NLBSE qualification.
No model weights, training scripts, or published predictor are changed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from skill_classifier import load_dataset, split_indices
from compare_skillscope_roberta import metrics
from v2_ensemble_rescue import MODEL_PATHS, baseline_validation, read_probabilities

THRESHOLD = 0.535
OUTPUT = Path('runs/v2_ensemble_rescue/fixed_0535_followup.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path, default=Path('dataset_v2/skillscope_data_v2.db'))
    args = parser.parse_args()
    if not args.db.is_file():
        parser.error(f'Missing v2 dataset: {args.db}')
    for model in MODEL_PATHS:
        if not (model / 'model.safetensors').is_file():
            parser.error(f'Missing checkpoint: {model}')
    if OUTPUT.exists():
        parser.error(f'Refusing to overwrite existing report: {OUTPUT}')
    texts, truth, labels = load_dataset(args.db)
    train, validation, test = split_indices(len(texts))
    if (len(texts), len(labels), len(train), len(validation), len(test)) != (3070, 217, 2149, 307, 614):
        raise ValueError('Unexpected v2 dataset or split')

    print('FROZEN ENSEMBLE THRESHOLD (selected on validation):', THRESHOLD, flush=True)
    print('TEST WAS VIEWED IN PRIOR EXPERIMENTS; FOLLOW-UP ONLY.', flush=True)
    print('Recreating train-only TF-IDF + Random Forest baseline...', flush=True)
    # Uses identical training/test row IDs and the train-only RF reproduction
    # as the other v2 comparison scripts. Does NOT run original MLSMOTE baseline.
    # Baseline helper accepts any evaluation indices, not just validation.
    rf = baseline_validation(args.db, truth, train, test)
    y = truth[test].astype(np.uint8)
    probs = []
    for checkpoint in MODEL_PATHS:
        print('FOLLOW-UP INFERENCE:', checkpoint, flush=True)
        probs.append(read_probabilities(checkpoint, texts, truth, labels, test))
    if not all(p.shape == (len(test), len(labels)) for p in probs):
        raise ValueError('Unexpected model output shape')
    combined = np.mean(np.stack(probs, axis=0), axis=0)
    ensemble = metrics(y, (combined >= THRESHOLD).astype(np.uint8))
    margins = {key: ensemble[key] - rf[key] for key in ('precision', 'recall', 'micro_f1')}
    clean_sweep = all(delta >= 0 for delta in margins.values()) and any(delta > 0 for delta in margins.values())
    print('FOLLOWUP RF TEST:', json.dumps(rf), flush=True)
    print('FOLLOWUP ENSEMBLE TEST:', json.dumps(ensemble), flush=True)
    print('FOLLOWUP ENSEMBLE DIFFERENCE:', json.dumps(margins), flush=True)
    print('FOLLOWUP CLEAN SWEEP:', 'YES' if clean_sweep else 'NO', flush=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({
        'classification': 'follow-up analysis of previously viewed test partition; not independent confirmation',
        'dataset': str(args.db), 'threshold': THRESHOLD,
        'model_paths': [str(path) for path in MODEL_PATHS],
        'test_rows': len(test), 'labels': len(labels),
        'baseline': 'leakage-controlled train-only RF reproduction, NOT official baseline',
        'rf_test': rf, 'ensemble_test': ensemble, 'margins': margins, 'clean_sweep_followup': clean_sweep,
        'caution': 'Threshold fixed from validation; prior test access limits evidential independence.'
    }, indent=2), encoding='utf-8')
    print('Saved:', OUTPUT, flush=True)
    print('Existing models and prediction interfaces unchanged.', flush=True)


if __name__ == '__main__':
    main()
