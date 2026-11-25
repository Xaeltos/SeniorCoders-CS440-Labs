r"""Corrected-v2 NLBSE 217-label inference with a 3-seed focal RoBERTa ensemble.

Runs three saved checkpoints sequentially (GPU-safe), averages sigmoid
probabilities and uses the validation-selected fixed threshold 0.535.
Does not read ground-truth labels or use dataset records for predictions.

Examples (PowerShell, from project root):
  .\.venv\Scripts\python.exe nlbse_predict_ensemble_v2.py --title "Fix connection timeout" --description "Implement retry logic" --check-label-order dataset_v2/skillscope_data_v2.db
  .\.venv\Scripts\python.exe nlbse_predict_ensemble_v2.py --input issues.jsonl --output predictions_ensemble.jsonl
"""
from __future__ import annotations

import argparse
import gc
import json
import sqlite3
import sys
from pathlib import Path

THRESHOLD = 0.535
LABEL_COUNT = 217
MODELS = (
    Path('runs/v2_roberta_focal/best_model'),
    Path('runs/v2_focal_seed_2028/best_model'),
    Path('runs/v2_focal_seed_2029/best_model'),
)


def ordered_names(config):
    if config.num_labels != LABEL_COUNT:
        raise ValueError(f'Expected {LABEL_COUNT} labels; got {config.num_labels}')
    names = [config.id2label.get(i, config.id2label.get(str(i))) for i in range(LABEL_COUNT)]
    if any(not isinstance(x, str) or not x or x == f'LABEL_{i}' for i, x in enumerate(names)):
        raise ValueError('Missing meaningful ordered label names')
    if len(set(names)) != LABEL_COUNT:
        raise ValueError('Duplicate label names')
    return names


def verify_order(names, db):
    if db is None:
        return
    if not db.is_file():
        raise FileNotFoundError(f'Order-check database missing: {db}')
    with sqlite3.connect(f'file:{db.resolve().as_posix()}?mode=ro', uri=True) as conn:
        cols = [row[1] for row in conn.execute('PRAGMA table_info("nlbse_tool_competition_data_by_issue")')]
    if cols[9:] != names:
        raise ValueError('Checkpoint output label ordering differs from corrected v2 database')


def parse_record(record, lineno):
    if not isinstance(record, dict):
        raise ValueError(f'Line {lineno}: input must be JSON object')
    title = record.get('title', record.get('issue text', ''))
    description = record.get('description', record.get('issue description', ''))
    if title is None:
        title = ''
    if description is None:
        description = ''
    if not isinstance(title, str) or not isinstance(description, str):
        raise ValueError(f'Line {lineno}: title/description must be strings')
    return title.strip() + ' </s> ' + description.strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--title', help='Single issue title')
    ap.add_argument('--description', default='', help='Single issue description')
    ap.add_argument('--input', type=Path, help='JSONL with title and description fields')
    ap.add_argument('--output', type=Path, help='JSONL prediction output (required with --input)')
    ap.add_argument('--check-label-order', type=Path, help='Read-only corrected database column check')
    ap.add_argument('--include-labels', action='store_true')
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()

    if args.self_test:
        assert parse_record({'title':'a','description':'b'}, 1) == 'a </s> b'
        assert parse_record({'issue text':'x'}, 1) == 'x </s> '
        assert len(MODELS) == 3 and LABEL_COUNT == 217
        print('SELF-TEST PASS: JSONL schemas, 3 checkpoints, 217-label contract')
        return
    if (args.title is None) == (args.input is None):
        ap.error('Provide exactly one of --title or --input')
    if args.input and not args.output:
        ap.error('Using --input requires --output')
    if args.batch_size < 1:
        ap.error('Batch size must be >= 1')
    if args.output and args.input and args.output.resolve() == args.input.resolve():
        ap.error('Refusing to overwrite input')
    if args.output and args.output.exists():
        ap.error(f'Refusing to overwrite existing output: {args.output}')
    for folder in MODELS:
        if not (folder / 'model.safetensors').is_file():
            ap.error(f'Missing ensemble checkpoint: {folder}')

    if args.input:
        with args.input.open('r', encoding='utf-8-sig') as fh:
            records = [(lineno, json.loads(line)) for lineno, line in enumerate(fh, 1) if line.strip()]
    else:
        records = [(1, {'title': args.title, 'description': args.description})]
    if not records:
        ap.error('No input records')
    texts = [parse_record(record, i) for i, record in records]

    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    summed = np.zeros((len(texts), LABEL_COUNT), dtype=np.float32)
    reference_names = None
    reference_tokenizer = None
    for idx, folder in enumerate(MODELS, start=1):
        print(f'Loading ensemble member {idx}/3: {folder}', file=sys.stderr, flush=True)
        tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(folder, local_files_only=True)
        names = ordered_names(model.config)
        if reference_names is None:
            reference_names = names
            verify_order(names, args.check_label_order)
            reference_tokenizer = tokenizer
        else:
            if names != reference_names:
                raise ValueError(f'Label order mismatch in {folder}')
            if tokenizer.get_vocab() != reference_tokenizer.get_vocab() or tokenizer.special_tokens_map != reference_tokenizer.special_tokens_map:
                raise ValueError(f'Tokenizer mismatch in {folder}')
        model.to(device).eval()
        with torch.inference_mode():
            for start in range(0, len(texts), args.batch_size):
                encoded = tokenizer(texts[start:start + args.batch_size], truncation=True,
                                    max_length=256, padding='max_length', return_tensors='pt')
                logits = model(**{k: v.to(device) for k, v in encoded.items()}).logits
                probabilities = torch.sigmoid(logits.float()).cpu().numpy()
                if probabilities.shape[1] != LABEL_COUNT:
                    raise RuntimeError(f'Incorrect logit width: {probabilities.shape}')
                summed[start:start + len(probabilities)] += probabilities
        model.to('cpu')
        del model, tokenizer
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    averaged = summed / len(MODELS)
    outputs = []
    for (lineno, _), probabilities in zip(records, averaged):
        result = {'row': lineno, 'vector': (probabilities >= THRESHOLD).astype(int).tolist()}
        if args.include_labels:
            result['labels'] = reference_names
        outputs.append(result)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as fh:
            for output in outputs:
                fh.write(json.dumps(output, ensure_ascii=False) + '\n')
        print(f'Wrote {len(outputs)} predictions; {LABEL_COUNT} labels; 3-model average at threshold {THRESHOLD}', file=sys.stderr)
    else:
        print(json.dumps(outputs[0], ensure_ascii=False))


if __name__ == '__main__':
    main()
