r"""Batch 217-skill predictions from the frozen RoBERTa checkpoint.

Input JSON Lines: one object per line with `title` and optional `description`.
Also accepts `issue text` and `issue description` keys from SkillScope.
Output JSON Lines: one object per input with a 217-element binary `vector` and
ordered `labels` only if --include-labels is passed. Neither DB labels nor API
classification outputs are read during prediction.

Usage (PowerShell):
  .\.venv\Scripts\python.exe nlbse_predict.py --self-test
  .\.venv\Scripts\python.exe nlbse_predict.py --input issues.jsonl --output predictions.jsonl
  .\.venv\Scripts\python.exe nlbse_predict.py --title "Kafka timeout" --description "Retry failures"

The fixed threshold 0.54 was selected on validation and verified once on test.
This script does not establish official competition acceptance.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

THRESHOLD = 0.54
LABEL_COUNT = 217
CHECKPOINT = Path('runs/roberta_recovery/best_model')


def ordered_names(config):
    if config.num_labels != LABEL_COUNT:
        raise ValueError(f'Expected {LABEL_COUNT} checkpoint labels; got {config.num_labels}')
    names = []
    for i in range(LABEL_COUNT):
        name = config.id2label.get(i, config.id2label.get(str(i)))
        if not isinstance(name, str) or not name or name == f'LABEL_{i}':
            raise ValueError(f'Checkpoint missing meaningful label for index {i}')
        names.append(name)
    if len(set(names)) != LABEL_COUNT:
        raise ValueError('Duplicate label names in checkpoint')
    return names


def verify_order(names, db):
    if db is None:
        return
    if not db.is_file():
        raise FileNotFoundError(f'Order-check database missing: {db}')
    with sqlite3.connect(f'file:{db.resolve().as_posix()}?mode=ro', uri=True) as conn:
        cols = [r[1] for r in conn.execute('PRAGMA table_info("nlbse_tool_competition_data_by_issue")')]
    canonical = cols[9:]
    if len(canonical) != LABEL_COUNT or canonical != names:
        mismatch = next(((i, x, y) for i, (x, y) in enumerate(zip(canonical, names)) if x != y), None)
        raise ValueError(f'Checkpoint output order does not match DB target columns: {mismatch}')


def normalize_record(record, i):
    if not isinstance(record, dict):
        raise ValueError(f'Input line {i} must be a JSON object')
    title = record.get('title', record.get('issue text', ''))
    desc = record.get('description', record.get('issue description', ''))
    if title is None:
        title = ''
    if desc is None:
        desc = ''
    if not isinstance(title, str) or not isinstance(desc, str):
        raise ValueError(f'Input line {i}: title/description must be strings')
    return title, desc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', type=Path, default=CHECKPOINT)
    ap.add_argument('--input', type=Path, help='JSON Lines input of issue titles and descriptions')
    ap.add_argument('--output', type=Path, help='JSON Lines output; required with --input')
    ap.add_argument('--title', help='Single-issue input')
    ap.add_argument('--description', default='')
    ap.add_argument('--check-label-order', type=Path, help='Optional read-only DB check of column order')
    ap.add_argument('--include-labels', action='store_true', help='Include ordered label names per output row')
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--self-test', action='store_true', help='Check input/output logic without loading model')
    args = ap.parse_args()
    if args.self_test:
        assert normalize_record({'title':'a','description':'b'}, 1) == ('a','b')
        assert normalize_record({'issue text':'a','issue description':'b'}, 2) == ('a','b')
        assert len([0]*LABEL_COUNT) == 217
        print('SELF-TEST PASS: input schemas and 217-position output contract')
        return
    if bool(args.input) == bool(args.title is not None):
        ap.error('provide exactly one of --input or --title')
    if args.input and not args.output:
        ap.error('--input requires --output')
    if args.batch_size < 1:
        ap.error('--batch-size must be positive')
    if not args.model.is_dir():
        ap.error(f'model checkpoint not found: {args.model}')

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, local_files_only=True)
    names = ordered_names(model.config)
    verify_order(names, args.check_label_order)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()

    if args.input:
        with args.input.open('r', encoding='utf-8-sig') as f:
            records = [(k, json.loads(line)) for k, line in enumerate(f, 1) if line.strip()]
    else:
        records = [(1, {'title': args.title, 'description': args.description})]
    if not records:
        raise ValueError('No input records provided')
    parsed = [normalize_record(rec, i) for i, rec in records]
    results = []
    for start in range(0, len(parsed), args.batch_size):
        batch = parsed[start:start+args.batch_size]
        texts = [title.strip() + ' </s> ' + desc.strip() for title, desc in batch]
        encoded = tok(texts, truncation=True, max_length=256, padding='max_length', return_tensors='pt')
        with torch.inference_mode():
            probs = torch.sigmoid(model(**{k:v.to(device) for k,v in encoded.items()}).logits.float()).cpu()
        if probs.shape != (len(batch), LABEL_COUNT):
            raise RuntimeError(f'Unexpected model shape {tuple(probs.shape)}')
        for j, row in enumerate(probs):
            vector = [int(x) for x in (row >= THRESHOLD).tolist()]
            result = {'row':records[start+j][0], 'vector':vector}
            if args.include_labels:
                result['labels'] = names
            results.append(result)
    if args.output:
        if args.output.resolve() == args.input.resolve():
            raise ValueError('Refusing to overwrite input file')
        if args.output.exists():
            raise FileExistsError(f'Output already exists (refusing overwrite): {args.output}')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as out:
            for result in results:
                out.write(json.dumps(result, ensure_ascii=False) + '\n')
        print(f'Wrote {len(results)} predictions to {args.output}; 217 positions each; threshold={THRESHOLD}', file=sys.stderr)
    else:
        print(json.dumps(results[0], ensure_ascii=False))

if __name__ == '__main__':
    main()
