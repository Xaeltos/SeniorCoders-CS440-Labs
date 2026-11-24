"""Predict NLBSE skill labels from issue text using a trained RoBERTa checkpoint.

No database or ground-truth labels are needed for inference.
Examples:
  python predict_skills.py --example
  python predict_skills.py --title "Fix connection timeout" --description "Retry failed requests"
  python predict_skills.py --title "..." --threshold 0.5 --json-output prediction.json

Threshold 0.30 is the BCE model's validation-selected global threshold.
Predictions are experimental and are not proof of the required skills.
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_MODEL = Path('runs/roberta_bce_pilot/best_model')
EXAMPLE_TITLE = 'Database connection pool times out under heavy load'
EXAMPLE_DESCRIPTION = (
    'The Java service exhausts its PostgreSQL connections after repeated requests. '
    'Investigate connection leaks, add timeout handling, and write regression tests.'
)


def predict(model_dir: Path, title: str, description: str, threshold: float, max_length: int):
    if not model_dir.is_dir():
        raise FileNotFoundError(f'Checkpoint not found: {model_dir}')
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), local_files_only=True)
    if model.config.num_labels != 217:
        raise ValueError(f'Expected 217 outputs; checkpoint has {model.config.num_labels}')
    label_names = []
    for i in range(model.config.num_labels):
        label = model.config.id2label.get(i)
        if not label or label == f'LABEL_{i}':
            raise ValueError('Checkpoint lacks meaningful skill label names; refusing to guess label order')
        label_names.append(label)
    if len(set(label_names)) != 217:
        raise ValueError('Checkpoint contains duplicate skill label names')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    text = title.strip() + ' </s> ' + description.strip()  # training's exact text separator
    inputs = tokenizer(text, truncation=True, max_length=max_length,
                       padding='max_length', return_tensors='pt')
    with torch.inference_mode():
        logits = model(**{k: v.to(device) for k, v in inputs.items()}).logits
        probabilities = torch.sigmoid(logits.float())[0].cpu().tolist()
    ranked = sorted(zip(label_names, probabilities), key=lambda x: x[1], reverse=True)
    predictions = [{'skill': name, 'probability': round(prob, 6)}
                   for name, prob in ranked if prob >= threshold]
    return {'checkpoint': str(model_dir), 'device': str(device),
            'threshold': threshold, 'title': title, 'description': description,
            'label_count': len(label_names), 'predicted_count': len(predictions),
            'predictions': predictions,
            'top_10_by_probability': [{'skill': name, 'probability': round(prob, 6)}
                                      for name, prob in ranked[:10]]}


def main():
    parser = argparse.ArgumentParser(description='NLBSE 217-skill inference from GitHub issue text')
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--title', type=str)
    parser.add_argument('--description', type=str, default='')
    parser.add_argument('--example', action='store_true', help='Use a built-in fictional issue')
    parser.add_argument('--threshold', type=float, default=0.30)
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--json-output', type=Path)
    args = parser.parse_args()
    if not 0 < args.threshold <= 1:
        parser.error('--threshold must be > 0 and <= 1')
    if args.max_length < 16:
        parser.error('--max-length must be at least 16')
    if args.example and args.title:
        parser.error('use either --example or --title, not both')
    if not args.example and not args.title:
        parser.error('provide --title or --example')
    title = EXAMPLE_TITLE if args.example else args.title
    description = EXAMPLE_DESCRIPTION if args.example else args.description
    result = predict(args.model, title, description, args.threshold, args.max_length)
    print(f"Device: {result['device']} | Output labels: {result['label_count']} | Threshold: {result['threshold']:.2f}")
    print(f"Issue: {result['title']}")
    print(f"Predicted skills: {result['predicted_count']}")
    if result['predictions']:
        for item in result['predictions']:
            print(f"  {item['probability']:.3f}  {item['skill']}")
    else:
        print('  (none above threshold)')
    print('Top 10 probabilities (diagnostic, whether or not above threshold):')
    for item in result['top_10_by_probability']:
        print(f"  {item['probability']:.3f}  {item['skill']}")
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(f"Saved {args.json_output}")


if __name__ == '__main__':
    main()
