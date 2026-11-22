
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

DB_PATH = Path("skillscope_data.db")
TABLE = "nlbse_tool_competition_data_by_issue"
SEED = 42

print("Loading competition dataset...")

if not DB_PATH.is_file():
    raise FileNotFoundError(f"Database not found: {DB_PATH}")

with sqlite3.connect(DB_PATH) as connection:
    data = pd.read_sql_query(f'SELECT * FROM "{TABLE}"', connection)

# First nine columns are metadata/input fields.
# Remaining columns are the 217 skill labels.
label_columns = list(data.columns[9:])

assert len(label_columns) == 217, "Unexpected label count"

# Combine issue text and description.
texts = (
    data["issue text"].fillna("").astype(str)
    + " "
    + data["issue description"].fillna("").astype(str)
)

# API-match counts become binary skill labels.
labels = (
    data[label_columns]
    .apply(pd.to_numeric, errors="coerce")
    .fillna(0)
    .gt(0)
    .astype(np.uint8)
    .to_numpy()
)

print(f"Examples: {len(data)}")
print(f"Labels: {len(label_columns)}")

# Split BEFORE TF-IDF fitting to prevent leakage.
X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.20,
    random_state=SEED
)

print("Extracting TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=1000,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("Training Random Forest...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=SEED,
    n_jobs=2
)

model.fit(X_train_tfidf, y_train)

print("Evaluating predictions...")

predictions = model.predict(X_test_tfidf)

precision, recall, f1, _ = precision_recall_fscore_support(
    y_test,
    predictions,
    average="micro",
    zero_division=0
)

results = {
    "model": "Random Forest + TF-IDF",
    "method": "Leakage-reduced baseline without MLSMOTE",
    "precision": float(precision),
    "recall": float(recall),
    "micro_f1": float(f1),
    "train_rows": len(X_train),
    "test_rows": len(X_test),
    "labels": len(label_columns),
    "random_seed": SEED,
    "tfidf_features": 1000,
    "oversampling": False
}

Path("baseline_metrics.json").write_text(
    json.dumps(results, indent=4),
    encoding="utf-8"
)

print("\nBASELINE RESULTS")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"Micro-F1:  {f1:.4f}")
print("\nSaved to baseline_metrics.json")
