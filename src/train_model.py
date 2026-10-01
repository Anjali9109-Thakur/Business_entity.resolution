"""Train a compact gradient-boosted matcher from numeric pair features."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split


IDENTIFIER_COLUMNS = {"source1_entity_id", "candidate_entity_id", "label"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default="work/entity_resolution.duckdb", type=Path)
    parser.add_argument("--feature-table", default="model_training_features")
    parser.add_argument("--model-path", default="models/entity_matcher.joblib", type=Path)
    args = parser.parse_args()
    if not args.feature_table.replace("_", "").isalnum():
        raise SystemExit("--feature-table may contain only letters, digits, and underscores.")

    connection = duckdb.connect(str(args.database.resolve()), read_only=True)
    table_info = connection.execute(f"PRAGMA table_info('{args.feature_table}')").fetchall()
    feature_names = [row[1] for row in table_info if row[1] not in IDENTIFIER_COLUMNS]
    if not feature_names:
        raise SystemExit(f"No features found in {args.feature_table}")

    cursor = connection.execute(
        f"SELECT {', '.join(feature_names)}, label FROM {args.feature_table}"
    )
    feature_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    reader = cursor.fetch_record_batch(rows_per_batch=100_000)
    for batch in reader:
        columns = batch.to_pydict()
        feature_chunks.append(
            np.column_stack([
                np.asarray(columns[name], dtype=np.float32) for name in feature_names
            ])
        )
        label_chunks.append(np.asarray(columns["label"], dtype=np.int8))
    connection.close()
    if not feature_chunks:
        raise SystemExit("Feature table is empty.")

    features = np.vstack(feature_chunks)
    labels = np.concatenate(label_chunks)
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    print(f"Loaded {len(labels):,} rows: positives={positives:,}, negatives={negatives:,}")
    if not positives or not negatives:
        raise SystemExit("Training requires both positive and negative examples.")

    x_train, x_holdout, y_train, y_holdout = train_test_split(
        features, labels, test_size=0.15, stratify=labels, random_state=42
    )
    positive_weight = (y_train == 0).sum() / (y_train == 1).sum()
    weights = np.where(y_train == 1, positive_weight, 1.0)
    model = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=250,
        max_leaf_nodes=31,
        l2_regularization=2.0,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )
    model.fit(x_train, y_train, sample_weight=weights)
    probabilities = model.predict_proba(x_holdout)[:, 1]
    print(f"Internal holdout ROC-AUC: {roc_auc_score(y_holdout, probabilities):.5f}")
    print(f"Internal holdout average precision: {average_precision_score(y_holdout, probabilities):.5f}")

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "feature_names": feature_names}, args.model_path
    )
    print(f"Saved model to {args.model_path}")


if __name__ == "__main__":
    main()
