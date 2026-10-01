# Business Entity Resolution — reproducible submission

This repository resolves Source-2 and Source-3 business records against the
deduplicated Source-1 reference records.  It uses only the supplied TSV files;
no external business database, geocoding API, or web lookup is used.

## Environment

Tested on Windows with Python 3.14.7, 8 GB RAM, and DuckDB 1.5.5.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The commands below use `<data_dir>` for the directory that directly contains
`train/` and `test/`.  For the supplied archive it is:

```text
dataset\6ab10eb3b23ba_student_resource\student_resource\dataset
```

## End-to-end reproduction

Run every command from the repository root.  The DuckDB steps are disk-backed;
do not use the `--row-limit` smoke-test option for a real submission.

```powershell
# 1. Read TSVs and build normalized, disk-backed tables.
python src\build_database.py --data-dir <data_dir>
python src\prepare_training_labels.py --ground-truth <data_dir>\train\train_ground_truth.tsv

# 2. Build training blocking index and create an untouched hash-fold validation set.
python src\build_target_blocks.py --split train
python src\generate_candidates.py --source-split train --target-split train --output-table validation_candidates_fuzzy --validation-fold 0 --max-block-size 100 --max-candidates 50
python src\evaluate_blocking.py --candidate-table validation_candidates_fuzzy --validation-fold 0

# 3. Generate train-only candidates (fold 0 remains untouched), features, and model.
python src\generate_candidates.py --source-split train --target-split train --output-table training_candidates_fuzzy --exclude-validation-fold 0 --max-block-size 100 --max-candidates 50
python src\build_training_features.py --candidate-table training_candidates_fuzzy --output-table model_training_features --positive-modulus 6 --negative-modulus 40
python src\train_model.py --feature-table model_training_features --model-path models\entity_matcher.joblib

# 4. Score validation candidates and select a precision-oriented threshold.
python src\score_candidates.py --candidate-table validation_candidates_fuzzy --source-split train --target-split train --output work\validation_scores.parquet
python src\evaluate_predictions.py --scores work\validation_scores.parquet --validation-fold 0 --thresholds 0.86,0.87,0.88,0.89,0.90,0.91,0.92,0.93,0.94

# 5. Build final test candidates and score each one.
python src\build_target_blocks.py --split test
python src\generate_candidates.py --source-split test --target-split test --output-table test_candidates_fuzzy --max-block-size 100 --max-candidates 50
python src\score_candidates.py --candidate-table test_candidates_fuzzy --source-split test --target-split test --output work\test_scores.parquet

# 6. Export both required TSVs.  The partitioned exporter is required on 8 GB RAM.
python src\export_submission_partitioned.py --candidate-table test_candidates_fuzzy --scores work\test_scores.parquet --threshold 0.910 --output-dir output
```

The final files are:

```text
output\matching_results.tsv
output\candidate_pairs.tsv
```

## Optional local dashboard

The Streamlit dashboard lets you inspect a selected Source-1 test record, its
final matches, model probabilities, candidate rank, blocking evidence, and
name/address similarity. It reads only local DuckDB and Parquet artifacts; it
does not call an external service.

```powershell
streamlit run src\app.py
```

Open the local URL printed in the terminal (normally `http://localhost:8501`).

## Design

`build_database.py` normalizes lower-cased business names and addresses,
including punctuation removal, legal-suffix-free name variants, compact forms,
postal-code extraction, and leading-address-number extraction.

Blocking combines exact normalized name/address keys, legal-suffix-free names,
postal/name anchors, house-number/name anchors, rare name and address tokens,
name prefixes/suffixes, and four-character name grams.  Keys whose target block
has more than 100 records are ignored and candidates are ranked by combined
blocking evidence; the final model receives at most 50 candidates per Source-1
record.

The matcher is a `HistGradientBoostingClassifier` trained on source-pair
features: Jaro-Winkler and normalized edit similarities for names/addresses,
length ratios, exact normalized fields, first-name-token equality, postal and
leading-number equality, source indicator, and the blocking score.  The
threshold is selected strictly on Source-1 hash fold 0 by macro F_0.5.

## Resource notes

The supplied data has about 19 million rows.  Do not load it all into pandas.
DuckDB uses a 4 GB limit for table/index construction; scoring is batched; final
TSV aggregation is partitioned into 128 on-disk Source-1 buckets to fit 8 GB
RAM.  The full run needs substantial free disk capacity for DuckDB and temporary
Parquet partitions.
