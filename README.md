# Strait Events

Method-code snapshot for the weekly GDELT--ACLED analysis and MUSE-ED/MUSE-ED++ dynamic event ranking. This repository contains code only, not licensed source data, checkpoints, or published result tables.

## Code map

| Area | Files | Purpose |
| --- | --- | --- |
| Event data | `scripts/build_graph.py`, `scripts/export_map.py` | Normalize ACLED records and build weekly temporal facts. |
| News indicators | `scripts/build_features.py` | Aggregate GDELT reporting into weekly indicators and a seven-entity news network. |
| Coverage and inference | `scripts/analyze_associations.py` | Separate unavailable weeks from observed zeros; run lag, HAC/BH, bootstrap, and auxiliary forecast checks. |
| Dynamic graph | `src/` and `scripts/prepare_context.py` | Train and evaluate MUSE-ED/MUSE-ED++ with lagged context and core-entity weighting. |

## Minimal workflow

Use Python 3.9 and a PyTorch/DGL build matched to your CPU or CUDA version. Place a licensed ACLED extract at `data/raw/acled.csv` and yearly `gdelt_7entities_*.parquet` files in `data/raw/gdelt/`. Run commands from the repository root:

```sh
python scripts/build_graph.py --input data/raw/acled.csv
python scripts/export_map.py --input data/raw/acled.csv
python scripts/build_features.py --gdelt-dir data/raw/gdelt --acled data/raw/acled.csv --granularity week
python scripts/analyze_associations.py --output-dir results/coverage_analysis
python scripts/prepare_context.py
python src/get_history_record.py --dataset ACLED_MUSE_7COUNTRY
python src/main.py --dataset ACLED_MUSE_7COUNTRY --mode pretrain --max-epochs 20
```

For dynamic training, use `src/main.py --mode train` with `--mechanism-feature-path results/model_context/mechanism_context_weekly.csv --mechanism-lag-only 1`. Core weighting uses `--core-protection 1 --core-replay-weight 3`; MUSE-ED++ additionally enables `--use-query-aware-context 1 --historical-contrastive-weight 0.03 --rule-branch-weight 0.03`. The nine lagged context columns are standardized from training slices inside the model loader. Select seeds and hyperparameters explicitly when comparing runs.

The coverage-aware cross-source analysis uses complete common weeks; the historical fact-ranking task retains its original time slices. These are different evaluation units. News indicators measure reporting, and attention weights or lagged associations are not causal effects. The code does not include official third-party baseline repositories.

## License and release

Project-owned code is available under the [MIT License](LICENSE). The graph encoder and convolutional scorer implement established modeling ideas; the research contribution concerns the cross-source analysis and its evaluation, not ownership of those algorithms. Source data and external baseline repositories retain their own terms. This repository remains private for now; project materials will be updated after paper acceptance. Verify provenance and redistribution rights for all remaining model files before changing repository visibility.
