# Freight Rate Assessment - v3 (no quote signal)

Author: Mohammad Ahmad Odeh

**This is the updated submission.** The main model is CatBoost on dollar targets with MAE loss, 798 trees, depth 6, learning rate 0.04, L2 regularization 5 and seed 42. Neither `quote_signal` nor `quote_based_rate` is used by the final main model. Coordinates and market index are retained. The December model is unchanged from v2: reduced-input CatBoost with 790 trees.

## What changed

After v2, an additional development ablation reran the main model with and without both quote features under the same training procedure. Removing both improved equal-weight mean monthly development MAE from **$122.39 to $115.18 (5.89%)**. It improved July and August but worsened September. This is development-selected evidence, not an independent test or proof of leakage.

| Main model | July MAE | August MAE | September MAE | Mean MAE | Mean RMSE |
|---|---:|---:|---:|---:|---:|
| v2 with quote | 120.68 | 134.44 | 112.04 | 122.39 | 624.73 |
| **v3 without quote** | **113.64** | **113.43** | **118.49** | **115.18** | **620.62** |

Each mean is an equal-weight mean of monthly metrics. In particular, mean monthly RMSE is not the RMSE pooled across all rows. The with-quote rerun reproduced v2. Both variants use the same seed, preprocessing, features except the two quote columns, and stopping rule; iteration counts are selected separately inside training.

## Files

- `train.py`: final refit of both frozen configurations and output generation. Contains `NoQuoteFeatures`.
- `reduced_model.py`: unchanged reduced feature/model helpers extracted from v2.
- `compare_models.py`: original model-family/weight-policy experiment and shared model factory.
- `quote_ablation/`: ablation script, unchanged original feature implementation, protocol, metrics, outer predictions and historical notes.
- `experiment_results/`: original v2 family/weight selection evidence, preserved as history.
- `v2_evidence/`: original reduced-model evaluation and v2 final-run provenance, preserved as history.
- `final_run.json`: **current v3 run**, feature list, versions and hashes.
- `validation_predictions.csv`: **current v3** 12,000 predictions, aligned to template IDs.
- `december_predictions.csv`: 31 predictions from the unchanged reduced model.
- `score.py`: unchanged organizer validator; `scorer_results/` contains the resulting chart.
- `Freight_Rate_Assessment_Report.pdf`, `Loom_Walkthrough.md`, `SOURCES.md`.
- `MANIFEST.sha256`: hashes of packaged files, excluding the manifest itself.

Historical files inside `experiment_results/`, `v2_evidence/`, and `quote_ablation/` describe their own runs. Their older selected model and statements that v2 was unchanged at that stage are not the current final selection; use this README and `final_run.json` for v3.

## Reproduce final predictions

Executed with Python 3.12.14 on Linux and four CPU threads. `requirements.txt` pins direct dependencies; it is not a full transitive lock. Platform/version changes can affect numerical reproducibility. Source datasets are not bundled.

Create `data/` beside `train.py` and place the original files there:

- `train-test.csv`
- `validation.csv`
- `validation-predictions-template.csv`
- `december-chart-inputs.csv`

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe score.py --predictions rerun/validation_predictions.csv --december-predictions rerun/december_predictions.csv
```

Linux/macOS: use `python3.12 -m venv .venv` and `.venv/bin/python` in the equivalent commands. Activation is optional.

`train.py` writes to `rerun/` by default, preserving packaged predictions. It refuses existing prediction outputs unless `--overwrite` is passed. Use `--output-dir other_run` for a new destination. It reads the frozen evidence, verifies the training hash, checks that the no-quote feature transformation matches the ablation exactly, then refits all 48,000 labeled rows. It does not rerun development selection or serialize models.

To check the packaged predictions without training:

```sh
python score.py --predictions validation_predictions.csv --december-predictions december_predictions.csv
```

To reproduce the quote experiment independently:

```sh
python quote_ablation/run_ablation.py --data data/train-test.csv --output quote_rerun
```

To reproduce the earlier family/weight search:

```sh
python compare_models.py --data-dir data --output-dir experiment_rerun --threads 4
```

These experiments do not automatically alter the frozen final selection. Both refuse to overwrite existing protocol directories.

## Temporal protocol

- Earlier fold: train January-May and stop on June; refit January-June; evaluate July and August separately with the same model.
- Later fold: train January-July and stop on August; refit January-August; evaluate September.
- Inner early stopping: maximum 800 trees, patience 60, dollar MAE.
- Without-quote selected counts: 778 and 798. Final refit uses the last fold's 798, not an iteration selected from October.
- October was inspected in v1. It is excluded from v2/v3 development and joins training only after the settings are frozen. No new October or hidden validation score is claimed.

The ablation followed inspection of v2 results and uses the same development months, increasing selection bias. There is only one seed, and selected iterations are near the cap. No 3,000-tree extension, multi-seed study, or new independent test was conducted. XGBoost and LSTM were not run.

## Features and preprocessing

Final main features: pickup, delivery, equipment, distance, weight, pickup/delivery latitude and longitude, market index, weekday, month, day-of-month, weight-missing flag, market-missing flag, originally-negative weight flag and originally-empty weight flag. IDs and target are excluded; quote columns are not required by `NoQuoteFeatures.transform`.

Negative weights are converted to absolute magnitudes, zero is missing, and missing weights use equipment-group medians with an overall median fallback. Market index uses its training median. All fitted statistics come exclusively from each training window during evaluation. CatBoost handles categories natively. Weight-policy comparisons tested native numeric NaN handling as well. The selected policy is an empirical choice, not proof of sign-entry errors; no clustering was performed. Target prices were never clipped, corrected or removed.

## December model

The fixed chart schema lacks coordinates and market fields, so it still requires a separate reduced model even after removing quote from the main model. Reduced inputs: pickup, delivery, equipment, distance, weight, calendar features and weight flags.

Its July/August/September MAEs are $136.74/$193.85/$108.12; mean $146.24 versus $178.13 for the reduced RF. This model's hyperparameters and 790-tree final fit are unchanged. The final script refits it for complete reproducibility; generated predictions were checked against v2.

## Verification and limits

The organizer scorer validates 12,000 expected unique IDs and 31 fixed December rows, finite positive rates, required columns and dates. It produces the chart; **Spotter computes hidden accuracy after submission**. Additional checks confirm template order, preservation of December fields, equivalence with the ablation feature transformation and unchanged December predictions.

Removing quote improves average development MAE but worsens September. It neither proves quote leakage nor guarantees future performance. Large errors remain; mean monthly RMSE remains much higher than MAE. No uncertainty intervals are supplied. December's calendar pattern is a prediction rather than observed seasonality, the chart axis is truncated, and no lane-specific December accuracy is established. Trees cannot reliably extrapolate an unseen-month trend.

Upload this folder's contents to an accessible GitHub repository, record the 2-3 minute walkthrough, and submit the required artifacts/links. Publishing and recording are not performed by this package. Explain the implementation and development assistance accurately.
