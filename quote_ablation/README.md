# Quote-signal ablation

A post-v2 development experiment. Both configurations were actually retrained, with the same v2 CatBoost parameters, absolute-negative/median weight treatment, seed 42, four threads, temporal folds and inner early-stopping rule. The no-quote variant drops both quote_signal and quote_based_rate; coordinates and market_index are retained. Each variant chooses its own tree count inside training, so this compares pipelines under the same training procedure, not fixed-tree-count models.

| Variant | July MAE | August MAE | September MAE | Mean monthly MAE | Mean monthly RMSE |
|---|---:|---:|---:|---:|---:|
| With quote | 120.68 | 134.44 | 112.04 | 122.39 | 624.73 |
| Without quote | 113.64 | 113.43 | 118.49 | 115.18 | 620.62 |

Removing quote improved equal-weight mean monthly MAE by $7.20 (5.89%). It improved July and August but worsened September by $6.45. Thus quote did not help average development performance in this experiment. This does not prove target leakage or guarantee future superiority.

With-quote rounds: 198 and 593. Without-quote rounds: 778 and 798, close to the 800-round cap. The budget was retained to isolate this ablation under the existing v2 protocol; no cap extension was attempted. One seed only. October and final validation were not evaluated. These same development months have already informed earlier choices; additional model selection increases selection bias.

The with-quote rerun reproduced v2 MAEs within numerical precision. All six monthly MAEs/RMSEs were independently recomputed from saved predictions. V2 final predictions, report and December chart were not changed. The reduced December model already excludes quote, and is not affected by this experiment.

Run after installing requirements.txt:

```sh
python run_ablation.py --data /path/to/train-test.csv --output fresh_results
```

The output must not already contain protocol.json. The original dataset is not bundled. compare_models.py is the unchanged v2 feature/model implementation. Results contain the new protocol, environment versions, metrics and row-level outer predictions.
