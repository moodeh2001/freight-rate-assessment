# Sources and implementation provenance

This experiment was implemented independently for Mohammad Odeh's assessment. No repository source files or predictions were copied into it.

Research references consulted:

- https://github.com/iamgrootns/spotter_assesment : LightGBM with an absolute-error objective and a log target; hypothesis of weight sign errors. Its published performance uses its own validation protocol and is not an expected score for this experiment.
- https://github.com/Piash7224/freigth-rate-prediction : CatBoost and chronological evaluation as another modeling direction.
- https://github.com/Peakeyy-ai/spotter-ml-assessment : comparison of boosting methods; published results are author-reported, not independently reproduced here.
- https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRegressor.html : estimator, callbacks, categorical handling, and parameters.
- https://catboost.ai/docs/en/concepts/python-reference_catboostregressor_fit : chronological eval_set and early stopping API.
- https://catboost.ai/docs/en/concepts/algorithm-missing-values-processing : native missing numeric value handling.

Differences from external repositories:

- Outer development labels are never used for early stopping. The last month INSIDE each training period selects the iteration count, then a fresh model is fitted on the whole outer training period.
- No lookups are learned from validation.csv. The development experiment never opens it; finalization reads it only to transform its rows with training-fitted preprocessing and predict their rates.
- No October score is calculated. October was inspected in v1 and cannot become a fresh holdout again.
- Weight sign correction is a hypothesis compared empirically with treating negatives as missing. Similar distributions do not prove a data-entry error.
- Original target values are retained for all fits and metrics, with no outlier removal or winsorization.
- No blending weights or model settings are copied as claimed optimal. This is a limited, explicitly defined search.
