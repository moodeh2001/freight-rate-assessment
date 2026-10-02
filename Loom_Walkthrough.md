# Loom walkthrough - v3, approximately 2 to 3 minutes

Open the v3 report, train.py, chart and successful scorer output. Rehearse in your own words.

## 0:00-0:30 - Objective and validation

Hi, I'm Mohammad Odeh. This project predicts freight shipment rates in dollars. I used 48,000 labeled shipments and produced predictions for 12,000 November-December shipments.

I used chronological development folds. Early stopping happens on a month inside the training window; then a fresh model is fitted on the whole training period before evaluating future months. October had been inspected in an earlier version, so I do not present it as a fresh test.

## 0:30-1:00 - Cleaning and model choice

I compared Random Forest, LightGBM and CatBoost. I also tested how to handle negative and missing weights. Absolute negative weights plus training equipment-median imputation performed best in the initial comparison. This does not prove the negatives were entry errors. Flags retain whether values were originally negative or missing.

The final model is CatBoost with an absolute-error objective, native categorical inputs, and 798 trees. It uses route, equipment, distance, weight, coordinates, market index and calendar features.

## 1:00-1:35 - Why quote was removed

After the initial model selection, I ran a specific ablation: remove both quote_signal and distance times quote_signal while retaining the other features and the same training procedure.

Mean monthly development MAE improved from 122 dollars and 39 cents to 115 dollars and 18 cents, about 5.9 percent. July and August improved, while September worsened. This supported selecting the model without quote. It does not prove data leakage or guarantee better hidden performance. This additional experiment reused development months and therefore adds selection bias.

I froze the selected settings and refitted on all January-October labels.

## 1:35-2:05 - December chart

The December chart still needs a separate model because its inputs also lack coordinates and market index. I kept the validated reduced CatBoost model unchanged. Only the date varies across these fixed-lane predictions.

The chart reflects learned calendar effects, not observed December seasonality. Its vertical axis is truncated. Rare high-priced shipments and unseen future months remain limitations.

## 2:05-2:30 - Reproduction and delivery

The supplied scorer accepted all 12,000 final predictions and all 31 December rows. It verifies file validity and creates the chart; Spotter calculates hidden accuracy later.

The package includes code, dependencies, development evidence and the report. The README gives the exact reproduction commands. The final training command preserves packaged outputs by writing to a new folder and verifies that its no-quote features match the evaluated ablation.

Use your own explanation and describe development assistance accurately if asked.
