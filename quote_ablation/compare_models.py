"""Bounded development experiment. Never uses October labels or final validation data.

Run from the original project: python compare_models.py --data-dir data
Outputs go to experiment_results; existing submission files are not touched.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

CATS = ["pickup", "delivery", "equipment"]
NUMS = ["distance", "weight", "day_of_week", "month", "day_of_month",
        "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
        "market_index", "quote_signal", "weight_missing", "market_index_missing",
        "quote_based_rate"]
# Outer evaluation labels never select the boosting iteration count.
FOLDS = [
    {"name": "july_august", "inner_cut": "2025-06-01", "train_end": "2025-07-01",
     "eval_end": "2025-09-01"},
    {"name": "september", "inner_cut": "2025-08-01", "train_end": "2025-09-01",
     "eval_end": "2025-10-01"},
]


@dataclass(frozen=True)
class Config:
    family: str
    target: str = "raw"
    weight: str = "invalid_median"

    @property
    def name(self):
        return f"{self.family}_{self.target}_{self.weight}"


class Features:
    """Fit medians and category levels only on the supplied training rows."""
    def __init__(self, config):
        self.config = config

    def weight_values(self, frame):
        w = frame["weight"].copy()
        if self.config.weight.startswith("abs_"):
            w = w.abs()
        return w.mask(w <= 0)

    def fit(self, frame):
        w = self.weight_values(frame)
        self.weight_medians = w.groupby(frame["equipment"]).median()
        self.weight_fallback = w.median()
        self.market_median = frame["market_index"].median()
        self.categories = {c: sorted(frame[c].dropna().astype(str).unique()) for c in CATS}
        return self

    def transform(self, frame):
        x = frame[CATS + ["distance", "weight", "pickup_lat", "pickup_lon",
                              "delivery_lat", "delivery_lon", "market_index", "quote_signal"]].copy()
        w = self.weight_values(frame)
        x["weight_missing"] = w.isna().astype(int)
        if self.config.weight.endswith("median"):
            w = w.fillna(frame["equipment"].map(self.weight_medians)).fillna(self.weight_fallback)
        x["weight"] = w
        # Keep market handling fixed to isolate the weight experiment.
        x["market_index_missing"] = x["market_index"].isna().astype(int)
        x["market_index"] = x["market_index"].fillna(self.market_median)
        x["day_of_week"] = frame["date"].dt.dayofweek
        x["month"] = frame["date"].dt.month
        x["day_of_month"] = frame["date"].dt.day
        x["quote_based_rate"] = frame["distance"] * frame["quote_signal"]
        if self.config.family != "rf":
            # Preserved in every boosting variant, including invalid_median.
            x["weight_was_negative"] = frame["weight"].lt(0).astype(int)
            x["weight_was_empty"] = frame["weight"].isna().astype(int)
        for c in CATS:
            if self.config.family == "lgb":
                x[c] = pd.Categorical(x[c], categories=self.categories[c])
            else:
                x[c] = x[c].fillna("__MISSING__").astype(str)
        return x


def make_model(config, rounds, threads):
    if config.family == "lgb":
        return lgb.LGBMRegressor(
            objective="regression_l1", metric="None", n_estimators=rounds,
            learning_rate=0.03, num_leaves=31, min_child_samples=25,
            subsample=0.85, subsample_freq=1, colsample_bytree=0.85,
            reg_lambda=1.0, random_state=42, n_jobs=threads,
            deterministic=True, force_col_wise=True, verbosity=-1,
        )
    if config.family == "cat":
        return CatBoostRegressor(
            iterations=rounds, depth=6, learning_rate=0.04, loss_function="MAE",
            eval_metric="MAE", l2_leaf_reg=5, random_seed=42,
            thread_count=threads, allow_writing_files=False, verbose=False,
            nan_mode="Min", cat_features=CATS,
        )
    return Pipeline([
        ("preprocessor", ColumnTransformer([
            ("categories", OneHotEncoder(handle_unknown="ignore"), CATS),
            ("numbers", "passthrough", NUMS),
        ])),
        ("regressor", RandomForestRegressor(n_estimators=150, max_depth=18,
            min_samples_leaf=5, max_features=0.8, random_state=42, n_jobs=threads)),
    ])


def encode_target(y, config):
    return np.log1p(y) if config.target == "log" else y


def dollars(prediction, config):
    return np.expm1(prediction) if config.target == "log" else np.asarray(prediction)


def fit_nested(frame, config, fold, threads):
    train = frame[frame.date < fold["train_end"]].copy()
    if config.family == "rf":
        rounds = 150
    else:
        inner_train = train[train.date < fold["inner_cut"]]
        inner_stop = train[train.date >= fold["inner_cut"]]
        assert inner_train.date.max() < inner_stop.date.min()
        prep = Features(config).fit(inner_train)
        model = make_model(config, 1200 if config.family == "lgb" else 800, threads)
        fit_x, stop_x = prep.transform(inner_train), prep.transform(inner_stop)
        fit_y, stop_y = encode_target(inner_train.posted_rate, config), encode_target(inner_stop.posted_rate, config)
        if config.family == "lgb":
            # Even when fitting log targets, early stopping measures dollar MAE.
            def dollar_mae(y, prediction):
                return "dollar_mae", float(np.mean(np.abs(dollars(y, config) - dollars(prediction, config)))), False
            model.fit(fit_x, fit_y, eval_set=[(stop_x, stop_y)], eval_metric=dollar_mae,
                      callbacks=[lgb.early_stopping(60, first_metric_only=True, verbose=False)])
            rounds = int(model.best_iteration_)
        else:
            # CatBoost candidate is raw-target only, so its MAE is in dollars.
            model.fit(fit_x, fit_y, eval_set=(stop_x, stop_y), early_stopping_rounds=60)
            rounds = int(model.get_best_iteration()) + 1
        if rounds < 1:
            raise RuntimeError("No valid best iteration")
    # Fresh preprocessing and model on the complete outer training window.
    prep = Features(config).fit(train)
    model = make_model(config, rounds, threads)
    model.fit(prep.transform(train), encode_target(train.posted_rate, config))
    return prep, model, rounds


def audit_weights(frame, output):
    # Summaries by equipment support investigation, not proof of sign corruption.
    rows = []
    for equip, group in frame.groupby("equipment"):
        positive = group.loc[group.weight > 0, "weight"]
        negative = group.loc[group.weight < 0, "weight"].abs()
        for label, values in [("positive", positive), ("absolute_negative", negative)]:
            q = values.quantile([0, .25, .5, .75, 1])
            rows.append({"equipment": equip, "population": label, "n": len(values),
                         **{f"q{int(k*100)}": float(v) for k,v in q.items()}})
    pd.DataFrame(rows).to_csv(output / "weight_audit_training_only.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiment_results"))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    source = args.data_dir / "train-test.csv"
    raw = pd.read_csv(source)
    raw["date"] = pd.to_datetime(raw["date"], errors="raise")
    # October and November-December play no part in this experiment.
    frame = raw.loc[raw.date < "2025-10-01"].copy().sort_values(["date", "load_id"])
    del raw
    if not frame.load_id.is_unique or frame.posted_rate.isna().any():
        raise ValueError("Missing targets or duplicate load IDs")
    if (frame.posted_rate <= 0).any() or (frame.distance <= 0).any():
        raise ValueError("Nonpositive target/distance requires a separate investigation")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if (output / "protocol.json").exists():
        raise SystemExit("Output directory already contains a run. Choose a new --output-dir to preserve it.")
    protocol = {
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "versions": {k: importlib.metadata.version(k) for k in ["numpy","pandas","scikit-learn","lightgbm","catboost"]},
        "folds": FOLDS, "seed": 42, "threads": args.threads,
        "model_stage": ["rf/raw", "lgb/raw", "lgb/log", "cat/raw"],
        "initial_weight_policy": "invalid_median",
        "weight_stage": ["invalid_median","abs_median","invalid_native","abs_native"],
        "ranking": "Unweighted mean dollar MAE over July, August, September; RMSE breaks ties.",
        "selection_note": "Select best boosting family/target first, then compare four weight policies for that candidate only. Finally compare its winner with original RF. Not an exhaustive search.",
        "october": "Excluded from training and evaluation; previously inspected in v1, not a fresh holdout.",
        "targets": "No clipping, deletion or correction of target prices.",
        "budget": "LGB max 1200 rounds; CatBoost max 800; patience 60; no outer-label early stopping.",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    audit_weights(frame[frame.date < "2025-09-01"], output)
    all_rows = []
    configs = {}

    def run(config):
        configs[config.name] = asdict(config)
        for fold in FOLDS:
            start = time.monotonic()
            print(f"START {config.name} | {fold['name']}", flush=True)
            prep, model, rounds = fit_nested(frame, config, fold, args.threads)
            outer = frame[(frame.date >= fold["train_end"]) & (frame.date < fold["eval_end"])].copy()
            pred = dollars(model.predict(prep.transform(outer)), config)
            if not np.isfinite(pred).all() or (pred <= 0).any():
                raise ValueError("Nonfinite/nonpositive model predictions")
            preds = outer[["load_id","date","posted_rate"]].copy()
            preds["predicted_rate"] = pred
            preds.to_csv(output / f"oof_{config.name}_{fold['name']}.csv", index=False)
            for month, group in preds.groupby(preds.date.dt.strftime("%Y-%m")):
                row = {"candidate": config.name, "month": month, "rows": len(group),
                       "mae": mean_absolute_error(group.posted_rate, group.predicted_rate),
                       "rmse": mean_squared_error(group.posted_rate, group.predicted_rate) ** .5,
                       "rounds": rounds, "fold_seconds": round(time.monotonic()-start,2)}
                all_rows.append(row)
                print(f"  {month}: MAE={row['mae']:.2f} RMSE={row['rmse']:.2f} rounds={rounds}", flush=True)
            pd.DataFrame(all_rows).to_csv(output / "monthly_metrics.csv", index=False)

    def rank(names=None):
        result = pd.DataFrame(all_rows).groupby("candidate").agg(
            mean_monthly_mae=("mae","mean"), mean_monthly_rmse=("rmse","mean"),
            worst_month_mae=("mae","max"), months=("month","nunique"))
        if names is not None:
            result = result.loc[names]
        assert result.months.eq(3).all()
        return result.sort_values(["mean_monthly_mae","mean_monthly_rmse"])

    stage_a = [Config("rf"), Config("lgb"), Config("lgb", "log"), Config("cat")]
    for config in stage_a:
        run(config)
    stage_a_ranking = rank()
    stage_a_ranking.to_csv(output / "model_ranking.csv")
    boost_name = rank([c.name for c in stage_a if c.family != "rf"]).index[0]
    boost = Config(**configs[boost_name])
    print(f"Best boosting configuration: {boost_name}; testing weight policies.", flush=True)
    weight_configs = [boost]
    for policy in ["abs_median", "invalid_native", "abs_native"]:
        candidate = replace(boost, weight=policy)
        weight_configs.append(candidate)
        run(candidate)
    weight_ranking = rank([c.name for c in weight_configs])
    weight_ranking.to_csv(output / "weight_ranking.csv")
    final_ranking = rank()
    final_ranking.to_csv(output / "all_candidates_ranking.csv")
    winner = final_ranking.index[0]
    selection = {"candidate": winner, "config": configs[winner],
                 "selection_mean_monthly_mae": float(final_ranking.iloc[0].mean_monthly_mae),
                 "note": "Development-selected winner of a small search; not an unbiased future-performance estimate.",
                 "final_submission_files_changed": False}
    (output / "selection.json").write_text(json.dumps(selection, indent=2))
    print("\nFINAL DEVELOPMENT RANKING\n" + final_ranking.to_string(), flush=True)
    print("\nNo October score or final predictions generated. Original submission is unchanged.", flush=True)


if __name__ == "__main__":
    main()
