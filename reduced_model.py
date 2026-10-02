"""Finalize the development-selected model and evaluate the reduced-input fallback."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from compare_models import Config, Features, CATS, FOLDS, make_model

BASIC_NUMS = ['distance', 'weight', 'day_of_week', 'month', 'day_of_month']


class ReducedFeatures:
    def __init__(self, config):
        self.config = config

    def weights(self, frame):
        w = frame.weight.abs() if self.config.weight.startswith('abs_') else frame.weight.copy()
        return w.mask(w <= 0)

    def fit(self, frame):
        w = self.weights(frame)
        self.medians = w.groupby(frame.equipment).median()
        self.fallback = w.median()
        return self

    def transform(self, frame):
        x = frame[CATS + ['distance', 'weight']].copy()
        w = self.weights(frame)
        x['weight'] = w.fillna(frame.equipment.map(self.medians)).fillna(self.fallback)
        x['day_of_week'] = frame.date.dt.dayofweek
        x['month'] = frame.date.dt.month
        x['day_of_month'] = frame.date.dt.day
        if self.config.family == 'cat':
            x['weight_missing'] = w.isna().astype(int)
            x['weight_was_negative'] = frame.weight.lt(0).astype(int)
            x['weight_was_empty'] = frame.weight.isna().astype(int)
        for c in CATS:
            x[c] = x[c].fillna('__MISSING__').astype(str)
        return x


def reduced_model(config, rounds, threads):
    if config.family == 'cat':
        return make_model(config, rounds, threads)
    return Pipeline([
        ('preprocessor', ColumnTransformer([
            ('categories', OneHotEncoder(handle_unknown='ignore'), CATS),
            ('numbers', 'passthrough', BASIC_NUMS)])),
        ('regressor', RandomForestRegressor(n_estimators=150, max_depth=18,
            min_samples_leaf=5, max_features=0.8, random_state=42, n_jobs=threads))])


def metrics(frame, prediction):
    err = frame.posted_rate.to_numpy() - prediction
    return {'mae': float(np.mean(abs(err))), 'rmse': float(np.sqrt(np.mean(err ** 2)))}


def check_predictions(pred):
    if not np.isfinite(pred).all() or (pred <= 0).any():
        raise ValueError('Predictions must be finite and positive; no silent clipping applied.')

