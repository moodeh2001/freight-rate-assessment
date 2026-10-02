"""Refit the frozen v3 no-quote main model and unchanged December model."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import platform
import numpy as np
import pandas as pd
from compare_models import Config, Features, CATS, make_model
from reduced_model import ReducedFeatures, check_predictions


class NoQuoteFeatures(Features):
    """Same features as the ablation, without requiring either quote column."""
    def transform(self, frame):
        x = frame[CATS + ['distance', 'weight', 'pickup_lat', 'pickup_lon',
                          'delivery_lat', 'delivery_lon', 'market_index']].copy()
        w = self.weight_values(frame)
        x['weight_missing'] = w.isna().astype(int)
        x['weight'] = w.fillna(frame.equipment.map(self.weight_medians)).fillna(self.weight_fallback)
        x['market_index_missing'] = x.market_index.isna().astype(int)
        x['market_index'] = x.market_index.fillna(self.market_median)
        x['day_of_week'] = frame.date.dt.dayofweek
        x['month'] = frame.date.dt.month
        x['day_of_month'] = frame.date.dt.day
        x['weight_was_negative'] = frame.weight.lt(0).astype(int)
        x['weight_was_empty'] = frame.weight.isna().astype(int)
        for c in CATS:
            x[c] = x[c].fillna('__MISSING__').astype(str)
        return x


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output-dir', type=Path, default=Path('rerun'))
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--overwrite', action='store_true')
    a = parser.parse_args()
    if a.threads < 1:
        parser.error('--threads must be positive')
    base = Path(__file__).resolve().parent
    paths = [a.output_dir / n for n in ['validation_predictions.csv', 'december_predictions.csv']]
    if not a.overwrite and any(p.exists() for p in paths):
        parser.error('Prediction files already exist; choose a new output directory or --overwrite.')
    a.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((base/'quote_ablation/results/protocol.json').read_text())
    source = a.data_dir/'train-test.csv'
    if sha(source) != protocol['input_sha256']:
        raise ValueError('Training input differs from the reviewed ablation.')
    metrics = pd.read_csv(base/'quote_ablation/results/metrics.csv')
    rounds = int(metrics.loc[(metrics.variant == 'without_quote') &
                            (metrics.month == '2025-09'), 'rounds'].item())
    reduced = json.loads((base/'v2_evidence/reduced_selection.json').read_text())
    if rounds != 798 or reduced['rounds'] != 790 or reduced['candidate'] != 'cat_raw_abs_median':
        raise ValueError('Frozen selection evidence changed; review before refitting.')
    config = Config('cat', 'raw', 'abs_median')
    frame = pd.read_csv(source, parse_dates=['date']).sort_values(['date', 'load_id'])
    if len(frame) != 48000 or not frame.load_id.is_unique or frame.posted_rate.isna().any():
        raise ValueError('Unexpected training data.')
    prep = NoQuoteFeatures(config).fit(frame)
    x = prep.transform(frame)
    # Regression check against the exact transformation evaluated in the ablation.
    reference = Features(config).fit(frame).transform(frame).drop(columns=['quote_signal', 'quote_based_rate'])
    pd.testing.assert_frame_equal(x, reference)
    pd.testing.assert_frame_equal(x, prep.transform(frame.drop(columns='quote_signal')))
    print(f'Final main-model fit: {len(frame):,} rows, {rounds} trees, no quote inputs.', flush=True)
    model = make_model(config, rounds, a.threads)
    model.fit(x, frame.posted_rate)
    valid = pd.read_csv(a.data_dir/'validation.csv', parse_dates=['date'])
    template = pd.read_csv(a.data_dir/'validation-predictions-template.csv')
    if len(valid) != 12000 or not valid.load_id.is_unique or not template.load_id.is_unique or set(valid.load_id) != set(template.load_id):
        raise ValueError('Validation/template ID mismatch.')
    pred = np.asarray(model.predict(prep.transform(valid)))
    check_predictions(pred)
    template['predicted_rate'] = template.load_id.map(pd.Series(pred, index=valid.load_id))
    template[['load_id', 'predicted_rate']].to_csv(paths[0], index=False)
    print('Refitting unchanged December configuration: 790 trees.', flush=True)
    dec_prep = ReducedFeatures(config).fit(frame)
    dec_model = make_model(config, reduced['rounds'], a.threads)
    dec_model.fit(dec_prep.transform(frame), frame.posted_rate)
    dec = pd.read_csv(a.data_dir/'december-chart-inputs.csv')
    dec_x = dec.copy(); dec_x['date'] = pd.to_datetime(dec_x.date)
    dec_pred = np.asarray(dec_model.predict(dec_prep.transform(dec_x)))
    check_predictions(dec_pred)
    dec['predicted_rate'] = dec_pred
    dec.to_csv(paths[1], index=False)
    info = {'version': 'v3-no-quote', 'main_rounds': rounds, 'main_features': list(x.columns),
            'excluded': ['quote_signal', 'quote_based_rate'], 'reduced_rounds': reduced['rounds'],
            'training_rows': len(frame), 'validation_rows': len(valid), 'december_rows': len(dec),
            'seed': 42, 'threads': a.threads, 'python': platform.python_version(),
            'versions': {p: importlib.metadata.version(p) for p in ['pandas','numpy','catboost','lightgbm','scikit-learn','matplotlib']},
            'code_sha256': {n: sha(base/n) for n in ['train.py','compare_models.py','reduced_model.py']},
            'input_sha256': {n: sha(a.data_dir/n) for n in ['train-test.csv','validation.csv','validation-predictions-template.csv','december-chart-inputs.csv']},
            'output_sha256': {p.name: sha(p) for p in paths},
            'note': 'Frozen post-v2 development selection; no new October or hidden validation score.'}
    (a.output_dir/'final_run.json').write_text(json.dumps(info, indent=2))
    print('Created 12,000 main predictions and 31 December predictions.', flush=True)


if __name__ == '__main__':
    main()
