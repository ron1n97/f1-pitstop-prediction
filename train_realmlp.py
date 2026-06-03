"""
RealMLP training pipeline — based on yekenot's top notebook (113 votes).
Uses StratifiedKFold + TargetEncoder + original dataset.

Usage:
    python train_realmlp.py
    python train_realmlp.py --no-submit
"""

import argparse
import random
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder
from datetime import datetime
import sys

import os
# MPS (Apple GPU) causes hangs with pytabkit — force CPU like we do for Bi-LSTM
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from pytabkit import RealMLP_TD_Classifier

warnings.filterwarnings('ignore')

from src.config import DATA_DIR, SUBMISSIONS_DIR

# ── Config ────────────────────────────────────────────────────────────────────
class CFG:
    FOLDS = 5
    SEED = 42
    TE = True

PARAMS = {
    'random_state': 42,
    'verbosity': 1,
    'val_metric_name': '1-auc_ovr',

    'n_ens': 20,
    'n_epochs': 8,
    'batch_size': 256,
    'use_early_stopping': False,

    'lr': 0.019,
    'wd': 0.01,
    'sq_mom': 0.99,
    'lr_sched': 'lin_cos_log_15',
    'first_layer_lr_factor': 0.25,

    'embedding_size': 6,
    'max_one_hot_cat_size': 18,
    'hidden_sizes': [512, 256, 128],
    'act': 'silu',
    'p_drop': 0.05,
    'p_drop_sched': 'invsqrtp1e-3',

    'plr_hidden_1': 16,
    'plr_hidden_2': 8,
    'plr_act_name': 'gelu',
    'plr_lr_factor': 0.1151,
    'plr_sigma': 2.33,

    'ls_eps': 0.01,
    'ls_eps_sched': 'sqrt_cos',

    'add_front_scale': False,
    'bias_init_mode': 'neg-uniform-dynamic-2',
    'tfms': ['one_hot', 'median_center', 'robust_scale',
             'smooth_clip', 'embedding', 'l2_normalize'],
}

ID = 'id'
TARGET = 'PitNextLap'

# ── Seed ──────────────────────────────────────────────────────────────────────
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

seed_everything(CFG.SEED)


# ── Feature engineering (same as yekenot) ────────────────────────────────────
category_map = {}

def feature_engineering(df, cat_cols, num_cols, fit=False):
    important_combos = [
        ('Race', 'Compound'),
        ('Race', 'Year'),
    ]

    # Arithmetic interactions
    df['_LapNumber_/_RaceProgress'] = (df['LapNumber'] / (df['RaceProgress'] + 1e-6)).astype('float32')
    df['_TyreLife_/_LapNumber'] = (df['TyreLife'] / df['LapNumber'].clip(lower=1)).astype('float32')
    df['_LapTime (s)_*_Cumulative_Degradation'] = (df['LapTime (s)'] * df['Cumulative_Degradation']).astype('float32')
    df['_LapTime (s)_*_Cumulative_Degradation_abs'] = (df['LapTime (s)'] * df['Cumulative_Degradation'].abs()).astype('float32')
    df['_LapTime (s)_/_Cumulative_Degradation_abs'] = (df['LapTime (s)'] / (df['Cumulative_Degradation'].abs() + 1e-6)).astype('float32')

    # Categorize numericals
    for col in num_cols + ['_LapNumber_/_RaceProgress', '_TyreLife_/_LapNumber']:
        cat_name = f"{col}_cat_" if col in num_cols else f"{col[1:]}_cat_"
        if fit:
            codes, uniques = np.floor(df[col]).factorize()
            category_map[col] = uniques
        else:
            uniques = category_map[col]
            code_map = {cat: i for i, cat in enumerate(uniques)}
            codes = np.floor(df[col]).map(code_map).fillna(-1).astype('int32')
        df[cat_name] = codes.astype(str)

    # Count encoding
    for col in cat_cols + ['Year_cat_', 'PitStop_cat_']:
        count_name = f"_{col}_count" if col in cat_cols else f"_{col[:-1]}_count"
        if fit:
            count_map = df[col].value_counts()
            category_map[count_name] = count_map
        else:
            count_map = category_map[count_name]
        df[count_name] = df[col].map(count_map).fillna(0).astype('int32')

    # Discretize numericals
    bin_config = {'RaceProgress': [200], 'LapTime (s)': [7]}
    for col, bins_list in bin_config.items():
        for n_bins in bins_list:
            bin_name = f"{col}_{n_bins}_quantile_bin_"
            if fit:
                kb = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile', subsample=None)
                binned = kb.fit_transform(df[[col]]).ravel().astype('int32')
                category_map[bin_name] = kb
            else:
                kb = category_map[bin_name]
                binned = kb.transform(df[[col]]).ravel().astype('int32')
            df[bin_name] = binned.astype(str)

    # Interaction categories
    combo_names = []
    for cols in important_combos:
        combo_name = '_'.join(cols) + '_'
        combo_names.append(combo_name)
        combo_series = df[cols[0]].astype(str)
        for col in cols[1:]:
            combo_series = combo_series + '_' + df[col].astype(str)
        if fit:
            codes, uniques = pd.factorize(combo_series, sort=False)
            category_map[combo_name] = uniques
        else:
            uniques = category_map[combo_name]
            code_map = {cat: i for i, cat in enumerate(uniques)}
            codes = combo_series.map(code_map).fillna(-1).astype('int32')
        df[combo_name] = codes.astype(str)

    new_cat_cols = [col for col in df.columns if col.endswith('_')]
    new_num_cols = [col for col in df.columns if col.startswith('_')]
    return df, new_cat_cols, new_num_cols, combo_names


def main(args):
    print("=== Loading data ===")
    train = pd.read_csv(DATA_DIR / "train.csv")
    test  = pd.read_csv(DATA_DIR / "test.csv")
    orig  = pd.read_csv(DATA_DIR / "original" / "f1_strategy_dataset_v4.csv")

    print(f"Train: {train.shape}, Test: {test.shape}, Orig: {orig.shape}")

    # Prep original
    orig = orig.drop(columns=['Normalized_TyreLife'], errors='ignore')
    y_orig = orig[TARGET]
    orig = orig.drop(columns=[TARGET])

    # Prep train/test
    X = train.drop(columns=[ID, TARGET])
    y = train[TARGET]
    train_id = train[ID]
    X_test = test.drop(columns=[ID])
    test_id = test[ID]

    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(exclude=['object']).columns.tolist()
    print(f"cat_cols: {len(cat_cols)}, num_cols: {len(num_cols)}")

    print("\n=== Feature engineering ===")
    X, new_cat_cols, new_num_cols, combo_names = feature_engineering(X, cat_cols, num_cols, fit=True)
    X_test, _, _, _ = feature_engineering(X_test, cat_cols, num_cols, fit=False)
    orig, _, _, _ = feature_engineering(orig, cat_cols, num_cols, fit=False)
    cat_cols = cat_cols + new_cat_cols
    num_cols = num_cols + new_num_cols
    print(f"After FE — cat: {len(cat_cols)}, num: {len(num_cols)}, total features: {X.shape[1]}")

    print("\n=== Training RealMLP (StratifiedKFold) ===")
    skf = StratifiedKFold(n_splits=CFG.FOLDS, shuffle=True, random_state=CFG.SEED)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    for fold, ((tr_idx, val_idx), (or_tr_idx, _)) in enumerate(
            zip(skf.split(X, y), skf.split(orig, y_orig)), 1):

        X_tr = X.iloc[tr_idx].copy()
        orig_tr = orig.iloc[or_tr_idx].copy()
        X_tr = pd.concat([X_tr, orig_tr], axis=0).reset_index(drop=True)
        y_tr = pd.concat([y.iloc[tr_idx], y_orig.iloc[or_tr_idx]], axis=0).reset_index(drop=True)
        X_val = X.iloc[val_idx].copy()
        y_val = y.iloc[val_idx]
        X_tst = X_test.copy()

        # Target encoding on Race×Compound, Race×Year combos
        if CFG.TE:
            te_cols = combo_names
            TE = TargetEncoder(cv=CFG.FOLDS, smooth='auto', shuffle=True, random_state=CFG.SEED)
            tr_enc = TE.fit_transform(X_tr[te_cols], y_tr)
            val_enc = TE.transform(X_val[te_cols])
            tst_enc = TE.transform(X_tst[te_cols])
            te_names = [f"_{col}TE" for col in te_cols]
            X_tr[te_names] = tr_enc
            X_val[te_names] = val_enc
            X_tst[te_names] = tst_enc

        if fold == 1:
            print(f"Total features per fold: {len(X_tr.columns)}")

        print(f"\n{'#'*16}")
        print(f"### Fold {fold}/{CFG.FOLDS} ...")
        print(f"{'#'*16}")

        model = RealMLP_TD_Classifier(**PARAMS)
        model.fit(X_tr, y_tr, X_val, y_val)

        val_preds = model.predict_proba(X_val)[:, 1]
        fold_test_preds = model.predict_proba(X_tst)[:, 1]

        oof_preds[val_idx] = val_preds
        test_preds += fold_test_preds / CFG.FOLDS

        fold_score = roc_auc_score(y_val, val_preds)
        print(f"Fold {fold} | AUC: {fold_score:.5f}")
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    oof_score = roc_auc_score(y, oof_preds)
    print(f"\n{'='*30}")
    print(f"Overall OOF AUC: {oof_score:.5f}")
    print(f"{'='*30}")

    # Save OOF
    oof_df = pd.DataFrame({ID: train_id, TARGET: oof_preds})
    oof_df.to_csv(SUBMISSIONS_DIR / "oof_realmlp.csv", index=False)

    # Save submission
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    desc = f"RealMLP StratKFold +orig +TE OOF={oof_score:.4f}"
    out = SUBMISSIONS_DIR / f"submission_realmlp_{ts}.csv"
    sub = pd.DataFrame({ID: test_id, TARGET: test_preds})
    sub.to_csv(out, index=False)
    print(f"Saved: {out}")

    if not args.no_submit:
        import subprocess
        cmd = [
            ".venv/bin/kaggle", "competitions", "submit",
            "-c", "playground-series-s6e5",
            "-f", str(out),
            "-m", desc
        ]
        print(f"\nSubmitting...")
        subprocess.run(cmd, check=True)

    return str(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-submit", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    # Override seed if provided
    if args.seed != 42:
        CFG.SEED = args.seed
        PARAMS['random_state'] = args.seed
    main(args)
