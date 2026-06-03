"""
RealMLP v2 — добавляем temporal фичи поверх yekenot baseline.
Тест: помогают ли lag/rolling/deg фичи RealMLP?

Usage:
    python train_realmlp_v2.py --n-ens 5 --no-submit   # быстрый тест
    python train_realmlp_v2.py --n-ens 20               # полный прогон
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

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from pytabkit import RealMLP_TD_Classifier

warnings.filterwarnings('ignore')

from src.config import DATA_DIR, SUBMISSIONS_DIR, COMPOUND_MAP, TYRE_LIMITS

ID = 'id'
TARGET = 'PitNextLap'

# ── Temporal фичи которые добавляем поверх сырых ─────────────────────────────
TEMPORAL_COLS = [
    "tyre_life_lag1", "tyre_life_lag2", "tyre_life_lag3",
    "lap_time_roll3", "lap_time_roll5", "lap_time_roll10",
    "pace_degradation", "pace_degradation_10",
    "lap_time_delta_pct",
    "laps_remaining", "laps_vs_tyre",
    "near_tyre_limit",
    "gap_pressure", "stint_lap",
    "deg_per_lap",
]

category_map = {}


def build_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Temporal FE — работает на отсортированном по race_id+Driver+LapNumber df."""
    df = df.copy()

    df["compound_enc"] = df["Compound"].map(COMPOUND_MAP).fillna(2).astype(int)
    df["total_laps_est"] = (df["LapNumber"] / df["RaceProgress"].replace(0, np.nan)).round()
    df["laps_remaining"] = (df["total_laps_est"] - df["LapNumber"]).fillna(0).clip(lower=0)

    grp = df.groupby(["race_id", "Driver"])

    for lag in [1, 2, 3]:
        df[f"tyre_life_lag{lag}"] = grp["TyreLife"].shift(lag)

    lag1_time = grp["LapTime (s)"].shift(1).replace(0, np.nan)
    df["lap_time_delta_pct"] = df["LapTime_Delta"] / lag1_time

    df["lap_time_roll3"] = grp["LapTime (s)"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    df["lap_time_roll5"] = grp["LapTime (s)"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    df["lap_time_roll10"] = grp["LapTime (s)"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=1).mean()
    )

    df["pace_degradation"] = df["LapTime (s)"] - df["lap_time_roll5"]
    df["pace_degradation_10"] = df["LapTime (s)"] - df["lap_time_roll10"]
    df["gap_pressure"] = grp["Position_Change"].diff()

    near_limit = pd.Series(False, index=df.index)
    for enc, limit in TYRE_LIMITS.items():
        near_limit |= (df["compound_enc"] == enc) & (df["TyreLife"] >= limit)
    df["near_tyre_limit"] = near_limit.astype(int)
    df["laps_vs_tyre"] = df["laps_remaining"] - df["TyreLife"]

    df["stint_lap"] = grp["Stint"].transform(
        lambda x: x.groupby((x != x.shift()).cumsum()).cumcount() + 1
    )
    df["deg_per_lap"] = df["Cumulative_Degradation"] / df["TyreLife"].replace(0, np.nan)

    fill_cols = [c for c in TEMPORAL_COLS if c in df.columns]
    df[fill_cols] = df[fill_cols].fillna(0)
    return df


def feature_engineering(df, cat_cols, num_cols, fit=False):
    important_combos = [('Race', 'Compound'), ('Race', 'Year')]

    df['_LapNumber_/_RaceProgress'] = (df['LapNumber'] / (df['RaceProgress'] + 1e-6)).astype('float32')
    df['_TyreLife_/_LapNumber'] = (df['TyreLife'] / df['LapNumber'].clip(lower=1)).astype('float32')
    df['_LapTime (s)_*_Cumulative_Degradation'] = (df['LapTime (s)'] * df['Cumulative_Degradation']).astype('float32')
    df['_LapTime (s)_*_Cumulative_Degradation_abs'] = (df['LapTime (s)'] * df['Cumulative_Degradation'].abs()).astype('float32')
    df['_LapTime (s)_/_Cumulative_Degradation_abs'] = (df['LapTime (s)'] / (df['Cumulative_Degradation'].abs() + 1e-6)).astype('float32')

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

    for col in cat_cols + ['Year_cat_', 'PitStop_cat_']:
        count_name = f"_{col}_count" if col in cat_cols else f"_{col[:-1]}_count"
        if fit:
            count_map = df[col].value_counts()
            category_map[count_name] = count_map
        else:
            count_map = category_map[count_name]
        df[count_name] = df[col].map(count_map).fillna(0).astype('int32')

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
    seed = args.seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    params = {
        'random_state': seed,
        'verbosity': 1,
        'val_metric_name': '1-auc_ovr',
        'n_ens': args.n_ens,
        'n_epochs': 5,
        'batch_size': 256,
        'use_early_stopping': False,
        'lr': 0.019, 'wd': 0.01, 'sq_mom': 0.99,
        'lr_sched': 'lin_cos_log_15', 'first_layer_lr_factor': 0.25,
        'embedding_size': 6, 'max_one_hot_cat_size': 18,
        'hidden_sizes': [512, 256, 128], 'act': 'silu',
        'p_drop': 0.05, 'p_drop_sched': 'invsqrtp1e-3',
        'plr_hidden_1': 16, 'plr_hidden_2': 8, 'plr_act_name': 'gelu',
        'plr_lr_factor': 0.1151, 'plr_sigma': 2.33,
        'ls_eps': 0.01, 'ls_eps_sched': 'sqrt_cos',
        'add_front_scale': False, 'bias_init_mode': 'neg-uniform-dynamic-2',
        'tfms': ['one_hot', 'median_center', 'robust_scale',
                 'smooth_clip', 'embedding', 'l2_normalize'],
    }

    print("=== Loading data ===")
    train = pd.read_csv(DATA_DIR / "train.csv")
    test  = pd.read_csv(DATA_DIR / "test.csv")
    orig  = pd.read_csv(DATA_DIR / "original" / "f1_strategy_dataset_v4.csv")
    orig  = orig.drop(columns=['Normalized_TyreLife'], errors='ignore')

    # ── Temporal FE: мёржим всё для правильного порядка лапов ────────────────
    print("Building temporal features...")
    train["_split"] = "train"; test["_split"] = "test"
    test[TARGET] = np.nan
    orig["_split"] = "orig"

    df_all = pd.concat([train, test, orig], ignore_index=True)
    df_all["race_id"] = df_all["Race"].astype(str) + "_" + df_all["Year"].astype(str) + "_" + df_all["_split"]
    df_all = df_all.sort_values(["race_id", "Driver", "LapNumber"]).reset_index(drop=True)
    df_all = build_temporal_features(df_all)

    # Разделяем обратно
    train_df = df_all[df_all["_split"] == "train"].copy()
    test_df  = df_all[df_all["_split"] == "test"].copy()
    orig_df  = df_all[df_all["_split"] == "orig"].copy()

    # ── yekenot FE ────────────────────────────────────────────────────────────
    print("Building yekenot features...")
    base_cols = [c for c in train.columns if c not in [ID, TARGET, "_split"]]
    cat_cols = train[base_cols].select_dtypes(include=['object']).columns.tolist()
    num_cols = train[base_cols].select_dtypes(exclude=['object']).columns.tolist()

    X      = train_df[base_cols + TEMPORAL_COLS].copy()
    X_test = test_df[base_cols + TEMPORAL_COLS].copy()
    X_orig = orig_df[[c for c in base_cols + TEMPORAL_COLS if c in orig_df.columns]].copy()
    for c in base_cols + TEMPORAL_COLS:
        if c not in X_orig.columns:
            X_orig[c] = 0.0

    y       = train_df[TARGET]
    y_orig  = orig_df[TARGET]
    train_id = train_df[ID]
    test_id  = test_df[ID]

    # num_cols расширяем temporal
    num_cols_ext = num_cols + [c for c in TEMPORAL_COLS if c not in num_cols]

    X,      new_cat, new_num, combo_names = feature_engineering(X,      cat_cols, num_cols_ext, fit=True)
    X_test, _,       _,       _           = feature_engineering(X_test, cat_cols, num_cols_ext, fit=False)
    X_orig, _,       _,       _           = feature_engineering(X_orig, cat_cols, num_cols_ext, fit=False)

    cat_cols_all = cat_cols + new_cat
    print(f"Total features: {X.shape[1]}  (cat={len(cat_cols_all)}, num+temporal={len(num_cols_ext + new_num)})")

    # ── Training ──────────────────────────────────────────────────────────────
    print(f"\n=== Training RealMLP v2 (n_ens={args.n_ens}) ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_preds  = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    for fold, ((tr_idx, val_idx), (or_tr_idx, _)) in enumerate(
            zip(skf.split(X, y), skf.split(X_orig, y_orig)), 1):

        X_tr = pd.concat([X.iloc[tr_idx], X_orig.iloc[or_tr_idx]], axis=0).reset_index(drop=True)
        y_tr = pd.concat([y.iloc[tr_idx], y_orig.iloc[or_tr_idx]], axis=0).reset_index(drop=True)
        X_val = X.iloc[val_idx].copy()
        y_val = y.iloc[val_idx]
        X_tst = X_test.copy()

        # Target encoding
        te_cols = combo_names
        TE = TargetEncoder(cv=5, smooth='auto', shuffle=True, random_state=seed)
        tr_enc  = TE.fit_transform(X_tr[te_cols], y_tr)
        val_enc = TE.transform(X_val[te_cols])
        tst_enc = TE.transform(X_tst[te_cols])
        te_names = [f"_{c}TE" for c in te_cols]
        X_tr[te_names]  = tr_enc
        X_val[te_names] = val_enc
        X_tst[te_names] = tst_enc

        if fold == 1:
            print(f"Features per fold: {len(X_tr.columns)}")

        print(f"\n{'#'*16}\n### Fold {fold}/5 ...\n{'#'*16}")

        model = RealMLP_TD_Classifier(**params)
        model.fit(X_tr, y_tr, X_val, y_val)

        val_preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_preds
        test_preds += model.predict_proba(X_tst)[:, 1] / 5

        print(f"Fold {fold} | AUC: {roc_auc_score(y_val, val_preds):.5f}")

    oof_score = roc_auc_score(y, oof_preds)
    print(f"\n{'='*30}\nOverall OOF AUC: {oof_score:.5f}\n{'='*30}")

    ts   = datetime.now().strftime('%Y%m%d_%H%M')
    desc = f"RealMLP_v2 +temporal n_ens={args.n_ens} OOF={oof_score:.4f}"
    out  = SUBMISSIONS_DIR / f"submission_realmlp_v2_{ts}.csv"
    pd.DataFrame({ID: test_id, TARGET: test_preds}).to_csv(out, index=False)
    print(f"Saved: {out}")

    if not args.no_submit:
        import subprocess
        subprocess.run([
            ".venv/bin/kaggle", "competitions", "submit",
            "-c", "playground-series-s6e5", "-f", str(out), "-m", desc
        ], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-submit", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-ens", type=int, default=5)
    args = parser.parse_args()
    main(args)
