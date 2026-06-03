"""
Full ensemble training (LGB + XGB + CatBoost) with original F1 dataset augmentation.
Each fold's train portion gets the entire original dataset appended.
Generates submission at the end.

Usage:
    python train_with_orig.py
    python train_with_orig.py --no-submit   # skip Kaggle submission
"""

import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from datetime import datetime
import json
import joblib

from src.config import (
    DATA_DIR, MODELS_DIR, SUBMISSIONS_DIR, TARGET,
    GBDT_FEATURES, N_FOLDS,
    LGB_PARAMS, XGB_PARAMS, CAT_PARAMS,
)
from src.features import load_combined, build_features, split_back, add_fastf1_priors


ORIG_PATH = DATA_DIR / "original" / "f1_strategy_dataset_v4.csv"


# ── Optuna best LGB params (35 trials) ────────────────────────────────────────
BEST_LGB_PATH = MODELS_DIR / "best_lgb_params.json"


def load_lgb_params():
    if BEST_LGB_PATH.exists():
        with open(BEST_LGB_PATH) as f:
            d = json.load(f)
        d.pop("best_auc", None)
        return d
    print("  [LGB] No Optuna params found, using defaults from config.")
    return {k: v for k, v in LGB_PARAMS.items() if k not in ("objective", "metric", "verbose", "random_state", "n_estimators")}


def load_original():
    """Load original dataset and apply same feature engineering."""
    print(f"\n  Loading original dataset from {ORIG_PATH}...")
    orig = pd.read_csv(ORIG_PATH)
    print(f"  Original shape: {orig.shape}")

    # Drop columns not in competition train
    for c in ["Normalized_TyreLife", "id"]:
        if c in orig.columns:
            orig.drop(columns=[c], inplace=True)

    # Mark as original so build_features knows this is "train" data
    orig["_split"] = "train"

    # Create race_id (same logic as features.py)
    orig["race_id"] = orig["Race"].astype(str) + "_" + orig["Year"].astype(str) + "_orig"
    orig = orig.sort_values(["race_id", "Driver", "LapNumber"]).reset_index(drop=True)

    # Build features — need a minimal stub so build_features doesn't crash
    # (it groups by race_id + Driver, so it'll work fine)
    orig = build_features(orig)

    # Keep only GBDT_FEATURES + target
    keep = [c for c in GBDT_FEATURES if c in orig.columns] + [TARGET]
    missing = [c for c in GBDT_FEATURES if c not in orig.columns]
    if missing:
        print(f"  [WARN] Missing features in orig, will fill with 0: {missing}")
        for c in missing:
            orig[c] = 0.0

    return orig[GBDT_FEATURES + [TARGET]]


def train_lgb_with_orig(train_df, orig_df, verbose=True):
    """Train LGB with original data appended to each fold's train portion."""
    neg = (train_df[TARGET] == 0).sum()
    pos = (train_df[TARGET] == 1).sum()

    optuna_params = load_lgb_params()
    params = {
        "objective": "binary",
        "metric": "auc",
        "n_estimators": 2000,
        "verbose": -1,
        "random_state": 42,
        "scale_pos_weight": neg / pos,
        **optuna_params,
    }

    X_tr_base = train_df[GBDT_FEATURES].values
    y_base = train_df[TARGET].values
    X_orig = orig_df[GBDT_FEATURES].values
    y_orig = orig_df[TARGET].values
    groups = train_df["race_id"].values

    oof = np.zeros(len(train_df))
    models = []
    gkf = GroupKFold(n_splits=N_FOLDS)

    for fold, (ti, vi) in enumerate(gkf.split(X_tr_base, y_base, groups)):
        X_fold_tr = np.vstack([X_tr_base[ti], X_orig])
        y_fold_tr = np.concatenate([y_base[ti], y_orig])
        X_val = X_tr_base[vi]
        y_val = y_base[vi]

        # Recompute scale_pos_weight for augmented fold
        neg_f = (y_fold_tr == 0).sum()
        pos_f = (y_fold_tr == 1).sum()
        params_fold = {**params, "scale_pos_weight": neg_f / pos_f}

        model = lgb.LGBMClassifier(**params_fold)
        model.fit(
            X_fold_tr, y_fold_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)],
        )

        oof[vi] = model.predict_proba(X_val)[:, 1]
        fold_auc = roc_auc_score(y_val, oof[vi])
        models.append(model)

        if verbose:
            print(f"  [LGB] Fold {fold + 1}/{N_FOLDS}  AUC={fold_auc:.4f}  trees={model.best_iteration_}")

    oof_auc = roc_auc_score(y_base, oof)
    if verbose:
        print(f"  [LGB] OOF AUC={oof_auc:.4f}")

    joblib.dump(models, MODELS_DIR / "lgb_models.pkl")
    np.save(MODELS_DIR / "oof_lgb.npy", oof)
    return oof, models


def train_xgb_with_orig(train_df, orig_df, verbose=True):
    """Train XGB with original data appended to each fold's train portion."""
    neg = (train_df[TARGET] == 0).sum()
    pos = (train_df[TARGET] == 1).sum()

    params = {
        **XGB_PARAMS,
        "scale_pos_weight": neg / pos,
        "early_stopping_rounds": 100,
    }

    X_tr_base = train_df[GBDT_FEATURES].values
    y_base = train_df[TARGET].values
    X_orig = orig_df[GBDT_FEATURES].values
    y_orig = orig_df[TARGET].values
    groups = train_df["race_id"].values

    oof = np.zeros(len(train_df))
    models = []
    gkf = GroupKFold(n_splits=N_FOLDS)

    for fold, (ti, vi) in enumerate(gkf.split(X_tr_base, y_base, groups)):
        X_fold_tr = np.vstack([X_tr_base[ti], X_orig])
        y_fold_tr = np.concatenate([y_base[ti], y_orig])
        X_val = X_tr_base[vi]
        y_val = y_base[vi]

        neg_f = (y_fold_tr == 0).sum()
        pos_f = (y_fold_tr == 1).sum()
        params_fold = {**params, "scale_pos_weight": neg_f / pos_f}

        model = xgb.XGBClassifier(**params_fold)
        model.fit(
            X_fold_tr, y_fold_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        oof[vi] = model.predict_proba(X_val)[:, 1]
        fold_auc = roc_auc_score(y_val, oof[vi])
        models.append(model)

        if verbose:
            print(f"  [XGB] Fold {fold + 1}/{N_FOLDS}  AUC={fold_auc:.4f}")

    oof_auc = roc_auc_score(y_base, oof)
    if verbose:
        print(f"  [XGB] OOF AUC={oof_auc:.4f}")

    joblib.dump(models, MODELS_DIR / "xgb_models.pkl")
    np.save(MODELS_DIR / "oof_xgb.npy", oof)
    return oof, models


def train_cat_with_orig(train_df, orig_df, verbose=True):
    """Train CatBoost with original data appended to each fold's train portion."""
    neg = (train_df[TARGET] == 0).sum()
    pos = (train_df[TARGET] == 1).sum()

    params = {
        **CAT_PARAMS,
        "scale_pos_weight": neg / pos,
    }

    X_tr_base = train_df[GBDT_FEATURES].values
    y_base = train_df[TARGET].values
    X_orig = orig_df[GBDT_FEATURES].values
    y_orig = orig_df[TARGET].values
    groups = train_df["race_id"].values

    oof = np.zeros(len(train_df))
    models = []
    gkf = GroupKFold(n_splits=N_FOLDS)

    for fold, (ti, vi) in enumerate(gkf.split(X_tr_base, y_base, groups)):
        X_fold_tr = np.vstack([X_tr_base[ti], X_orig])
        y_fold_tr = np.concatenate([y_base[ti], y_orig])
        X_val = X_tr_base[vi]
        y_val = y_base[vi]

        neg_f = (y_fold_tr == 0).sum()
        pos_f = (y_fold_tr == 1).sum()
        params_fold = {**params, "scale_pos_weight": neg_f / pos_f}

        model = CatBoostClassifier(**params_fold)
        model.fit(
            X_fold_tr, y_fold_tr,
            eval_set=(X_val, y_val),
            use_best_model=True,
            verbose=False,
        )

        oof[vi] = model.predict_proba(X_val)[:, 1]
        fold_auc = roc_auc_score(y_val, oof[vi])
        models.append(model)

        if verbose:
            print(f"  [CAT] Fold {fold + 1}/{N_FOLDS}  AUC={fold_auc:.4f}")

    oof_auc = roc_auc_score(y_base, oof)
    if verbose:
        print(f"  [CAT] OOF AUC={oof_auc:.4f}")

    joblib.dump(models, MODELS_DIR / "cat_models.pkl")
    np.save(MODELS_DIR / "oof_cat.npy", oof)
    return oof, models


def main(args):
    print("=== Loading & merging train+test for temporal FE ===")
    df = load_combined(DATA_DIR / "train.csv", DATA_DIR / "test.csv")
    df = build_features(df)
    train_df, test_df = split_back(df)

    print(f"Train rows: {len(train_df):,}  |  Target=1: {train_df[TARGET].sum():,.0f} ({train_df[TARGET].mean()*100:.1f}%)")

    # Load original dataset
    orig_df = load_original()
    print(f"  Original rows: {len(orig_df):,}  |  Target=1 rate: {orig_df[TARGET].mean()*100:.1f}%")

    print("\n=== LightGBM ===")
    oof_lgb, lgb_models = train_lgb_with_orig(train_df, orig_df)

    print("\n=== XGBoost ===")
    oof_xgb, xgb_models = train_xgb_with_orig(train_df, orig_df)

    print("\n=== CatBoost ===")
    oof_cat, cat_models = train_cat_with_orig(train_df, orig_df)

    # Ensemble OOF
    oof_ensemble = (oof_lgb + oof_xgb + oof_cat) / 3
    ensemble_auc = roc_auc_score(train_df[TARGET], oof_ensemble)

    print(f"\n{'='*55}")
    print(f"  LGB  OOF AUC : {roc_auc_score(train_df[TARGET], oof_lgb):.4f}")
    print(f"  XGB  OOF AUC : {roc_auc_score(train_df[TARGET], oof_xgb):.4f}")
    print(f"  CAT  OOF AUC : {roc_auc_score(train_df[TARGET], oof_cat):.4f}")
    print(f"  Ensemble OOF : {ensemble_auc:.4f}")
    print(f"{'='*55}")

    # Generate predictions
    print("\n=== Generating submission ===")
    from src.model_lgb import predict_lgb
    from src.model_xgb import predict_xgb
    from src.model_cat import predict_cat

    p_lgb = predict_lgb(test_df, lgb_models)
    p_xgb = predict_xgb(test_df, xgb_models)
    p_cat = predict_cat(test_df, cat_models)
    p_final = (p_lgb + p_xgb + p_cat) / 3

    sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
    id_to_pred = dict(zip(test_df["id"].values, p_final))
    sample_sub[TARGET] = sample_sub["id"].map(id_to_pred)

    desc = f"LGB+XGB+CAT +orig_data OOF={ensemble_auc:.4f}"
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out = SUBMISSIONS_DIR / f"submission_orig_{ts}.csv"
    sample_sub.to_csv(out, index=False)
    print(f"Saved: {out}")
    print(f"Description: {desc}")
    print(f"Mean probability: {p_final.mean():.4f}")

    if not args.no_submit:
        import subprocess
        cmd = [
            "kaggle", "competitions", "submit",
            "-c", "playground-series-s6e5",
            "-f", str(out),
            "-m", desc
        ]
        print(f"\nSubmitting: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    return str(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-submit", action="store_true", help="Skip Kaggle submission")
    args = parser.parse_args()
    main(args)
