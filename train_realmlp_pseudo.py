"""
RealMLP + pseudo-labeling experiment.

Adds confident test predictions (p > threshold or p < 1-threshold) into train.
Did not improve public LB vs plain RealMLP.

Usage:
    python train_realmlp_pseudo.py --no-submit
    python train_realmlp_pseudo.py --threshold 0.95
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

from src.config import DATA_DIR, SUBMISSIONS_DIR
from train_realmlp import feature_engineering

ID = 'id'
TARGET = 'PitNextLap'

PARAMS = {
    'random_state': 42,
    'verbosity': 1,
    'val_metric_name': '1-auc_ovr',
    'n_ens': 20,
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


def main(args):
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    print("=== Loading data ===")
    train = pd.read_csv(DATA_DIR / "train.csv")
    test  = pd.read_csv(DATA_DIR / "test.csv")
    orig  = pd.read_csv(DATA_DIR / "original" / "f1_strategy_dataset_v4.csv")
    orig  = orig.drop(columns=['Normalized_TyreLife'], errors='ignore')

    # Pseudo-labels из лучшего submission
    pseudo_sub = pd.read_csv(args.pseudo_file)
    print(f"Pseudo file: {args.pseudo_file}")
    print(f"Threshold: >{args.threshold} or <{1-args.threshold}")

    # Мёржим test с pseudo-labels
    test_with_labels = test.merge(pseudo_sub, on=ID)
    confident = test_with_labels[
        (test_with_labels[TARGET] > args.threshold) |
        (test_with_labels[TARGET] < (1 - args.threshold))
    ].copy()
    confident[TARGET] = (confident[TARGET] > 0.5).astype(int)
    confident = confident.drop(columns=[ID])

    print(f"Confident pseudo-labels: {len(confident):,}")
    print(f"  pos: {confident[TARGET].sum():,} ({confident[TARGET].mean()*100:.1f}%)")
    print(f"  neg: {(1-confident[TARGET]).sum():,}")

    # Prep orig
    y_orig = orig[TARGET]
    orig   = orig.drop(columns=[TARGET])

    # Prep train/test
    X      = train.drop(columns=[ID, TARGET])
    y      = train[TARGET]
    train_id = train[ID]
    X_test   = test.drop(columns=[ID])
    test_id  = test[ID]

    # Pseudo — берём фичи (без id и target)
    X_pseudo = confident.drop(columns=[TARGET], errors='ignore')
    y_pseudo = confident[TARGET]

    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(exclude=['object']).columns.tolist()

    print("\n=== Feature engineering ===")
    X,       _, _, combo_names = feature_engineering(X,       cat_cols, num_cols, fit=True)
    X_test,  _, _, _           = feature_engineering(X_test,  cat_cols, num_cols, fit=False)
    orig,    _, _, _           = feature_engineering(orig,    cat_cols, num_cols, fit=False)
    X_pseudo,_, _, _           = feature_engineering(X_pseudo,cat_cols, num_cols, fit=False)

    print(f"Features: {X.shape[1]}")

    print("\n=== Training RealMLP + Pseudo-labeling ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds  = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    for fold, ((tr_idx, val_idx), (or_tr_idx, _)) in enumerate(
            zip(skf.split(X, y), skf.split(orig, y_orig)), 1):

        # train fold + orig + pseudo
        X_tr = pd.concat([
            X.iloc[tr_idx],
            orig.iloc[or_tr_idx],
            X_pseudo,
        ], axis=0).reset_index(drop=True)
        y_tr = pd.concat([
            y.iloc[tr_idx],
            y_orig.iloc[or_tr_idx],
            y_pseudo,
        ], axis=0).reset_index(drop=True)

        X_val = X.iloc[val_idx].copy()
        y_val = y.iloc[val_idx]
        X_tst = X_test.copy()

        # Target encoding
        TE = TargetEncoder(cv=5, smooth='auto', shuffle=True, random_state=42)
        te_names = [f"_{c}TE" for c in combo_names]
        X_tr[te_names]  = TE.fit_transform(X_tr[combo_names], y_tr)
        X_val[te_names] = TE.transform(X_val[combo_names])
        X_tst[te_names] = TE.transform(X_tst[combo_names])

        if fold == 1:
            print(f"Train size per fold: {len(X_tr):,} (orig fold + {len(orig.iloc[or_tr_idx]):,} orig + {len(X_pseudo):,} pseudo)")

        print(f"\n{'#'*16}\n### Fold {fold}/5 ...\n{'#'*16}")
        model = RealMLP_TD_Classifier(**PARAMS)
        model.fit(X_tr, y_tr, X_val, y_val)

        val_preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_preds
        test_preds += model.predict_proba(X_tst)[:, 1] / 5
        print(f"Fold {fold} | AUC: {roc_auc_score(y_val, val_preds):.5f}")

    oof_score = roc_auc_score(y, oof_preds)
    print(f"\n{'='*30}\nOverall OOF AUC: {oof_score:.5f}\n{'='*30}")

    ts   = datetime.now().strftime('%Y%m%d_%H%M')
    desc = f"RealMLP +pseudo(thr={args.threshold}) OOF={oof_score:.4f}"
    out  = SUBMISSIONS_DIR / f"submission_pseudo_{ts}.csv"
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
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--pseudo-file", type=str,
                        default="submissions/submission_realmlp_20260523_1132.csv")
    args = parser.parse_args()
    main(args)
