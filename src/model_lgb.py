import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from src.config import GBDT_FEATURES, TARGET, N_FOLDS, LGB_PARAMS, MODELS_DIR
import joblib


def train_lgb(df, verbose: bool = True):
    neg = (df[TARGET] == 0).sum()
    pos = (df[TARGET] == 1).sum()

    params = {**LGB_PARAMS, "scale_pos_weight": neg / pos}

    oof = np.zeros(len(df))
    models = []
    gkf = GroupKFold(n_splits=N_FOLDS)

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(df, df[TARGET], df["race_id"])):
        X_tr = df.iloc[tr_idx][GBDT_FEATURES]
        y_tr = df.iloc[tr_idx][TARGET]
        X_val = df.iloc[val_idx][GBDT_FEATURES]
        y_val = df.iloc[val_idx][TARGET]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )

        oof[val_idx] = model.predict_proba(X_val)[:, 1]
        fold_auc = roc_auc_score(y_val, oof[val_idx])
        models.append(model)

        if verbose:
            print(f"  [LGB] Fold {fold + 1}/{N_FOLDS}  AUC={fold_auc:.4f}")

    oof_auc = roc_auc_score(df[TARGET], oof)
    if verbose:
        print(f"  [LGB] OOF AUC={oof_auc:.4f}")

    joblib.dump(models, MODELS_DIR / "lgb_models.pkl")
    return oof, models


def predict_lgb(test_df, models) -> np.ndarray:
    preds = np.mean([m.predict_proba(test_df[GBDT_FEATURES])[:, 1] for m in models], axis=0)
    return preds
