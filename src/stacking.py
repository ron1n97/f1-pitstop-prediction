import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from src.config import TARGET, N_FOLDS, MODELS_DIR
import joblib


def build_meta_input(df, oof_lgb, oof_xgb, oof_cat, oof_lstm):
    extra_cols = ["TyreLife", "laps_remaining", "near_tyre_limit"]
    extra = df[extra_cols].values
    meta = np.column_stack([oof_lgb, oof_xgb, oof_cat, oof_lstm, extra])
    return meta


def train_meta(df, oof_lgb, oof_xgb, oof_cat, oof_lstm, verbose: bool = True):
    meta_input = build_meta_input(df, oof_lgb, oof_xgb, oof_cat, oof_lstm)
    y = df[TARGET].values

    scaler = StandardScaler()
    meta_scaled = scaler.fit_transform(meta_input)

    meta_model = Ridge(alpha=1.0)

    # Manual CV AUC for Ridge (predict() not predict_proba())
    cv_aucs = []
    for tr_idx, val_idx in GroupKFold(N_FOLDS).split(meta_scaled, y, df["race_id"]):
        m = Ridge(alpha=1.0)
        m.fit(meta_scaled[tr_idx], y[tr_idx])
        preds = np.clip(m.predict(meta_scaled[val_idx]), 0, 1)
        from sklearn.metrics import roc_auc_score as _auc
        cv_aucs.append(_auc(y[val_idx], preds))
    if verbose:
        print(f"  [META] AUC={np.mean(cv_aucs):.4f} ± {np.std(cv_aucs):.4f}")

    meta_model.fit(meta_scaled, y)

    joblib.dump({"scaler": scaler, "meta_model": meta_model}, MODELS_DIR / "meta.pkl")
    return meta_model, scaler


def predict_meta(test_df, test_lgb, test_xgb, test_cat, test_lstm, meta_model, scaler) -> np.ndarray:
    meta_input = build_meta_input(test_df, test_lgb, test_xgb, test_cat, test_lstm)
    meta_scaled = scaler.transform(meta_input)
    raw = meta_model.predict(meta_scaled)
    # Ridge gives unconstrained output — clip to [0,1] as probability proxy
    return np.clip(raw, 0.0, 1.0)
