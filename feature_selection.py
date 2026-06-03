"""
Feature selection script — быстрый прокси через LightGBM + StratifiedKFold.

Логика:
1. Строим baseline (текущие фичи RealMLP + наши temporal)
2. Считаем permutation importance
3. Смотрим какие новые фичи дают прирост OOF

Usage:
    python feature_selection.py              # baseline + importance
    python feature_selection.py --test-new   # тест новых фич поверх baseline
"""

import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance

from src.config import DATA_DIR, TARGET, COMPOUND_MAP, TYRE_LIMITS

N_FOLDS = 5
SEED = 42

# ── Фичи которые сейчас в RealMLP (сырые) ────────────────────────────────────
REALMLP_RAW = [
    "TyreLife", "LapNumber", "Stint", "Position",
    "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
    "RaceProgress", "Position_Change", "PitStop", "Year",
]

# ── Наши temporal фичи из GBDT pipeline ──────────────────────────────────────
TEMPORAL_FEATURES = [
    "tyre_life_lag1", "tyre_life_lag2", "tyre_life_lag3",
    "lap_time_roll3", "lap_time_roll5", "lap_time_roll10",
    "pace_degradation", "pace_degradation_10",
    "lap_time_delta_pct",
    "laps_remaining", "laps_vs_tyre",
    "near_tyre_limit", "position_norm",
    "gap_pressure", "stint_lap",
    "deg_per_lap",
]

# ── Новые кандидаты ───────────────────────────────────────────────────────────
NEW_FEATURES = [
    "laps_since_last_pit",       # кругов с последнего пит-стопа
    "tyre_age_pct",              # TyreLife / типичный лимит (0-1)
    "lap_time_trend",            # линейный тренд lap time за последние 5 кругов
    "position_delta_3",          # изменение позиции за 3 круга
    "cumdeg_per_lap",            # Cumulative_Degradation / TyreLife
    "is_first_stint",            # Stint == 1
    "laps_remaining_sq",         # laps_remaining^2 (нелинейность)
    "tyre_x_deg",                # TyreLife * deg_per_lap
    "pace_vs_roll10",            # LapTime vs rolling10 (относительная деградация)
    "stint_progress",            # stint_lap / laps_remaining (насколько далеко в стинте)
]


def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Строим все фичи — baseline + новые кандидаты."""
    df = df.copy()

    # ── Базовые ───────────────────────────────────────────────────────────────
    df["compound_enc"] = df["Compound"].map(COMPOUND_MAP).fillna(2).astype(int)
    df["total_laps_est"] = (df["LapNumber"] / df["RaceProgress"].replace(0, np.nan)).round()
    df["laps_remaining"] = (df["total_laps_est"] - df["LapNumber"]).fillna(0).clip(lower=0)
    df["position_norm"] = df["Position"] / df.groupby("race_id")["Position"].transform("max")

    grp = df.groupby(["race_id", "Driver"])

    # Лаги
    for lag in [1, 2, 3]:
        df[f"tyre_life_lag{lag}"] = grp["TyreLife"].shift(lag)

    lag1_time = grp["LapTime (s)"].shift(1).replace(0, np.nan)
    df["lap_time_delta_pct"] = df["LapTime_Delta"] / lag1_time

    # Роллинги
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

    # near_tyre_limit
    near_limit = pd.Series(False, index=df.index)
    for enc, limit in TYRE_LIMITS.items():
        near_limit |= (df["compound_enc"] == enc) & (df["TyreLife"] >= limit)
    df["near_tyre_limit"] = near_limit.astype(int)
    df["laps_vs_tyre"] = df["laps_remaining"] - df["TyreLife"]

    df["stint_lap"] = grp["Stint"].transform(
        lambda x: x.groupby((x != x.shift()).cumsum()).cumcount() + 1
    )
    df["deg_per_lap"] = df["Cumulative_Degradation"] / df["TyreLife"].replace(0, np.nan)

    # ── Новые кандидаты ───────────────────────────────────────────────────────

    # Кругов с последнего пит-стопа
    df["laps_since_last_pit"] = grp["PitStop"].transform(
        lambda x: x.groupby(x.cumsum()).cumcount()
    )

    # Возраст шины относительно типичного лимита
    tyre_limit_series = df["compound_enc"].map(TYRE_LIMITS).fillna(40)
    df["tyre_age_pct"] = (df["TyreLife"] / tyre_limit_series).clip(0, 2)

    # Линейный тренд lap time за 5 кругов (положительный = деградация)
    def lap_time_trend(x):
        def rolling_slope(s):
            if len(s) < 2:
                return 0.0
            y = s.values
            x_ = np.arange(len(y))
            if np.std(y) < 1e-9:
                return 0.0
            return np.polyfit(x_, y, 1)[0]
        return x.shift(1).rolling(5, min_periods=2).apply(rolling_slope, raw=False)

    df["lap_time_trend"] = grp["LapTime (s)"].transform(lap_time_trend)

    # Изменение позиции за 3 круга
    df["position_delta_3"] = grp["Position"].diff(3)

    # Cumulative degradation per lap
    df["cumdeg_per_lap"] = df["Cumulative_Degradation"] / df["TyreLife"].replace(0, np.nan)

    # Первый стинт
    df["is_first_stint"] = (df["Stint"] == 1).astype(int)

    # Нелинейность laps_remaining
    df["laps_remaining_sq"] = df["laps_remaining"] ** 2

    # Взаимодействие
    df["tyre_x_deg"] = df["TyreLife"] * df["deg_per_lap"].fillna(0)

    # Темп относительно rolling10
    df["pace_vs_roll10"] = (df["LapTime (s)"] - df["lap_time_roll10"]) / (df["lap_time_roll10"].replace(0, np.nan))

    # Прогресс стинта
    df["stint_progress"] = df["stint_lap"] / (df["laps_remaining"] + 1)

    # Fill NaN
    fill_cols = TEMPORAL_FEATURES + NEW_FEATURES
    df[fill_cols] = df[[c for c in fill_cols if c in df.columns]].fillna(0)

    return df


def run_lgb_cv(X, y, features, label=""):
    """5-fold StratifiedKFold LGB, возвращает OOF AUC."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    neg, pos = (y == 0).sum(), (y == 1).sum()

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 118,
        "min_child_samples": 74,
        "n_estimators": 500,
        "verbose": -1,
        "random_state": SEED,
        "scale_pos_weight": neg / pos,
    }

    models = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )
        oof[val_idx] = model.predict_proba(X_val)[:, 1]
        models.append(model)

    auc = roc_auc_score(y, oof)
    print(f"  [{label}] OOF AUC: {auc:.5f}")
    return auc, oof, models


def permutation_importance_report(models, X_df, y, features, n_repeats=5):
    """Permutation importance усреднённая по фолдам."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    importances = np.zeros(len(features))

    for (_, val_idx), model in zip(skf.split(X_df, y), models):
        X_val = X_df[val_idx]
        y_val = y[val_idx]
        result = permutation_importance(
            model, X_val, y_val,
            n_repeats=n_repeats,
            scoring="roc_auc",
            random_state=SEED,
            n_jobs=-1,
        )
        importances += result.importances_mean

    importances /= N_FOLDS
    imp_df = pd.DataFrame({
        "feature": features,
        "importance": importances,
    }).sort_values("importance", ascending=False)
    return imp_df


def main(args):
    print("=== Loading data ===")
    train = pd.read_csv(DATA_DIR / "train.csv")
    test  = pd.read_csv(DATA_DIR / "test.csv")

    # Merge for temporal FE (нужен порядок лапов)
    train["_split"] = "train"
    test["_split"] = "test"
    test[TARGET] = np.nan
    df = pd.concat([train, test], ignore_index=True)
    df["race_id"] = df["Race"].astype(str) + "_" + df["Year"].astype(str)
    df = df.sort_values(["race_id", "Driver", "LapNumber"]).reset_index(drop=True)

    print("Building features...")
    df = build_all_features(df)
    train_df = df[df["_split"] == "train"].copy()
    y = train_df[TARGET].values.astype(int)

    # ── Baseline: только сырые RealMLP фичи ──────────────────────────────────
    baseline_features = REALMLP_RAW + ["compound_enc"]
    missing = [f for f in baseline_features if f not in train_df.columns]
    baseline_features = [f for f in baseline_features if f in train_df.columns]

    print(f"\n{'='*55}")
    print("1. BASELINE (RealMLP raw features only)")
    X_base = train_df[baseline_features].values
    auc_base, oof_base, models_base = run_lgb_cv(X_base, y, baseline_features, "BASELINE")

    # ── + Temporal фичи ───────────────────────────────────────────────────────
    temporal_avail = [f for f in TEMPORAL_FEATURES if f in train_df.columns]
    full_features = baseline_features + temporal_avail

    print(f"\n{'='*55}")
    print(f"2. BASELINE + TEMPORAL ({len(temporal_avail)} фичи)")
    X_temp = train_df[full_features].values
    auc_temp, oof_temp, models_temp = run_lgb_cv(X_temp, y, full_features, "+TEMPORAL")
    print(f"  Delta vs baseline: {auc_temp - auc_base:+.5f}")

    # ── + Новые фичи (по одной) ───────────────────────────────────────────────
    if args.test_new:
        print(f"\n{'='*55}")
        print("3. ТЕСТ НОВЫХ ФИЧ (по одной поверх BASELINE+TEMPORAL)")
        results = []
        for feat in NEW_FEATURES:
            if feat not in train_df.columns:
                print(f"  [{feat}] not found, skip")
                continue
            test_features = full_features + [feat]
            X_test = train_df[test_features].values
            auc_new, _, _ = run_lgb_cv(X_test, y, test_features, feat)
            delta = auc_new - auc_temp
            results.append({"feature": feat, "auc": auc_new, "delta": delta})
            print(f"  Delta: {delta:+.5f}  {'✓' if delta > 0.0001 else '✗'}")

        print(f"\n{'='*55}")
        print("ИТОГ новых фич:")
        results_df = pd.DataFrame(results).sort_values("delta", ascending=False)
        print(results_df.to_string(index=False))

        print(f"\n{'='*55}")
        print("4. ВСЕ ПОЛЕЗНЫЕ фичи вместе")
        good_feats = results_df[results_df["delta"] > 0]["feature"].tolist()
        if good_feats:
            all_features = full_features + good_feats
            X_all = train_df[all_features].values
            auc_all, oof_all, models_all = run_lgb_cv(X_all, y, all_features, "ALL_GOOD")
            print(f"  Delta vs +temporal: {auc_all - auc_temp:+.5f}")
            print(f"  Delta vs baseline:  {auc_all - auc_base:+.5f}")
        else:
            print("  Нет фич с положительным delta")
            models_all = models_temp
            all_features = full_features

    # ── Permutation importance ─────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("5. PERMUTATION IMPORTANCE (baseline+temporal)")
    imp_df = permutation_importance_report(models_temp, X_temp, y, full_features)
    print(imp_df.to_string(index=False))

    print(f"\n{'='*55}")
    print("ВЫВОД:")
    print(f"  Baseline OOF:  {auc_base:.5f}")
    print(f"  +Temporal OOF: {auc_temp:.5f}  ({auc_temp-auc_base:+.5f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-new", action="store_true", help="Test new feature candidates one by one")
    args = parser.parse_args()
    main(args)
