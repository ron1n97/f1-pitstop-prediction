import pandas as pd
import numpy as np
from pathlib import Path
from src.config import COMPOUND_MAP, TYRE_LIMITS, TARGET


def load_combined(train_path: str, test_path: str):
    tr = pd.read_csv(train_path)
    te = pd.read_csv(test_path)

    tr["_split"] = "train"
    te["_split"] = "test"
    te[TARGET] = np.nan

    df = pd.concat([tr, te], ignore_index=True)
    df["race_id"] = df["Race"].astype(str) + "_" + df["Year"].astype(str)
    df = df.sort_values(["race_id", "Driver", "LapNumber"]).reset_index(drop=True)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── Basic encodings ────────────────────────────────────────────────────────
    df["compound_enc"] = df["Compound"].map(COMPOUND_MAP).fillna(2).astype(int)

    df["total_laps_est"] = (df["LapNumber"] / df["RaceProgress"].replace(0, np.nan)).round()
    df["laps_remaining"] = (df["total_laps_est"] - df["LapNumber"]).fillna(0).clip(lower=0)
    df["position_norm"] = df["Position"] / df.groupby("race_id")["Position"].transform("max")

    # ── Grouped temporal features ──────────────────────────────────────────────
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

    # ── Domain-knowledge flags ─────────────────────────────────────────────────
    near_limit = pd.Series(False, index=df.index)
    for enc, limit in TYRE_LIMITS.items():
        near_limit |= (df["compound_enc"] == enc) & (df["TyreLife"] >= limit)
    df["near_tyre_limit"] = near_limit.astype(int)
    df["laps_vs_tyre"] = df["laps_remaining"] - df["TyreLife"]

    # Stint lap counter
    df["stint_lap"] = grp["Stint"].transform(
        lambda x: x.groupby((x != x.shift()).cumsum()).cumcount() + 1
    )

    # ── NEW: деградация на круг ────────────────────────────────────────────────
    # Скорость деградации = накопленная деградация / возраст шины
    df["deg_per_lap"] = df["Cumulative_Degradation"] / df["TyreLife"].replace(0, np.nan)

    # ── NEW: нормализованный возраст шины ─────────────────────────────────────
    # Насколько шина "выработана" относительно лимита для данного компаунда
    tyre_limit_series = df["compound_enc"].map(TYRE_LIMITS).fillna(40)
    df["tyre_life_pct_of_limit"] = df["TyreLife"] / tyre_limit_series

    # ── NEW: глобальная статистика Driver/Race/Compound (не target encoding!) ──
    # Используем только агрегаты не связанные с таргетом — безопасно
    train_mask = df["_split"] == "train"

    # Средний TyreLife по Driver (стиль езды на шинах)
    for col in ["Driver", "Race", "Compound"]:
        means = df[train_mask].groupby(col)["TyreLife"].mean().rename(f"{col}_avg_tyre")
        df[f"{col}_pit_rate"] = df[col].map(means).fillna(means.mean())

    # ── NEW: stint priors — средняя длина стинта по Race+Compound ─────────────
    # Сколько в среднем кругов едут на данном компаунде на данной трассе
    stint_stats = (
        df[train_mask]
        .groupby(["Race", "Compound"])["TyreLife"]
        .agg(avg_tyre_life="mean", std_tyre_life="std")
        .reset_index()
    )
    df = df.merge(stint_stats, on=["Race", "Compound"], how="left")
    df["avg_tyre_life"] = df["avg_tyre_life"].fillna(df["TyreLife"].mean())
    df["std_tyre_life"] = df["std_tyre_life"].fillna(5.0)

    # Насколько текущий TyreLife превышает исторический средний (в сигмах)
    df["tyre_life_zscore"] = (df["TyreLife"] - df["avg_tyre_life"]) / (df["std_tyre_life"] + 1e-6)

    # ── NEW: взаимодействия ───────────────────────────────────────────────────
    df["tyre_x_laps_remaining"] = df["TyreLife"] * df["laps_remaining"]
    df["near_limit_x_laps_remaining"] = df["near_tyre_limit"] * df["laps_remaining"]
    df["deg_x_tyre_life"] = df["deg_per_lap"] * df["TyreLife"]

    # ── Fill NaN ───────────────────────────────────────────────────────────────
    fill_cols = [
        "tyre_life_lag1", "tyre_life_lag2", "tyre_life_lag3",
        "lap_time_delta_pct", "lap_time_roll3", "lap_time_roll5", "lap_time_roll10",
        "pace_degradation", "pace_degradation_10", "gap_pressure",
        "laps_remaining", "laps_vs_tyre",
        "deg_per_lap", "tyre_life_pct_of_limit", "tyre_life_zscore",
        "deg_x_tyre_life",
    ]
    df[fill_cols] = df[fill_cols].fillna(0)
    return df


def split_back(df):
    train = df[df["_split"] == "train"].copy()
    test  = df[df["_split"] == "test"].copy()
    return train, test


def add_fastf1_priors(df: pd.DataFrame, priors_path: str = None) -> pd.DataFrame:
    """
    Джойним leave-one-out priors из FastF1 (2022-2024) по Race + Year.
    Для каждой (Race, Year) prior считался по другим годам — утечки нет.
    """
    from src.config import DATA_DIR
    path = priors_path or DATA_DIR / "fastf1_priors.csv"

    if not Path(str(path)).exists():
        print(f"  [FastF1] priors not found at {path}, skipping")
        return df

    priors = pd.read_csv(path)
    prior_cols = [c for c in priors.columns if c not in ("Race", "Year")]
    print(f"  [FastF1] Loaded priors: {priors.shape}, cols: {prior_cols}")

    df = df.merge(priors, on=["Race", "Year"], how="left")

    for col in prior_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    return df
