from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
SUBMISSIONS_DIR = ROOT / "submissions"

for d in (DATA_DIR, MODELS_DIR, SUBMISSIONS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── Real column names from the competition CSV ─────────────────────────────────
# Train/test columns: id, Driver, Compound, Race, Year, PitStop, LapNumber,
#   Stint, TyreLife, Position, LapTime (s), LapTime_Delta,
#   Cumulative_Degradation, RaceProgress, Position_Change
# Target (train only): PitNextLap

TARGET = "PitNextLap"

COMPOUND_MAP = {"SOFT": 0, "MEDIUM": 1, "HARD": 2, "INTERMEDIATE": 3, "WET": 4}

# Tyre life thresholds (domain knowledge) by compound_enc
TYRE_LIMITS = {0: 25, 1: 38, 2: 52, 3: 18, 4: 15}

GBDT_FEATURES = [
    # Raw
    "TyreLife", "compound_enc", "Stint", "PitStop",
    "LapNumber", "Position", "RaceProgress", "Position_Change",
    "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
    # Temporal
    "tyre_life_lag1", "tyre_life_lag2", "tyre_life_lag3",
    "lap_time_roll3", "lap_time_roll5", "lap_time_roll10",
    "pace_degradation", "pace_degradation_10",
    "lap_time_delta_pct",
    "laps_remaining", "laps_vs_tyre",
    "near_tyre_limit", "position_norm",
    "gap_pressure", "stint_lap",
    # Degradation rate
    "deg_per_lap",
]

SEQ_FEATURES = [
    "TyreLife", "lap_time_delta_pct", "Position", "Cumulative_Degradation", "pace_degradation",
]
SEQ_LEN = 5

# ── Model hyperparameters ──────────────────────────────────────────────────────

N_FOLDS = 5
RANDOM_SEED = 42

LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "n_estimators": 1000,
    "verbose": -1,
    "random_state": RANDOM_SEED,
}

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_estimators": 1000,
    "random_state": 7,
    "tree_method": "hist",
    "verbosity": 0,
}

CAT_PARAMS = {
    "iterations": 1000,
    "learning_rate": 0.05,
    "depth": 6,
    "eval_metric": "AUC",
    "random_seed": 13,
    "verbose": 0,
    "early_stopping_rounds": 50,
}

LSTM_HIDDEN = 64
LSTM_LAYERS = 2
LSTM_DROPOUT = 0.3
LSTM_LR = 1e-3
LSTM_WEIGHT_DECAY = 1e-4
LSTM_EPOCHS = 30
LSTM_BATCH = 512
