# F1 Pit Stop Prediction

**Kaggle Playground Series S6E5** — binary classification: will a driver pit next lap?  
Metric: ROC-AUC | Final result: **0.95384 LB, 568th/3024 (top 19%)**

---

## What worked

The single biggest jump came from switching to **RealMLP** (pytabkit) instead of a tuned GBDT ensemble. One model swap: +0.013 LB.

| Approach | OOF AUC | LB AUC |
|---|---|---|
| LGB + XGB + CatBoost baseline | 0.9268 | 0.9384 |
| + original dataset augmentation | 0.9346 | 0.9411 |
| RealMLP n_ens=5 | 0.9530 | 0.9536 |
| **RealMLP n_ens=20** | **0.9541** | **0.9538** |
| + pseudo-labeling (thr=0.95) | 0.9542 | 0.9537 |

What didn't help: temporal features (lag/rolling), blending RealMLP with GBDT, pseudo-labeling.

---

## Approach

### Model

`RealMLP_TD_Classifier` from [pytabkit](https://github.com/LennartPurucker/pytabkit) — an ensemble of small neural networks optimized for tabular data.

```python
PARAMS = {
    'n_ens': 20,
    'n_epochs': 5,
    'hidden_sizes': [512, 256, 128],
    'act': 'silu',
    'lr': 0.019,
    'plr_hidden_1': 16, 'plr_hidden_2': 8,  # periodic linear representations
    'ls_eps': 0.01,                           # label smoothing
    'tfms': ['one_hot', 'median_center', 'robust_scale', 'smooth_clip', 'embedding', 'l2_normalize'],
}
```

### Training setup

- **StratifiedKFold** (5 folds) — same races in train and val, which makes Target Encoding safe
- **Original dataset** (101k rows from `f1_strategy_dataset_v4.csv`) added to each fold's training split
- **TargetEncoder** (cv=5) on Race×Compound and Race×Year interaction features

### Feature engineering

Based on [yekenot's public notebook](https://www.kaggle.com/code/yekenot/f1-pitstop-realmlp):

- Arithmetic interactions: `LapNumber/RaceProgress`, `TyreLife/LapNumber`, `LapTime * CumulativeDegradation`
- Every numerical column gets floor-factorized into a categorical embedding column
- Count encoding for categorical columns
- Quantile binning: RaceProgress (200 bins), LapTime (7 bins)
- Combo categories: Race_Compound, Race_Year → Target Encoded

---

## Quick start

```bash
# 1. Clone and install
git clone https://github.com/ron1n97/f1-pitstop-prediction
cd f1-pitstop-prediction
pip install -r requirements.txt

# 2. Download data (requires ~/.kaggle/kaggle.json)
python download_data.py

# 3. Train RealMLP (main model, ~100 min on CPU)
python train_realmlp.py --no-submit

# 4. Train with pseudo-labeling
python train_realmlp_pseudo.py --no-submit --threshold 0.95
```

**Note:** pytabkit hangs on Apple Silicon MPS. Force CPU:
```python
os.environ["CUDA_VISIBLE_DEVICES"] = ""
```

---

## Project structure

```
train_realmlp.py          main pipeline (StratKFold + orig + TE)
train_realmlp_pseudo.py   + pseudo-labeling from confident test predictions
train_realmlp_v2.py       experiment: temporal features in RealMLP (didn't help)
train_with_orig.py        GBDT ensemble (LGB+XGB+CatBoost) with orig data
feature_selection.py      LGB proxy for feature importance analysis
src/
  config.py               paths, constants, compound maps
submissions/              saved prediction files
```

---

## Key lessons

**Architecture > feature engineering.** I spent most of my time building temporal features (lags, rolling means, degradation rates) for GBDT. In RealMLP they made things worse — each new numerical feature creates an extra categorical embedding column, and the noise outweighed the signal.

**StratifiedKFold vs GroupKFold.** Top notebooks use StratifiedKFold even though it puts the same races in train and val. This isn't leakage for the model itself, but it does enable Target Encoding — which GroupKFold would make impossible. The tradeoff is worth it.

**Original dataset is free signal.** Always check if the competition provides a source dataset. Adding 101k rows to training with no additional risk gave a consistent OOF boost across all folds.
