# F1 Pit Stop Prediction

[![Solution Note PDF](https://img.shields.io/badge/Solution%20Note-PDF-red?style=for-the-badge&logo=adobeacrobatreader&logoColor=white)](docs/pdf/research_note.pdf)
[![Markdown](https://img.shields.io/badge/Solution%20Note-Markdown-blue?style=for-the-badge)](docs/research_note.md)
[![Kaggle](https://img.shields.io/badge/Kaggle-PS%20S6E5-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/playground-series-s6e5)
[![Result](https://img.shields.io/badge/LB-0.95384%20·%20568%2F3024%20(top%2018.8%25)-success?style=for-the-badge)](https://www.kaggle.com/competitions/playground-series-s6e5)

Kaggle Playground Series S6E5 — binary classification: **will a driver pit on the next lap?**  
Metric: ROC-AUC. Final selected submission: **0.95384 public LB · 568th / 3024 teams · top 18.8%** (beat ~81% of the field).

Start here: **[open the PDF](docs/pdf/research_note.pdf)** · or the [Markdown note](docs/research_note.md) · [Kaggle writeup](https://www.kaggle.com/competitions/playground-series-s6e5/writeups/568th-place-solution-realmlp-original-dataset).

This is a clean, reproducible solution writeup + code — not a notebook dump.

---

## Bottom line

The biggest gain was not feature engineering. It was switching from a tuned GBDT stack to **RealMLP** ([pytabkit](https://github.com/LennartPurucker/pytabkit)), then training with the competition’s **original** 101k-row source dataset.

| Approach | OOF AUC | Public LB |
|---|---|---|
| LGB + XGB + CatBoost baseline | 0.9268 | 0.9384 |
| + original dataset augmentation | 0.9346 | 0.9411 |
| RealMLP `n_ens=5` | 0.9530 | 0.9536 |
| **RealMLP `n_ens=20` (selected)** | **0.9541** | **0.9538** |
| + pseudo-labeling (`thr=0.95`) | 0.9542 | 0.9537 |

What did **not** help: temporal lags/rolling features inside RealMLP, RealMLP↔GBDT blends, pseudo-labeling.

Top solutions used 100–200 model logit stacks. This repo is the opposite bet: one strong tabular NN, done carefully.

---

## Task

- **Input:** lap-level F1 state (tyre compound/life, stint, position, lap time, race progress, …)
- **Target:** `PitNextLap` ∈ {0, 1}
- **Train / test:** ~439k / ~188k rows (synthetic playground data)
- **Original source:** [f1_strategy_dataset_v4.csv](https://www.kaggle.com/datasets/vanshjasuja16/f1-strategy-dataset-pit-stop-prediction) (~101k real-style rows)

Competition: [Predicting F1 Pit Stops](https://www.kaggle.com/competitions/playground-series-s6e5) · longer narrative: [Kaggle writeup](https://www.kaggle.com/competitions/playground-series-s6e5/writeups/568th-place-solution-realmlp-original-dataset)

---

## Approach

### Model

`RealMLP_TD_Classifier` — ensemble of small NNs for tabular data (periodic linear representations, label smoothing, strong preprocessing).

```python
PARAMS = {
    'n_ens': 20,
    'n_epochs': 8,
    'hidden_sizes': [512, 256, 128],
    'act': 'silu',
    'lr': 0.019,
    'plr_hidden_1': 16, 'plr_hidden_2': 8,
    'ls_eps': 0.01,
    'tfms': ['one_hot', 'median_center', 'robust_scale',
             'smooth_clip', 'embedding', 'l2_normalize'],
}
```

Full dict lives in [`train_realmlp.py`](train_realmlp.py). Feature recipe follows [yekenot’s public notebook](https://www.kaggle.com/code/yekenot/f1-pitstop-realmlp).

### Training setup

- **StratifiedKFold (5)** — same races can appear in train and val; this is intentional so Target Encoding works
- **Original dataset** appended to each fold’s training split
- **TargetEncoder** (`cv=5`) on `Race×Compound` and `Race×Year`

### Feature engineering (high level)

- Arithmetic interactions: `LapNumber/RaceProgress`, `TyreLife/LapNumber`, `LapTime × CumulativeDegradation`
- Floor-factorized categorical embeddings for numerical columns
- Count encoding for categoricals
- Quantile bins: RaceProgress (200), LapTime (7)
- Combo categories → target-encoded

---

## Quick start

```bash
git clone https://github.com/ron1n97/f1-pitstop-prediction
cd f1-pitstop-prediction
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# needs ~/.kaggle/kaggle.json
python download_data.py          # competition CSVs + original dataset

# main model (~100 min on CPU; force CPU — MPS hangs with pytabkit)
python train_realmlp.py --no-submit
```

Optional experiments:

```bash
python train_with_orig.py --no-submit              # GBDT baseline + orig
python train_realmlp_v2.py --no-submit             # temporal FE (worse)
python train_realmlp_pseudo.py --no-submit --threshold 0.95
```

---

## Repository layout

```
docs/
  research_note.md        # solution note (start here)
  pdf/research_note.pdf   # same note as PDF with figures
  experiment_log.md       # what was tried and dropped
figures/                  # placement, LB ladder, ROC, folds, …
scripts/make_figures.py   # regenerate figures from OOF + logs
train_realmlp.py          # main pipeline (StratKFold + orig + TE)
train_realmlp_pseudo.py   # pseudo-labeling experiment
train_realmlp_v2.py       # temporal features in RealMLP (did not help)
train_with_orig.py        # LGB+XGB+CatBoost + orig
feature_selection.py      # LGB proxy / importance checks
download_data.py          # Kaggle competition + original dataset
src/                      # shared config, GBDT/LSTM helpers from early work
```

`data/`, `models/`, `submissions/`, and `logs/` are gitignored — download locally and train.  
Figures need a local `data/train.csv` + `submissions/oof_realmlp.csv` (produced by training).

---

## Key lessons

1. **Architecture > clever FE (here).** A day of temporal GBDT features plateaued at ~0.941 LB. RealMLP jumped to ~0.954 with mostly public FE.
2. **StratifiedKFold vs GroupKFold is a product decision.** Grouping by race is “cleaner,” but it blocks useful Target Encoding. Stratified won on LB.
3. **Original data is free signal.** Always check for a source dataset in Playground comps.
4. **Pseudo-labels and blends can flatter OOF.** Confident negatives (~115k) nudged OOF up and LB down; RealMLP/GBDT correlation ~0.92 left little blend juice.

More detail: [`docs/experiment_log.md`](docs/experiment_log.md).
