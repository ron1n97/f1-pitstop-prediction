# Experiment log

Short record of what moved the needle and what burned time. Scores are public LB unless noted.

## Kept

| Idea | Effect | Notes |
|---|---|---|
| RealMLP (`pytabkit`) instead of GBDT stack | **+~0.013 LB** | Largest single jump; `n_ens=20` selected |
| Append original `f1_strategy_dataset_v4` (~101k) | +~0.003 LB on GBDT; also used in RealMLP folds | Fold-wise concat into train split |
| StratifiedKFold + Target Encoding on Race×Compound / Race×Year | Enabled TE; better than GroupKFold for this recipe | Same races can land in train and val |
| yekenot-style FE (interactions, floor cats, quantile bins, counts) | Baseline for RealMLP | Did not reinvent from scratch |

## Dropped / negative

| Idea | Effect | Notes |
|---|---|---|
| Temporal FE (lags, rolling lap times, pace degradation) | ≈0 for LGB proxy; **hurt** RealMLP OOF (~0.953→0.952) | Extra nums → extra noisy embeddings |
| RealMLP + GBDT blends (90/10 … 70/30) | All ≤ pure RealMLP | Pred correlation ~0.916 |
| Pseudo-labeling (p>0.95 or p<0.05) | OOF ↑, LB ↓ | ~119k rows added; ~115k confident negatives |
| Heavy Optuna on LGB before architecture search | Plateau ~0.941 | Wrong bottleneck |

## Infrastructure notes

- Apple Silicon: set `CUDA_VISIBLE_DEVICES=""` — pytabkit + MPS hung in practice
- Runtime: ~20 min/fold, ~100 min full 5-fold RealMLP `n_ens=20` on CPU (M-series)
- Final selected file: RealMLP `n_ens=20` submission (public **0.95384**)

## Would do differently next Playground

1. Start with RealMLP / TabM on day one (and a free Kaggle T4)
2. Confirm the model family before investing in exotic temporal FE
3. Keep one “boring” submission and one “greedy” blend slot when the LB is ultra-tight
