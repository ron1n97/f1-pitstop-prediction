import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.config import (
    SEQ_FEATURES, TARGET, N_FOLDS,
    LSTM_HIDDEN, LSTM_LAYERS, LSTM_DROPOUT,
    LSTM_LR, LSTM_WEIGHT_DECAY, LSTM_EPOCHS, LSTM_BATCH,
    MODELS_DIR,
)
from src.dataset import PitStopDataset


class BiLSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


def _train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def _predict(model, loader, device) -> np.ndarray:
    model.eval()
    preds = []
    for x, _ in loader:
        x = x.to(device)
        logits = model(x)
        preds.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(preds)


def train_lstm(df, verbose: bool = True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    neg = (df[TARGET] == 0).sum()
    pos = (df[TARGET] == 1).sum()
    pos_weight = torch.tensor([neg / pos], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    oof = np.zeros(len(df))
    saved_models: list[str] = []
    gkf = GroupKFold(n_splits=N_FOLDS)
    y = df[TARGET].astype(int).values
    groups = df["race_id"].values

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(df, y, groups)):
        tr_df = df.iloc[tr_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        tr_loader = DataLoader(PitStopDataset(tr_df), batch_size=LSTM_BATCH, shuffle=True, num_workers=0)
        val_loader = DataLoader(PitStopDataset(val_df), batch_size=LSTM_BATCH * 2, shuffle=False, num_workers=0)

        model = BiLSTMClassifier(
            input_size=len(SEQ_FEATURES),
            hidden_size=LSTM_HIDDEN,
            num_layers=LSTM_LAYERS,
            dropout=LSTM_DROPOUT,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LSTM_LR, weight_decay=LSTM_WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=LSTM_EPOCHS)

        best_auc = 0.0
        best_state = None
        patience_cnt = 0

        for epoch in range(LSTM_EPOCHS):
            _train_one_epoch(model, tr_loader, optimizer, criterion, device)
            scheduler.step()

            val_preds = _predict(model, val_loader, device)
            val_labels = val_df[TARGET].values
            auc = roc_auc_score(val_labels, val_preds)

            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= 7:  # early stop after 7 epochs no improvement
                    break

        model.load_state_dict(best_state)
        val_preds = _predict(model, val_loader, device)
        oof[val_idx] = val_preds

        path = str(MODELS_DIR / f"lstm_fold{fold}.pt")
        torch.save(model.state_dict(), path)
        saved_models.append(path)

        if verbose:
            print(f"  [LSTM] Fold {fold + 1}/{N_FOLDS}  AUC={best_auc:.4f}")

    oof_auc = roc_auc_score(df[TARGET], oof)
    if verbose:
        print(f"  [LSTM] OOF AUC={oof_auc:.4f}")

    return oof, saved_models


def predict_lstm(test_df, saved_models: list[str]) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader = DataLoader(PitStopDataset(test_df), batch_size=1024, shuffle=False, num_workers=0)
    all_preds = []

    for path in saved_models:
        model = BiLSTMClassifier(
            input_size=len(SEQ_FEATURES),
            hidden_size=LSTM_HIDDEN,
            num_layers=LSTM_LAYERS,
            dropout=LSTM_DROPOUT,
        ).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        all_preds.append(_predict(model, loader, device))

    return np.mean(all_preds, axis=0)
