#!/usr/bin/env python3
"""Regenerate solution-note figures from competition logs + OOF predictions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
DATA = ROOT / "data"
OOF = ROOT / "submissions" / "oof_realmlp.csv"

# Competition outcome (selected submission)
RANK = 568
N_TEAMS = 3024
PUBLIC_LB = 0.95384
OOF_SELECTED = 0.95407  # RealMLP n_ens=20 overall OOF from logs/realmlp_n20.log

# Ladder from README / logs (public LB where known)
LADDER = [
    ("GBDT baseline", 0.9268, 0.9384),
    ("+ original data", 0.9346, 0.9411),
    ("RealMLP n_ens=5", 0.9530, 0.9536),
    ("RealMLP n_ens=20\n(selected)", 0.9541, 0.9538),
    ("+ pseudo-labels", 0.9542, 0.9537),
]

FOLDS = [0.95513, 0.95305, 0.95392, 0.95329, 0.95499]  # realmlp_n20.log


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
        }
    )


def placement() -> None:
    pct_from_top = 100.0 * RANK / N_TEAMS
    beat = 100.0 * (N_TEAMS - RANK) / N_TEAMS
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))

    ax = axes[0]
    ax.barh([0], [N_TEAMS], color="#d9d9d9", height=0.55, label="All teams")
    ax.barh([0], [RANK], color="#1f4e79", height=0.55, label=f"Rank #{RANK}")
    ax.set_xlim(0, N_TEAMS)
    ax.set_yticks([])
    ax.set_xlabel("Team rank (1 = best)")
    ax.set_title(f"Placement: #{RANK} / {N_TEAMS}")
    ax.axvline(RANK, color="#c0392b", ls="--", lw=1.2)
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[1]
    wedges, _, autotexts = ax.pie(
        [RANK, N_TEAMS - RANK],
        labels=[f"Ahead of you\n({RANK})", f"Teams you beat\n({N_TEAMS - RANK})"],
        colors=["#d9d9d9", "#2e7d32"],
        autopct="%1.1f%%",
        startangle=90,
        explode=(0.02, 0.02),
        textprops={"fontsize": 9},
    )
    for t in autotexts:
        t.set_fontsize(9)
    ax.set_title(f"Top {pct_from_top:.1f}%  ·  beat {beat:.1f}% of field")

    fig.suptitle(
        f"Kaggle PS S6E5 — public LB {PUBLIC_LB:.5f}",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIG / "placement.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def lb_ladder() -> None:
    labels = [r[0] for r in LADDER]
    oof = [r[1] for r in LADDER]
    lb = [r[2] for r in LADDER]
    x = np.arange(len(LADDER))
    w = 0.36

    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    b1 = ax.bar(x - w / 2, oof, w, label="OOF AUC", color="#4c78a8")
    b2 = ax.bar(x + w / 2, lb, w, label="Public LB AUC", color="#f58518")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0.92, 0.958)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Score ladder — architecture jump dominates FE")
    ax.axhline(PUBLIC_LB, color="#c0392b", ls="--", lw=1, label=f"Selected LB {PUBLIC_LB:.5f}")
    ax.legend(loc="lower right", fontsize=8)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.0004,
                f"{h:.4f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    fig.tight_layout()
    fig.savefig(FIG / "lb_ladder.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fold_auc() -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    folds = np.arange(1, 6)
    ax.bar(folds, FOLDS, color="#4c78a8", width=0.6)
    ax.axhline(OOF_SELECTED, color="#c0392b", ls="--", label=f"Overall OOF {OOF_SELECTED:.5f}")
    ax.set_ylim(0.950, 0.957)
    ax.set_xlabel("Fold")
    ax.set_ylabel("AUC")
    ax.set_title("RealMLP n_ens=20 — per-fold OOF AUC")
    for i, v in enumerate(FOLDS):
        ax.text(i + 1, v + 0.00015, f"{v:.5f}", ha="center", fontsize=8)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fold_auc.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def load_oof() -> tuple[np.ndarray, np.ndarray]:
    train = pd.read_csv(DATA / "train.csv", usecols=["id", "PitNextLap"])
    oof = pd.read_csv(OOF)
    m = train.merge(oof, on="id", suffixes=("_y", "_p"))
    y = m["PitNextLap_y"].to_numpy()
    p = m["PitNextLap_p"].to_numpy()
    return y, p


def roc() -> None:
    y, p = load_oof()
    fpr, tpr, _ = roc_curve(y, p)
    auc = roc_auc_score(y, p)

    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.plot(fpr, tpr, color="#1f4e79", lw=2, label=f"RealMLP OOF (AUC={auc:.5f})")
    ax.plot([0, 1], [0, 1], color="#999", ls="--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("OOF ROC curve — selected RealMLP")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "roc_oof.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def reliability() -> None:
    y, p = load_oof()
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=15, strategy="quantile")

    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.plot([0, 1], [0, 1], color="#999", ls="--", lw=1, label="Perfect")
    ax.plot(mean_pred, frac_pos, "o-", color="#1f4e79", lw=1.5, label="RealMLP OOF")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Empirical positive rate")
    ax.set_title("Reliability diagram (quantile bins)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "reliability_oof.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def score_path() -> None:
    """Compact path graphic: what helped vs what did not."""
    steps = [
        ("Temporal FE\nfor GBDT", "≈0", "#bdbdbd"),
        ("GBDT + orig", "+0.003 LB", "#4c78a8"),
        ("→ RealMLP", "+0.013 LB", "#2e7d32"),
        ("n_ens 5→20", "+0.0002 LB", "#4c78a8"),
        ("Pseudo / blend", "≤0 / −LB", "#c0392b"),
    ]
    fig, ax = plt.subplots(figsize=(8.0, 2.8))
    ax.set_xlim(0, len(steps) + 0.2)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Research path — where the AUC actually moved", loc="left", fontsize=11)
    for i, (title, delta, color) in enumerate(steps):
        x = i + 0.5
        ax.add_patch(
            plt.Rectangle((x - 0.38, 0.28), 0.76, 0.44, facecolor=color, alpha=0.85, ec="none")
        )
        ax.text(x, 0.50, title, ha="center", va="center", color="white", fontsize=8, fontweight="bold")
        ax.text(x, 0.14, delta, ha="center", va="center", fontsize=8, color="#333")
        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(x + 0.55, 0.50),
                xytext=(x + 0.42, 0.50),
                arrowprops=dict(arrowstyle="->", color="#666"),
            )
    fig.tight_layout()
    fig.savefig(FIG / "research_path.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def pred_hist() -> None:
    y, p = load_oof()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.hist(p[y == 0], bins=50, alpha=0.65, density=True, label="No pit (y=0)", color="#4c78a8")
    ax.hist(p[y == 1], bins=50, alpha=0.65, density=True, label="Pit next lap (y=1)", color="#f58518")
    ax.set_xlabel("Predicted P(PitNextLap)")
    ax.set_ylabel("Density")
    ax.set_title("OOF score distribution by class")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "score_by_class.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    style()
    placement()
    lb_ladder()
    fold_auc()
    roc()
    reliability()
    score_path()
    pred_hist()
    print(f"Wrote figures to {FIG}")


if __name__ == "__main__":
    main()
