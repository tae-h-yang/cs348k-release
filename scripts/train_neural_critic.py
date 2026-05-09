"""
Neural Surrogate Critic — GPU training.

Replaces the slow MuJoCo inverse-dynamics heuristic critic with a fast 1D CNN
that maps a qpos window → risk score.  Target: Spearman ρ > 0.85 at < 0.5ms
inference on the RTX 4060.

Architecture: 4-layer causal 1D CNN, <200k params.
Labels: heuristic risk scores computed from the same clips used for best-of-K.
Training set: sliding windows from all 234 labeled .npy clips (156 WC + 78 PS).

Usage:
    conda run -n base python scripts/train_neural_critic.py

Outputs:
    results/neural_critic/
        model.pt          — trained weights
        train_log.csv     — epoch losses
        validation.png    — pred vs true scatter + loss curve
        config.json       — window size, architecture hyper-params
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from physics_eval.online_critic import OnlineSegmentCritic

REPO = Path(__file__).parent.parent
DATA_DIRS = [REPO / "data" / "guided_ablation", REPO / "data" / "steered_ablation"]
OUT_DIR = REPO / "results" / "neural_critic"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ── Hyper-parameters ──────────────────────────────────────────────────────────
WINDOW = 32        # frames per training sample
STRIDE = 4         # stride for sliding window (overlap = WINDOW - STRIDE)
QPOS_DIM = 36      # input channels (full qpos)
HIDDEN = 64        # CNN base channels
BATCH = 256
EPOCHS = 80
LR = 3e-4
WEIGHT_DECAY = 1e-4
VAL_FRAC = 0.15

# ── Dataset ───────────────────────────────────────────────────────────────────

class RiskWindowDataset(Dataset):
    """Sliding-window qpos segments with heuristic risk labels."""

    def __init__(self, critic: OnlineSegmentCritic, data_dirs, window=WINDOW, stride=STRIDE):
        self.windows: list[np.ndarray] = []
        self.labels: list[float] = []

        npy_files = []
        for d in data_dirs:
            npy_files.extend(sorted(d.glob("*.npy")))

        print(f"Labeling {len(npy_files)} clips with heuristic critic...")
        skipped = 0
        for path in npy_files:
            qpos = np.load(path).astype(np.float64)
            T = len(qpos)
            if T < window:
                skipped += 1
                continue
            for start in range(0, T - window + 1, stride):
                seg = qpos[start : start + window]
                risk = critic.score_segment(seg)
                if not np.isfinite(risk):
                    continue
                self.windows.append(seg.astype(np.float32))
                self.labels.append(float(risk))

        print(f"  Clips: {len(npy_files) - skipped} usable / {len(npy_files)} total")
        print(f"  Windows: {len(self.windows)}  (label range: {min(self.labels):.1f}–{max(self.labels):.1f})")

        # Log-transform labels: right-skewed risk scores (most near 0, few huge)
        arr = np.array(self.labels, dtype=np.float32)
        self.labels_log = np.log1p(arr)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        # qpos: (WINDOW, 36) → transpose to (36, WINDOW) for 1D conv over time
        x = torch.from_numpy(self.windows[idx]).T   # (36, WINDOW)
        y = torch.tensor(self.labels_log[idx])
        return x, y


# ── Model ─────────────────────────────────────────────────────────────────────

class NeuralCritic(nn.Module):
    """
    4-layer 1D CNN.  Input: (B, 36, WINDOW).  Output: (B,) log-risk scalar.

    Causal design (no future frames visible) via left-padding only, so this
    can run online (segment by segment) at deployment time.
    """

    def __init__(self, in_ch=QPOS_DIM, hidden=HIDDEN, window=WINDOW):
        super().__init__()
        self.net = nn.Sequential(
            # Layer 1: temporal feature extraction
            nn.Conv1d(in_ch, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            # Layer 2: wider context
            nn.Conv1d(hidden, hidden * 2, kernel_size=5, padding=4, dilation=2),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(),
            # Layer 3: compress time
            nn.Conv1d(hidden * 2, hidden * 2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(),
            # Layer 4: further compress
            nn.Conv1d(hidden * 2, hidden, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),   # global average pool → (B, hidden, 1)
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):                # x: (B, 36, W)
        return self.head(self.net(x)).squeeze(-1)   # (B,)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    critic = OnlineSegmentCritic()
    dataset = RiskWindowDataset(critic, DATA_DIRS)

    n_val = max(1, int(len(dataset) * VAL_FRAC))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val],
                                      generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)

    model = NeuralCritic().to(DEVICE)
    print(f"Model params: {count_params(model):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    loss_fn = nn.HuberLoss(delta=1.0)

    train_log = []
    best_val_loss = float("inf")
    best_rho = -1.0

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(x)
        train_loss /= n_train
        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0.0
        preds_all, labels_all = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x)
                val_loss += loss_fn(pred, y).item() * len(x)
                preds_all.extend(pred.cpu().numpy())
                labels_all.extend(y.cpu().numpy())
        val_loss /= n_val
        rho, _ = spearmanr(preds_all, labels_all)

        train_log.append({"epoch": epoch, "train_loss": train_loss,
                          "val_loss": val_loss, "spearman_rho": rho})

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}: train={train_loss:.4f}  val={val_loss:.4f}  rho={rho:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_rho = rho
            torch.save(model.state_dict(), OUT_DIR / "model.pt")

    print(f"\nBest val_loss={best_val_loss:.4f}  best_rho={best_rho:.3f}")
    print(f"Saved: {OUT_DIR / 'model.pt'}")

    # Save training log
    with open(OUT_DIR / "train_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch","train_loss","val_loss","spearman_rho"])
        w.writeheader(); w.writerows(train_log)

    # Config
    config = {"window": WINDOW, "stride": STRIDE, "qpos_dim": QPOS_DIM, "hidden": HIDDEN,
              "n_params": count_params(model), "best_val_loss": best_val_loss,
              "best_spearman_rho": best_rho}
    with open(OUT_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ── Inference timing ──────────────────────────────────────────────────────
    model.load_state_dict(torch.load(OUT_DIR / "model.pt", map_location=DEVICE))
    model.eval()
    dummy = torch.randn(1, QPOS_DIM, WINDOW, device=DEVICE)
    # Warm up
    for _ in range(20):
        with torch.no_grad():
            model(dummy)
    # Time 1000 calls
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    N_TIMING = 1000
    with torch.no_grad():
        for _ in range(N_TIMING):
            model(dummy)
    torch.cuda.synchronize()
    ms_per_call = 1000 * (time.perf_counter() - t0) / N_TIMING
    print(f"Inference: {ms_per_call:.3f} ms/segment on {DEVICE}")
    config["inference_ms"] = ms_per_call
    with open(OUT_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ── Validation plot ───────────────────────────────────────────────────────
    model.eval()
    all_pred_log, all_true_log = [], []
    with torch.no_grad():
        for x, y in val_loader:
            all_pred_log.extend(model(x.to(DEVICE)).cpu().numpy())
            all_true_log.extend(y.numpy())
    all_pred_log = np.array(all_pred_log)
    all_true_log = np.array(all_true_log)
    all_pred = np.expm1(all_pred_log)
    all_true = np.expm1(all_true_log)
    rho_final, _ = spearmanr(all_pred, all_true)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.scatter(all_true, all_pred, alpha=0.4, s=15, color="#1565C0")
    m = max(all_true.max(), all_pred.max()) * 1.05
    ax.plot([0, m], [0, m], "r--", lw=1, label="Perfect")
    ax.set_xlabel("Heuristic risk (true)", fontsize=11)
    ax.set_ylabel("Neural critic prediction", fontsize=11)
    ax.set_title(f"Neural Critic Validation (Spearman ρ={rho_final:.3f})", fontsize=12)
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    epochs_arr = [r["epoch"] for r in train_log]
    ax.plot(epochs_arr, [r["train_loss"] for r in train_log], label="Train")
    ax.plot(epochs_arr, [r["val_loss"] for r in train_log], label="Val")
    ax2 = ax.twinx()
    ax2.plot(epochs_arr, [r["spearman_rho"] for r in train_log],
             color="green", linestyle="--", label="Spearman ρ")
    ax2.set_ylabel("Spearman ρ", color="green", fontsize=10)
    ax2.tick_params(axis="y", labelcolor="green")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Huber Loss"); ax.set_title("Training Curves")
    ax.legend(loc="upper right"); ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "validation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_DIR / 'validation.png'}")

    print(f"\n{'='*50}")
    print(f"NEURAL CRITIC SUMMARY")
    print(f"  Spearman ρ (val):  {rho_final:.3f}  (target ≥ 0.85)")
    print(f"  Inference time:    {ms_per_call:.3f} ms  (target < 0.5 ms)")
    print(f"  Parameters:        {count_params(model):,}")
    print(f"  Best val loss:     {best_val_loss:.4f}")
    print(f"  Saved to:          {OUT_DIR}")


if __name__ == "__main__":
    train()
