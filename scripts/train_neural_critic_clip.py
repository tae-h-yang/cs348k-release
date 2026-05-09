"""
Clip-level neural critic (fixes R5 from reviewer).

The segment-level neural critic (train_neural_critic.py) fails as a WC-K
selector because it's trained on 32-frame window labels and applied to full
200-frame clips via max-window aggregation (31% agreement with heuristic).

This model operates on full clips:
  Input: (N, 36, T) where T is variable → pooled to fixed repr
  Output: scalar log-risk score for the entire clip

Architecture: same 1D CNN but with global avg pool, applied to full clip T.
Training: full-clip labels from a guided-ablation CSV. The default remains the
original 39 K=1 clips, but --csv/--data_dir can point at the 210-row extended
K=1/K=8 run.

Outputs: results/neural_critic_clip/
"""
from __future__ import annotations
import argparse
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

REPO = Path(__file__).parent.parent
DATA_DIR = REPO / "data" / "guided_ablation"
CSV_PATH = REPO / "results" / "guided_ablation_full.csv"
OUT_DIR = REPO / "results" / "neural_critic_clip"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

QPOS_DIM = 36
HIDDEN = 64
BATCH = 32
EPOCHS = 100
LR = 3e-4
WEIGHT_DECAY = 1e-4
VAL_FRAC = 0.2

# ─── Dataset ─────────────────────────────────────────────────────────────────

class ClipDataset(Dataset):
    def __init__(self, csv_path: Path, data_dir: Path, k_filter: int | None):
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))

        self.clips = []
        self.labels = []
        skipped = 0

        for row in rows:
            k = int(row["K"])
            if k_filter is not None and k != k_filter:
                continue
            clip_name = row["clip"]
            full_risk = float(row["full_risk"])
            if not np.isfinite(full_risk):
                skipped += 1
                continue

            npy_path = data_dir / f"{clip_name}_K{k}.npy"
            if not npy_path.exists():
                skipped += 1
                continue

            qpos = np.load(npy_path).astype(np.float32)
            self.clips.append(qpos)
            self.labels.append(float(np.log1p(full_risk)))

        print(f"Loaded {len(self.clips)} clips from {csv_path} (skipped {skipped})")
        arr = np.array(self.labels)
        print(f"  Label range: {np.expm1(arr.min()):.1f}–{np.expm1(arr.max()):.1f}")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.clips[idx]).T  # (36, T)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


def collate_variable_length(batch):
    """Pad/trim to max length within batch."""
    xs, ys = zip(*batch)
    max_T = max(x.shape[1] for x in xs)
    padded = torch.zeros(len(xs), QPOS_DIM, max_T)
    for i, x in enumerate(xs):
        T = x.shape[1]
        padded[i, :, :T] = x
    return padded, torch.stack(ys)


# ─── Model ────────────────────────────────────────────────────────────────────

class ClipLevelCritic(nn.Module):
    """
    1D CNN over full clip. Variable T handled by AdaptiveAvgPool1d(1).
    Same architecture as segment model but no fixed window assumption.
    """
    def __init__(self, in_ch=QPOS_DIM, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kernel_size=9, padding=4),
            nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Conv1d(hidden, hidden*2, kernel_size=7, padding=6, dilation=2),
            nn.BatchNorm1d(hidden*2), nn.ReLU(),
            nn.Conv1d(hidden*2, hidden*2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden*2), nn.ReLU(),
            nn.Conv1d(hidden*2, hidden*2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden*2), nn.ReLU(),
            nn.Conv1d(hidden*2, hidden, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):  # (B, 36, T)
        return self.head(self.net(x)).squeeze(-1)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─── Training ─────────────────────────────────────────────────────────────────

def write_train_log(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def train(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = ClipDataset(Path(args.csv), Path(args.data_dir), args.k_filter)
    if len(dataset) < 10:
        print("Not enough clips, aborting.")
        return

    n_val = max(1, int(len(dataset) * VAL_FRAC))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val],
                                      generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True,
                              num_workers=0, collate_fn=collate_variable_length)
    val_loader = DataLoader(val_set, batch_size=BATCH, shuffle=False,
                            num_workers=0, collate_fn=collate_variable_length)

    model = ClipLevelCritic().to(DEVICE)
    print(f"Model params: {count_params(model):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.HuberLoss(delta=1.0)

    best_val_loss = float("inf")
    best_rho = -1.0
    train_log = []

    start = time.time()
    for epoch in range(1, args.epochs + 1):
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

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}: train={train_loss:.4f}  val={val_loss:.4f}  rho={rho:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_rho = rho
            torch.save(model.state_dict(), out_dir / "model.pt")

    print(f"\nBest val_loss={best_val_loss:.4f}  best_rho={best_rho:.3f}")
    write_train_log(out_dir / "train_log.csv", train_log)

    # Config
    config = {"n_params": count_params(model), "best_val_loss": best_val_loss,
              "best_spearman_rho": best_rho, "model_type": "clip_level",
              "csv": str(args.csv), "data_dir": str(args.data_dir),
              "k_filter": args.k_filter, "n_clips": len(dataset),
              "epochs": args.epochs, "train_seconds": time.time() - start}
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Plot
    model.load_state_dict(torch.load(out_dir / "model.pt", map_location=DEVICE))
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
    rho_f, _ = spearmanr(all_pred, all_true)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.scatter(all_true, all_pred, alpha=0.6, s=40, color="#1565C0")
    m = max(all_true.max(), all_pred.max()) * 1.05
    ax.plot([0, m], [0, m], "r--", lw=1, label="Perfect")
    ax.set_xlabel("Heuristic risk (true)"); ax.set_ylabel("Clip-level neural prediction")
    ax.set_title(f"Clip-Level Neural Critic (ρ={rho_f:.3f})")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    epochs_arr = [r["epoch"] for r in train_log]
    ax.plot(epochs_arr, [r["train_loss"] for r in train_log], label="Train")
    ax.plot(epochs_arr, [r["val_loss"] for r in train_log], label="Val")
    ax2 = ax.twinx()
    ax2.plot(epochs_arr, [r["spearman_rho"] for r in train_log],
             color="green", linestyle="--", label="ρ")
    ax2.set_ylabel("Spearman ρ", color="green")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Huber Loss")
    ax.set_title("Training Curves (Clip-Level)")
    ax.legend(loc="upper right"); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "validation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n{'='*50}")
    print(f"CLIP-LEVEL NEURAL CRITIC")
    print(f"  Spearman ρ: {rho_f:.3f}  (target ≥ 0.85)")
    print(f"  Val loss:   {best_val_loss:.4f}")
    print(f"  Params:     {count_params(model):,}")
    print(f"  Saved to:   {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--data_dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument(
        "--k_filter",
        type=int,
        default=1,
        help="Only train on this K value. Use -1 to include all K values.",
    )
    parsed = parser.parse_args()
    if parsed.k_filter < 0:
        parsed.k_filter = None
    train(parsed)
