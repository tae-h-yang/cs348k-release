"""Larger clip-level neural critic trained on all available labeled qpos clips.

This is a stronger attempt than train_neural_critic_clip.py:

- combines guided K sweep, 105-identity extension, candidate audit, and WC/PS
  steering labels,
- de-duplicates exact qpos arrays,
- splits by clip identity to reduce leakage,
- trains a wider residual temporal CNN with attention pooling,
- writes predictions, train log, validation plot, and config.

It still learns the heuristic risk score, not real physical execution.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
QPOS_DIM = 36


@dataclass(frozen=True)
class ClipRecord:
    key: str
    group: str
    path: Path
    risk: float
    source: str


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def collect_records() -> list[ClipRecord]:
    records: list[ClipRecord] = []

    for row in _rows(ROOT / "results" / "guided_ablation_full.csv"):
        k = int(row["K"])
        clip = row["clip"]
        path = ROOT / "data" / "guided_ablation" / f"{clip}_K{k}.npy"
        records.append(ClipRecord(f"guided:{clip}:K{k}", clip, path, float(row["full_risk"]), "guided_full"))

    for row in _rows(ROOT / "results" / "guided_ablation_extended.csv"):
        path = Path(row["path"])
        if not path.is_absolute():
            path = ROOT / path
        records.append(ClipRecord(
            f"extended:{row['clip']}:K{row['K']}",
            row["clip"],
            path,
            float(row["full_risk"]),
            "guided_extended",
        ))

    for row in _rows(ROOT / "results" / "candidate_audit.csv"):
        path = Path(row["path"])
        records.append(ClipRecord(
            f"audit:{row['clip']}:cand{row['candidate_k']}",
            row["clip"],
            path,
            float(row["full_risk"]),
            "candidate_audit",
        ))

    for row in _rows(ROOT / "results" / "steered_vs_wc_ablation.csv"):
        clip = row["clip"]
        for suffix, col in [("WC4", "risk_wc"), ("PS4", "risk_ps")]:
            path = ROOT / "data" / "steered_ablation" / f"{clip}_{suffix}.npy"
            records.append(ClipRecord(
                f"steered:{clip}:{suffix}",
                clip,
                path,
                float(row[col]),
                "steered",
            ))

    valid: list[ClipRecord] = []
    seen_hashes: set[str] = set()
    for rec in records:
        if not rec.path.exists() or not np.isfinite(rec.risk):
            continue
        arr = np.load(rec.path, mmap_mode="r")
        digest = hashlib.sha1(np.asarray(arr).tobytes()).hexdigest()
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        valid.append(rec)
    return valid


class ClipDataset(Dataset):
    def __init__(self, records: list[ClipRecord], augment: bool):
        self.records = records
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        qpos = np.load(rec.path).astype(np.float32)
        if self.augment and len(qpos) > 80:
            if random.random() < 0.35:
                crop = random.randint(96, len(qpos))
                start = random.randint(0, len(qpos) - crop)
                qpos = qpos[start:start + crop]
            if random.random() < 0.5:
                qpos = qpos.copy()
                qpos[:, :3] += np.random.normal(0.0, 0.002, size=qpos[:, :3].shape).astype(np.float32)
                qpos[:, 7:] += np.random.normal(0.0, 0.001, size=qpos[:, 7:].shape).astype(np.float32)
        x = torch.from_numpy(qpos).T
        y = torch.tensor(np.log1p(rec.risk), dtype=torch.float32)
        return x, y, rec.key, rec.source


def collate(batch):
    xs, ys, keys, sources = zip(*batch)
    max_t = max(x.shape[1] for x in xs)
    padded = torch.zeros(len(xs), QPOS_DIM, max_t)
    mask = torch.zeros(len(xs), max_t, dtype=torch.bool)
    for i, x in enumerate(xs):
        t = x.shape[1]
        padded[i, :, :t] = x
        mask[i, :t] = True
    return padded, mask, torch.stack(ys), list(keys), list(sources)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=5, padding=2 * dilation, dilation=dilation),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2 * dilation, dilation=dilation),
            nn.GroupNorm(8, channels),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class AttentionPool(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, x, mask):
        logits = self.score(x).squeeze(1)
        if logits.shape[1] != mask.shape[1]:
            mask = mask[:, :logits.shape[1]]
        logits = logits.masked_fill(~mask, -1e9)
        weights = torch.softmax(logits, dim=1).unsqueeze(1)
        return torch.sum(x * weights, dim=2)


class ClipCriticV2(nn.Module):
    def __init__(self, width: int = 192):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(QPOS_DIM, width, kernel_size=7, padding=3),
            nn.GroupNorm(8, width),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            ResidualBlock(width, 1),
            ResidualBlock(width, 2),
            ResidualBlock(width, 4),
            ResidualBlock(width, 8),
            ResidualBlock(width, 16),
        )
        self.pool = AttentionPool(width)
        self.head = nn.Sequential(
            nn.Linear(width, width // 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(width // 2, 1),
        )

    def forward(self, x, mask):
        z = self.blocks(self.stem(x))
        pooled = self.pool(z, mask)
        return self.head(pooled).squeeze(-1)


def split_by_group(records: list[ClipRecord], val_frac: float, seed: int):
    groups = sorted({r.group for r in records})
    rng = random.Random(seed)
    rng.shuffle(groups)
    n_val = max(1, int(round(len(groups) * val_frac)))
    val_groups = set(groups[:n_val])
    train = [r for r in records if r.group not in val_groups]
    val = [r for r in records if r.group in val_groups]
    return train, val


def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    preds = []
    labels = []
    rows = []
    with torch.no_grad():
        for x, mask, y, keys, sources in loader:
            x, mask, y = x.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)
            pred = model(x, mask)
            total_loss += loss_fn(pred, y).item() * len(x)
            pred_np = np.expm1(pred.cpu().numpy())
            true_np = np.expm1(y.cpu().numpy())
            preds.extend(pred_np.tolist())
            labels.extend(true_np.tolist())
            for key, source, p, t in zip(keys, sources, pred_np, true_np):
                rows.append({"key": key, "source": source, "pred_risk": float(p), "true_risk": float(t)})
    rho = spearmanr(preds, labels).statistic if len(set(labels)) > 1 else float("nan")
    mae = float(np.mean(np.abs(np.array(preds) - np.array(labels)))) if preds else float("nan")
    return total_loss / max(len(loader.dataset), 1), float(rho), mae, rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out_dir", type=Path, default=ROOT / "results" / "neural_critic_clip_v2")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    records = collect_records()
    train_records, val_records = split_by_group(records, val_frac=0.2, seed=args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Records: {len(records)} unique labeled clips")
    print(f"Train/val: {len(train_records)}/{len(val_records)}")
    by_source: dict[str, int] = {}
    for rec in records:
        by_source[rec.source] = by_source.get(rec.source, 0) + 1
    print(f"By source: {by_source}")

    train_loader = DataLoader(ClipDataset(train_records, augment=True), batch_size=args.batch,
                              shuffle=True, collate_fn=collate)
    val_loader = DataLoader(ClipDataset(val_records, augment=False), batch_size=args.batch,
                            shuffle=False, collate_fn=collate)

    model = ClipCriticV2(width=args.width).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params:,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.HuberLoss(delta=0.75)

    best_rho = -1.0
    best_loss = float("inf")
    log_rows = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x, mask, y, _, _ in train_loader:
            x, mask, y = x.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            pred = model(x, mask)
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item() * len(x)
        sched.step()
        train_loss /= max(len(train_loader.dataset), 1)

        val_loss, rho, mae, val_pred_rows = evaluate(model, val_loader, loss_fn)
        log_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                         "spearman_rho": rho, "mae_risk": mae})
        if epoch == 1 or epoch % 50 == 0:
            print(f"epoch {epoch:4d} train={train_loss:.4f} val={val_loss:.4f} rho={rho:.3f} mae={mae:.2f}")
        if np.isfinite(rho) and (rho > best_rho or (rho == best_rho and val_loss < best_loss)):
            best_rho = rho
            best_loss = val_loss
            torch.save(model.state_dict(), args.out_dir / "model.pt")
            write_csv(args.out_dir / "val_predictions.csv", val_pred_rows)

    write_csv(args.out_dir / "train_log.csv", log_rows)
    config = {
        "records": len(records),
        "train_records": len(train_records),
        "val_records": len(val_records),
        "sources": by_source,
        "epochs": args.epochs,
        "width": args.width,
        "n_params": n_params,
        "best_spearman_rho": best_rho,
        "best_val_loss": best_loss,
        "train_seconds": time.time() - start,
    }
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2))

    epochs = [r["epoch"] for r in log_rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    axes[0].plot(epochs, [r["train_loss"] for r in log_rows], label="train")
    axes[0].plot(epochs, [r["val_loss"] for r in log_rows], label="val")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("Huber loss")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)
    axes[1].plot(epochs, [r["spearman_rho"] for r in log_rows], color="#2E7D32")
    axes[1].axhline(0.85, color="#C62828", ls="--", lw=1)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("validation Spearman rho")
    axes[1].grid(alpha=0.25)
    fig.suptitle("Clip-Level Neural Critic V2")
    fig.savefig(args.out_dir / "training.png", dpi=160)
    plt.close(fig)

    print(f"Best rho={best_rho:.3f}, best val loss={best_loss:.4f}")
    print(f"Saved to {args.out_dir}")


if __name__ == "__main__":
    main()
