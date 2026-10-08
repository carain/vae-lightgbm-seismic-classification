from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "data" / "fujian_event_dataset"
DEFAULT_OUTPUT = ROOT / "models" / "event_scheme_a_v1"
LABELS = {"earthquake": 0, "blast": 1}


@dataclass
class Config:
    dataset: str
    output_dir: str
    seed: int = 42
    latent_dim: int = 128
    batch_size: int = 128
    epochs: int = 25
    learning_rate: float = 2e-4
    beta_kl: float = 1e-3
    kl_warmup_epochs: int = 5
    patience: int = 6
    max_events_per_split: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train compact 3C VAE and event-level LightGBM")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--beta-kl", type=float, default=1e-3)
    parser.add_argument("--kl-warmup-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--max-events-per-split", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 5, stride=2, padding=2, bias=False)
        self.norm1 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1, stride=2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = F.silu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return F.silu(x + residual)


class CompactThreeComponentVAE(nn.Module):
    def __init__(self, latent_dim: int = 128) -> None:
        super().__init__()
        channels = (32, 64, 96, 128, 192)
        blocks = []
        in_channels = 3
        for out_channels in channels:
            blocks.append(ResBlock(in_channels, out_channels))
            in_channels = out_channels
        self.encoder = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(16)
        self.fc_mu = nn.Linear(channels[-1] * 16, latent_dim)
        self.fc_logvar = nn.Linear(channels[-1] * 16, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, channels[-1] * 125)
        decoder_channels = (128, 96, 64, 32, 3)
        decoder = []
        current = channels[-1]
        for index, out_channels in enumerate(decoder_channels):
            decoder.append(nn.ConvTranspose1d(current, out_channels, 4, stride=2, padding=1))
            if index < len(decoder_channels) - 1:
                decoder.extend((nn.GroupNorm(min(8, out_channels), out_channels), nn.SiLU()))
            current = out_channels
        decoder.append(nn.Tanh())
        self.decoder = nn.Sequential(*decoder)
        self.latent_dim = latent_dim

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        pooled = self.pool(encoded).flatten(1)
        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled).clamp(-8.0, 8.0)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.fc_decode(z).view(-1, 192, 125))[..., :4000]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar


def choose_manifest(manifest: pd.DataFrame, maximum: int | None, seed: int) -> pd.DataFrame:
    if maximum is None:
        return manifest
    event_table = manifest[["event_id", "source_type", "split"]].drop_duplicates("event_id")
    chosen = []
    for split_index, (split, split_events) in enumerate(event_table.groupby("split", sort=True)):
        for class_index, (_, class_events) in enumerate(split_events.groupby("source_type", sort=True)):
            n = min(maximum, len(class_events))
            chosen.extend(class_events.sample(n=n, random_state=seed + 100 * split_index + class_index).event_id)
    return manifest[manifest.event_id.isin(chosen)].copy()


def group_indices(manifest: pd.DataFrame, split: str) -> tuple[list[str], list[np.ndarray]]:
    subset = manifest[manifest.split == split]
    groups = subset.groupby("event_id", sort=True).waveform_index.apply(lambda x: x.to_numpy(dtype=np.int64))
    return groups.index.tolist(), groups.tolist()


def select_one_station(groups: list[np.ndarray], epoch: int, seed: int) -> np.ndarray:
    selected = []
    for event_position, indices in enumerate(groups):
        position = (seed + 104729 * epoch + 1009 * event_position) % len(indices)
        selected.append(indices[position])
    return np.asarray(selected, dtype=np.int64)


def iter_batches(indices: np.ndarray, batch_size: int, shuffle: bool, rng: np.random.Generator):
    indices = indices.copy()
    if shuffle:
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield np.sort(indices[start : start + batch_size])


def vae_loss(reconstruction: torch.Tensor, target: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float):
    reconstruction_loss = F.mse_loss(reconstruction, target)
    kl_loss = -0.5 * torch.mean(1.0 + logvar - mu.square() - logvar.exp())
    return reconstruction_loss + beta * kl_loss, reconstruction_loss, kl_loss


def run_vae_epoch(
    model: CompactThreeComponentVAE,
    h5_dataset: h5py.Dataset,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
    beta: float,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    rng: np.random.Generator,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "reconstruction": 0.0, "kl": 0.0, "samples": 0}
    for batch_indices in iter_batches(indices, batch_size, training, rng):
        waveforms = torch.from_numpy(h5_dataset[batch_indices]).to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            reconstruction, mu, logvar = model(waveforms)
            loss, reconstruction_loss, kl_loss = vae_loss(reconstruction, waveforms, mu, logvar, beta)
        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        count = len(batch_indices)
        totals["loss"] += float(loss.detach()) * count
        totals["reconstruction"] += float(reconstruction_loss.detach()) * count
        totals["kl"] += float(kl_loss.detach()) * count
        totals["samples"] += count
    return {key: value / totals["samples"] for key, value in totals.items() if key != "samples"}


@torch.inference_mode()
def extract_latents(
    model: CompactThreeComponentVAE,
    h5_dataset: h5py.Dataset,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    mus, logvars = [], []
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        waveforms = torch.from_numpy(h5_dataset[batch_indices]).to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            mu, logvar = model.encode(waveforms)
        mus.append(mu.float().cpu().numpy())
        logvars.append(logvar.float().cpu().numpy())
    return np.concatenate(mus), np.concatenate(logvars)


def aggregate_events(manifest: pd.DataFrame, mu: np.ndarray, logvar: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    table = manifest.sort_values("waveform_index").reset_index(drop=True).copy()
    if len(table) != len(mu):
        raise ValueError("Latent row count does not match manifest")
    features, rows = [], []
    for event_id, positions in table.groupby("event_id", sort=True).indices.items():
        positions = np.asarray(positions)
        event_mu, event_logvar = mu[positions], logvar[positions]
        feature = np.concatenate(
            (event_mu.mean(0), event_mu.std(0), event_logvar.mean(0), event_logvar.std(0))
        ).astype(np.float32)
        first = table.iloc[positions[0]]
        features.append(feature)
        rows.append(
            {
                "event_id": event_id,
                "source_type": first.source_type,
                "label": int(first.label),
                "split": first.split,
                "n_stations": len(positions),
            }
        )
    return np.stack(features), pd.DataFrame(rows)


def classification_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    prediction = (probability >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "macro_precision": float(precision_score(y_true, prediction, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, prediction, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc_blast": float(average_precision_score(y_true, probability)),
        "brier": float(brier_score_loss(y_true, probability)),
        "confusion_matrix": confusion_matrix(y_true, prediction, labels=[0, 1]).tolist(),
    }


def select_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    candidates = np.linspace(0.05, 0.95, 901)
    scores = np.asarray([f1_score(y_true, probability >= value, average="macro") for value in candidates])
    return float(candidates[int(np.argmax(scores))])


def main() -> None:
    args = parse_args()
    config = Config(
        dataset=str(args.dataset), output_dir=str(args.output_dir), seed=args.seed,
        latent_dim=args.latent_dim, batch_size=args.batch_size, epochs=args.epochs,
        learning_rate=args.learning_rate, beta_kl=args.beta_kl, kl_warmup_epochs=args.kl_warmup_epochs,
        patience=args.patience, max_events_per_split=args.max_events_per_split,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}; {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}", flush=True)

    manifest = pd.read_csv(args.dataset / "manifest.csv", dtype={"event_id": str})
    manifest = choose_manifest(manifest, args.max_events_per_split, args.seed)
    manifest = manifest.sort_values("waveform_index").reset_index(drop=True)
    train_ids, train_groups = group_indices(manifest, "train")
    validation_ids, validation_groups = group_indices(manifest, "validation")
    print(f"events train={len(train_ids)}, validation={len(validation_ids)}; records={len(manifest)}", flush=True)

    model = CompactThreeComponentVAE(args.latent_dim).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"VAE parameters={parameter_count:,}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    rng = np.random.default_rng(args.seed)
    history = []
    best_loss, best_epoch, stale = math.inf, 0, 0

    with h5py.File(args.dataset / "waveforms.h5", "r") as h5:
        waveforms = h5["waveforms"]
        validation_indices = select_one_station(validation_groups, epoch=0, seed=args.seed + 77)
        for epoch in range(1, args.epochs + 1):
            started = time.perf_counter()
            beta = args.beta_kl * min(1.0, epoch / max(args.kl_warmup_epochs, 1))
            train_indices = select_one_station(train_groups, epoch=epoch, seed=args.seed)
            train_result = run_vae_epoch(
                model, waveforms, train_indices, args.batch_size, device, beta, optimizer, scaler, rng
            )
            validation_result = run_vae_epoch(
                model, waveforms, validation_indices, args.batch_size, device, args.beta_kl, None, scaler, rng
            )
            row = {
                "epoch": epoch, "beta": beta, "seconds": time.perf_counter() - started,
                **{f"train_{key}": value for key, value in train_result.items()},
                **{f"validation_{key}": value for key, value in validation_result.items()},
            }
            history.append(row)
            pd.DataFrame(history).to_csv(args.output_dir / "vae_history.csv", index=False)
            print(json.dumps(row), flush=True)
            if validation_result["loss"] < best_loss - 1e-6:
                best_loss, best_epoch, stale = validation_result["loss"], epoch, 0
                torch.save({"model_state": model.state_dict(), "config": asdict(config), "epoch": epoch}, args.output_dir / "vae_best.pt")
            else:
                stale += 1
                if stale >= args.patience:
                    print(f"early stopping at epoch {epoch}", flush=True)
                    break

        checkpoint = torch.load(args.output_dir / "vae_best.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        print(f"extracting latents from epoch {checkpoint['epoch']}", flush=True)
        all_indices = manifest.waveform_index.to_numpy(dtype=np.int64)
        mu, logvar = extract_latents(model, waveforms, all_indices, args.batch_size * 2, device)

    np.savez_compressed(args.output_dir / "station_latents.npz", mu=mu, logvar=logvar, waveform_index=all_indices)
    event_features, event_table = aggregate_events(manifest, mu, logvar)
    np.savez_compressed(args.output_dir / "event_features.npz", features=event_features)
    event_table.to_csv(args.output_dir / "event_feature_manifest.csv", index=False)

    masks = {split: event_table.split.eq(split).to_numpy() for split in ("train", "validation", "test")}
    classifier = LGBMClassifier(
        objective="binary", n_estimators=3000, learning_rate=0.02, num_leaves=31,
        max_depth=-1, min_child_samples=30, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.5,
        random_state=args.seed, n_jobs=-1, verbosity=-1,
    )
    classifier.fit(
        event_features[masks["train"]], event_table.loc[masks["train"], "label"],
        eval_set=[(event_features[masks["validation"]], event_table.loc[masks["validation"], "label"])],
        eval_metric="binary_logloss", callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
    )
    joblib.dump(classifier, args.output_dir / "event_lightgbm.joblib")

    validation_probability = classifier.predict_proba(event_features[masks["validation"]])[:, 1]
    threshold = select_threshold(event_table.loc[masks["validation"], "label"].to_numpy(), validation_probability)
    results = {"best_vae_epoch": best_epoch, "vae_parameters": parameter_count, "validation_selected_threshold": threshold}
    prediction_rows = []
    for split in ("validation", "test"):
        subset = event_table.loc[masks[split]].copy()
        probability = classifier.predict_proba(event_features[masks[split]])[:, 1]
        y_true = subset.label.to_numpy()
        results[split] = classification_metrics(y_true, probability, threshold)
        subset["p_blast"] = probability
        subset["prediction"] = (probability >= threshold).astype(int)
        prediction_rows.append(subset)
        report = classification_report(
            y_true, subset.prediction, target_names=["earthquake", "blast"], digits=4, zero_division=0
        )
        (args.output_dir / f"{split}_classification_report.txt").write_text(report, encoding="utf-8")
    pd.concat(prediction_rows).to_csv(args.output_dir / "event_predictions.csv", index=False)
    (args.output_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
