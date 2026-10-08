from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

from physical_features import (
    aggregate_event_features,
    aggregated_feature_names,
    extract_station_features,
)
from train_vae import classification_metrics, select_threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train physical-only and VAE-plus-physical event classifiers"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--vae-model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recompute", action="store_true")
    return parser.parse_args()


def make_classifier(seed: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        n_estimators=3000,
        learning_rate=0.02,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=30,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.5,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def extract_station_features_cached(args: argparse.Namespace, device: torch.device):
    cache = args.output_dir / "fujian_station_physics_features.npz"
    manifest = pd.read_csv(args.dataset / "manifest.csv", dtype={"event_id": str})
    manifest = manifest.sort_values("waveform_index").reset_index(drop=True)
    if cache.exists() and not args.recompute:
        saved = np.load(cache)
        if len(saved["features"]) != len(manifest):
            raise ValueError("Cached station features do not match the manifest")
        return saved["features"], manifest

    chunks = []
    started = time.perf_counter()
    with h5py.File(args.dataset / "waveforms.h5", "r") as handle:
        waveforms = handle["waveforms"]
        for start in range(0, len(manifest), args.batch_size):
            stop = min(start + args.batch_size, len(manifest))
            indices = manifest.waveform_index.iloc[start:stop].to_numpy(np.int64)
            tensor = torch.from_numpy(waveforms[indices]).to(device, non_blocking=True)
            with torch.inference_mode():
                chunks.append(extract_station_features(tensor).cpu().numpy().astype(np.float32))
            print(f"physical features {stop}/{len(manifest)} ({time.perf_counter()-started:.1f}s)")
    features = np.concatenate(chunks)
    np.savez_compressed(cache, features=features)
    return features, manifest


def train_variant(name: str, features: np.ndarray, table: pd.DataFrame, args: argparse.Namespace):
    masks = {split: table.split.eq(split).to_numpy() for split in ("train", "validation", "test")}
    model = make_classifier(args.seed)
    model.fit(
        features[masks["train"]],
        table.loc[masks["train"], "label"],
        eval_set=[(features[masks["validation"]], table.loc[masks["validation"], "label"])],
        eval_metric="binary_logloss",
        callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
    )
    validation_probability = model.predict_proba(features[masks["validation"]])[:, 1]
    threshold = select_threshold(
        table.loc[masks["validation"], "label"].to_numpy(), validation_probability
    )
    report = {"threshold": threshold, "best_iteration": int(model.best_iteration_ or model.n_estimators)}
    for split in ("validation", "test"):
        probability = model.predict_proba(features[masks[split]])[:, 1]
        report[split] = classification_metrics(
            table.loc[masks[split], "label"].to_numpy(), probability, threshold
        )
    joblib.dump(model, args.output_dir / f"{name}_lightgbm.joblib")
    return model, report


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    station_features, manifest = extract_station_features_cached(args, device)
    physical, event_ids = aggregate_event_features(
        station_features, manifest.event_id.astype(str).to_numpy()
    )
    table = pd.read_csv(args.vae_model_dir / "event_feature_manifest.csv", dtype={"event_id": str})
    vae = np.load(args.vae_model_dir / "event_features.npz")["features"]
    if not np.array_equal(event_ids.astype(str), table.event_id.astype(str).to_numpy()):
        raise ValueError("Physical and VAE event order differs")
    if len(physical) != len(vae):
        raise ValueError("Physical and VAE feature counts differ")
    np.savez_compressed(args.output_dir / "fujian_event_physics_features.npz", features=physical)

    physical_model, physical_report = train_variant("physics_only", physical, table, args)
    combined = np.concatenate((vae, physical), axis=1)
    combined_model, combined_report = train_variant("vae_physics_fusion", combined, table, args)

    physical_names = aggregated_feature_names()
    pd.DataFrame({
        "feature": physical_names,
        "gain": physical_model.booster_.feature_importance(importance_type="gain"),
    }).sort_values("gain", ascending=False).to_csv(
        args.output_dir / "physics_only_feature_importance.csv", index=False
    )
    pd.DataFrame({
        "feature": [f"vae_{i:03d}" for i in range(vae.shape[1])] + physical_names,
        "gain": combined_model.booster_.feature_importance(importance_type="gain"),
    }).sort_values("gain", ascending=False).to_csv(
        args.output_dir / "fusion_feature_importance.csv", index=False
    )

    baseline = json.loads((args.vae_model_dir / "metrics.json").read_text(encoding="utf-8"))
    report = {
        "vae_baseline": {
            "threshold": baseline["validation_selected_threshold"],
            "validation": baseline["validation"],
            "test": baseline["test"],
        },
        "physics_only": physical_report,
        "vae_physics_fusion": combined_report,
        "event_aggregation": "VAE station mean/std; physical station median/IQR",
        "seed": args.seed,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
