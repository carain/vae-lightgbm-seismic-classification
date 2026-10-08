from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
from obspy import Stream, UTCDateTime, read
from sklearn.metrics import confusion_matrix

from metrics import bootstrap_metrics, metric_set
from physical_features import extract_station_features
from train_vae import CompactThreeComponentVAE


ROOT = Path(__file__).resolve().parent
DEFAULT_VERTICAL = ROOT.parent / "data" / "Utah_test"
DEFAULT_CATALOG = ROOT.parent / "data" / "utah_catalog.csv"
DEFAULT_BASELINE_MODEL = ROOT.parent / "models" / "event_scheme_a_v1"
DEFAULT_FUSION_MODEL = ROOT.parent / "models" / "event_scheme_a_physics_fusion_v1"
DEFAULT_OUTPUT = ROOT.parent / "outputs" / "utah_physics_fusion_v1"
DATASERVICE = "https://service.earthscope.org/fdsnws/dataselect/1/query"
COMPONENTS = ("E", "N", "Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen-Fujian zero-shot evaluation on the Utah le/ex test set")
    parser.add_argument("--vertical-dir", type=Path, default=DEFAULT_VERTICAL)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--baseline-model-dir", type=Path, default=DEFAULT_BASELINE_MODEL)
    parser.add_argument("--fusion-model-dir", type=Path, default=DEFAULT_FUSION_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--download-batch-size", type=int, default=75)
    parser.add_argument("--inference-batch-size", type=int, default=256)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--redownload", action="store_true")
    return parser.parse_args()


def parse_filename(path: Path) -> dict:
    fields = path.name.removesuffix(".mseed").split(".")
    if len(fields) != 3:
        raise ValueError(f"Unexpected Utah filename: {path.name}")
    source_type, event_id, network = fields[0].split("-", 2)
    return {
        "source_type": source_type,
        "label": 0 if source_type == "le" else 1,
        "event_id": event_id,
        "network": network,
        "station": fields[1],
        "channel": fields[2],
        "vertical_path": str(path),
    }


def build_manifest(vertical_dir: Path, catalog_path: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(vertical_dir.glob("*.mseed")):
        item = parse_filename(path)
        if item["source_type"] not in {"le", "ex"} or item["channel"] != "HHZ":
            continue
        trace = read(str(path), headonly=True)[0]
        item.update(
            location=str(trace.stats.location), sample_rate_hz=float(trace.stats.sampling_rate),
            request_start=str(trace.stats.starttime), request_end=str(trace.stats.endtime),
        )
        rows.append(item)
    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise RuntimeError(f"No le/ex HHZ records found under {vertical_dir}")

    catalog = pd.read_csv(catalog_path, low_memory=False)
    catalog["event_id"] = catalog.id.astype(str).str.replace(r"\.0$", "", regex=True)
    picks = catalog[
        catalog.Typ.isin(["le", "ex"])
        & catalog.phase.astype(str).str.upper().eq("P")
        & catalog.chan.astype(str).eq("HHZ")
    ].copy()
    picks = picks.sort_values(["event_id", "net", "sta", "picktime"]).drop_duplicates(
        ["event_id", "net", "sta"], keep="first"
    )
    columns = ["event_id", "net", "sta", "picktime", "Date", "Time", "evla", "evlo", "Depth", "Mag", "Typ"]
    manifest = manifest.merge(
        picks[columns], left_on=["event_id", "network", "station"], right_on=["event_id", "net", "sta"],
        how="left", validate="many_to_one",
    )
    missing = manifest.picktime.isna()
    if missing.any():
        raise RuntimeError(f"Catalog P picks missing for {int(missing.sum())} local vertical records")
    offsets = []
    for row in manifest.itertuples():
        offsets.append(float(UTCDateTime(row.picktime) - UTCDateTime(row.request_start)))
    manifest["p_offset_from_request_start_s"] = offsets
    if not np.allclose(manifest.p_offset_from_request_start_s, 10.0, atol=0.03):
        raise RuntimeError("Local Utah files are not consistently aligned to P-10 s")
    manifest["download_batch"] = np.arange(len(manifest)) // 75
    return manifest


def fdsn_bulk_body(frame: pd.DataFrame) -> str:
    header = "quality=B\nformat=miniseed\nnodata=404\n"
    lines = []
    for row in frame.itertuples():
        location = row.location if row.location else "--"
        start = str(UTCDateTime(row.request_start)).replace("Z", "")
        end = str(UTCDateTime(row.request_end)).replace("Z", "")
        lines.append(f"{row.network} {row.station} {location} HH? {start} {end}")
    return header + "\n".join(lines) + "\n"


def download_batches(manifest: pd.DataFrame, cache_dir: Path, batch_size: int, redownload: bool) -> list[Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    session = requests.Session()
    for batch_number, start in enumerate(range(0, len(manifest), batch_size)):
        frame = manifest.iloc[start : start + batch_size]
        path = cache_dir / f"batch_{batch_number:03d}.mseed"
        paths.append(path)
        if path.exists() and path.stat().st_size > 0 and not redownload:
            print(f"download {batch_number + 1}: cached {path.stat().st_size / 1024**2:.2f} MiB", flush=True)
            continue
        body = fdsn_bulk_body(frame)
        last_error = None
        for attempt in range(1, 4):
            try:
                response = session.post(DATASERVICE, data=body.encode("ascii"), timeout=(30, 240))
                if response.status_code not in {200, 204, 404}:
                    response.raise_for_status()
                if response.status_code == 200 and response.content:
                    temporary = path.with_suffix(".part")
                    temporary.write_bytes(response.content)
                    temporary.replace(path)
                else:
                    path.write_bytes(b"")
                print(
                    f"download {batch_number + 1}/{int(np.ceil(len(manifest)/batch_size))}: "
                    f"rows={len(frame)}, status={response.status_code}, bytes={len(response.content):,}", flush=True,
                )
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                print(f"download {batch_number + 1} attempt {attempt} failed: {exc}", flush=True)
                time.sleep(2 * attempt)
        if last_error is not None:
            raise RuntimeError(f"EarthScope batch {batch_number} failed") from last_error
    return paths


def trace_for_window(stream: Stream, row, component: str):
    channel = f"HH{component}"
    candidates = stream.select(network=row.network, station=row.station, channel=channel)
    requested_start, requested_end = UTCDateTime(row.request_start), UTCDateTime(row.request_end)
    overlap = Stream(
        traces=[tr.copy() for tr in candidates if tr.stats.endtime >= requested_start and tr.stats.starttime <= requested_end]
    )
    if not overlap:
        return None
    overlap.merge(method=1, fill_value=None)
    if len(overlap) != 1:
        return None
    trace = overlap[0]
    if np.ma.isMaskedArray(trace.data) and np.ma.getmaskarray(trace.data).any():
        return None
    return trace


def prepare_trace(trace, p_time: UTCDateTime) -> tuple[np.ndarray | None, str]:
    if trace is None:
        return None, "component_missing"
    trace = trace.copy()
    if trace.stats.npts < 1000 or not np.isfinite(trace.data).all():
        return None, "raw_nonfinite_or_short"
    trace.detrend("simple")
    if not np.isclose(float(trace.stats.sampling_rate), 100.0):
        trace.resample(100.0, no_filter=False)
    try:
        trace.filter("bandpass", freqmin=1.0, freqmax=25.0, corners=2, zerophase=True)
    except Exception:
        return None, "filter_error"
    p_index = int(round(float(p_time - trace.stats.starttime) * 100.0))
    start, stop = p_index - 500, p_index + 3500
    if start < 0 or stop > trace.stats.npts:
        return None, "window_out_of_bounds"
    window = np.asarray(trace.data[start:stop], dtype=np.float32)
    if window.size != 4000 or not np.isfinite(window).all():
        return None, "window_invalid"
    peak = float(np.max(np.abs(window)))
    if peak <= 0:
        return None, "zero_amplitude"
    return window / peak, "ok"


def component_snr_db(window: np.ndarray) -> float:
    noise = np.asarray(window[200:500], dtype=np.float64)
    signal = np.asarray(window[500:800], dtype=np.float64)
    noise_energy = float(np.sum(noise * noise))
    signal_energy = float(np.sum(signal * signal))
    if noise_energy <= 0 or signal_energy <= 0:
        return -np.inf
    return float(10.0 * np.log10(signal_energy / noise_energy))


def load_and_preprocess(manifest: pd.DataFrame, batch_paths: list[Path], batch_size: int):
    accepted_rows, accepted_waveforms = [], []
    rejection = Counter()
    for batch_number, path in enumerate(batch_paths):
        frame = manifest.iloc[batch_number * batch_size : (batch_number + 1) * batch_size]
        if not path.exists() or path.stat().st_size == 0:
            rejection["batch_no_data"] += len(frame)
            continue
        try:
            stream = read(str(path))
        except Exception:
            rejection["batch_read_error"] += len(frame)
            continue
        for index, row in frame.iterrows():
            p_time = UTCDateTime(row.picktime)
            waves, snrs = [], []
            reason = "ok"
            for component in COMPONENTS:
                trace = trace_for_window(stream, row, component)
                wave, reason = prepare_trace(trace, p_time)
                if wave is None:
                    break
                waves.append(wave)
                snrs.append(component_snr_db(wave))
            if reason != "ok" or len(waves) != 3:
                rejection[reason] += 1
                continue
            if min(snrs) < 3.0:
                rejection["component_snr_below_3_db"] += 1
                continue
            accepted = row.to_dict()
            accepted.update({"snr_E_db": snrs[0], "snr_N_db": snrs[1], "snr_Z_db": snrs[2]})
            accepted_rows.append(accepted)
            accepted_waveforms.append(np.stack(waves).astype(np.float32))
        print(
            f"preprocess {batch_number + 1}/{len(batch_paths)}: accepted_total={len(accepted_rows)}, "
            f"rejected_total={sum(rejection.values())}", flush=True,
        )
    if not accepted_waveforms:
        raise RuntimeError(f"No Utah three-component records passed QC: {dict(rejection)}")
    return pd.DataFrame(accepted_rows), np.stack(accepted_waveforms), rejection


def encode_features(waveforms: np.ndarray, event_ids: np.ndarray, baseline_dir: Path, batch_size: int):
    checkpoint = torch.load(baseline_dir / "vae_best.pt", map_location="cpu", weights_only=False)
    latent_dim = int(checkpoint["config"]["latent_dim"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CompactThreeComponentVAE(latent_dim).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    mu_rows, logvar_rows, physics_rows = [], [], []
    for start in range(0, len(waveforms), batch_size):
        tensor = torch.from_numpy(waveforms[start : start + batch_size]).to(device)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            mu, logvar = model.encode(tensor)
        with torch.inference_mode():
            physical = extract_station_features(tensor)
        mu_rows.append(mu.float().cpu().numpy())
        logvar_rows.append(logvar.float().cpu().numpy())
        physics_rows.append(physical.float().cpu().numpy())
    mu, logvar, physics = map(np.concatenate, (mu_rows, logvar_rows, physics_rows))

    event_rows, vae_features, physical_features = [], [], []
    metadata = pd.DataFrame({"event_id": event_ids.astype(str), "position": np.arange(len(event_ids))})
    for event_id, group in metadata.groupby("event_id", sort=True):
        positions = group.position.to_numpy(dtype=int)
        event_mu, event_logvar, event_physical = mu[positions], logvar[positions], physics[positions]
        vae_features.append(np.concatenate((
            event_mu.mean(0), event_mu.std(0), event_logvar.mean(0), event_logvar.std(0)
        )).astype(np.float32))
        physical_features.append(np.concatenate((
            np.median(event_physical, axis=0),
            np.percentile(event_physical, 75, axis=0) - np.percentile(event_physical, 25, axis=0),
        )).astype(np.float32))
        event_rows.append({"event_id": event_id, "n_valid_stations": len(positions)})
    return pd.DataFrame(event_rows), np.stack(vae_features), np.stack(physical_features), str(device)


def plot_comparison(events: pd.DataFrame, results: dict, output_dir: Path) -> None:
    variants = [
        ("vae_baseline", "VAE + LightGBM", "#4C78A8"),
        ("physics_only", "Physical features", "#F58518"),
        ("vae_physics_fusion", "VAE + physical features", "#7A5195"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.1), constrained_layout=True)
    for ax, (name, title, color) in zip(axes, variants):
        cm = np.asarray(results[name]["confusion_matrix"])
        frac = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(frac, vmin=0, vmax=1, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i,j]}\n{frac[i,j]:.1%}", ha="center", va="center",
                        color="white" if frac[i,j] > 0.55 else "#172033", fontsize=11)
        ax.set_xticks([0, 1], ["Earthquake", "Explosion"])
        ax.set_yticks([0, 1], ["Earthquake", "Explosion"])
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("Utah catalog class")
        ax.set_title(
            f"{title}\nBA={results[name]['balanced_accuracy']:.3f}, AUC={results[name]['roc_auc']:.3f}",
            loc="left", color=color, weight="bold",
        )
        ax.spines[:].set_visible(False)
    fig.suptitle("Frozen Fujian models: zero-shot Utah le/ex evaluation", weight="bold")
    fig.savefig(output_dir / "utah_model_comparison.png", dpi=300, facecolor="white")
    fig.savefig(output_dir / "utah_model_comparison.pdf", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "earthscope_cache"
    started = time.perf_counter()
    manifest = build_manifest(args.vertical_dir, args.catalog)
    manifest.to_csv(args.output_dir / "requested_records.csv", index=False)
    batch_paths = download_batches(manifest, cache_dir, args.download_batch_size, args.redownload)
    accepted, waveforms, rejection = load_and_preprocess(manifest, batch_paths, args.download_batch_size)
    event_stub, vae, physics, device = encode_features(
        waveforms, accepted.event_id.to_numpy(), args.baseline_model_dir, args.inference_batch_size
    )
    event_meta = accepted.sort_values("event_id").drop_duplicates("event_id")[
        ["event_id", "source_type", "label", "evla", "evlo", "Depth", "Mag"]
    ]
    events = event_stub.merge(event_meta, on="event_id", how="left", validate="one_to_one")

    baseline_classifier = joblib.load(args.baseline_model_dir / "event_lightgbm.joblib")
    physics_classifier = joblib.load(args.fusion_model_dir / "physics_only_lightgbm.joblib")
    fusion_classifier = joblib.load(args.fusion_model_dir / "vae_physics_fusion_lightgbm.joblib")
    training = json.loads((args.fusion_model_dir / "metrics.json").read_text(encoding="utf-8"))
    thresholds = {name: float(training[name]["threshold"]) for name in (
        "vae_baseline", "physics_only", "vae_physics_fusion"
    )}
    probabilities = {
        "vae_baseline": baseline_classifier.predict_proba(vae)[:, 1],
        "physics_only": physics_classifier.predict_proba(physics)[:, 1],
        "vae_physics_fusion": fusion_classifier.predict_proba(np.concatenate((vae, physics), axis=1))[:, 1],
    }
    y = events.label.to_numpy(dtype=np.int8)
    results = {}
    for name, probability in probabilities.items():
        threshold = thresholds[name]
        events[f"p_blast_{name}"] = probability
        events[f"prediction_{name}"] = (probability >= threshold).astype(np.int8)
        metrics = metric_set(y, probability, threshold)
        metrics["event_bootstrap_95_ci"] = bootstrap_metrics(y, probability, threshold, args.bootstrap, args.seed)
        results[name] = metrics

    raw_event_counts = manifest.drop_duplicates("event_id").source_type.value_counts().to_dict()
    eligible_event_counts = events.source_type.value_counts().to_dict()
    report = {
        "protocol": {
            "evaluation": "zero-shot; Utah labels were not used for fitting, feature selection, calibration, or thresholds",
            "catalog_labels": "le=earthquake and ex=explosion; qb is absent from the referenced Utah_test cohort",
            "models": "frozen Fujian event-level VAE+LightGBM, physical-only LightGBM, and direct VAE+physical LightGBM",
            "thresholds_from_fujian_validation": thresholds,
            "component_order": "ENZ", "channel_family": "HH",
            "preprocessing": "linear detrend; order-2 zero-phase 1-25 Hz bandpass; P-5:P+35 s; per-component peak normalization",
            "snr": "each component requires 10*log10(sum(P:P+3)^2 / sum(P-3:P)^2) >= 3 dB",
            "event_aggregation": "VAE mean/std of station mu/logvar; physical median/IQR",
            "waveform_service": DATASERVICE,
        },
        "quality_control": {
            "requested_records": int(len(manifest)), "requested_events": int(manifest.event_id.nunique()),
            "requested_event_counts": {str(k): int(v) for k, v in raw_event_counts.items()},
            "eligible_records": int(len(accepted)), "eligible_events": int(len(events)),
            "eligible_event_counts": {str(k): int(v) for k, v in eligible_event_counts.items()},
            "rejections": {str(k): int(v) for k, v in rejection.items()},
        },
        "results": results, "device": device, "runtime_seconds": time.perf_counter() - started,
        "reproducibility_sha256": hashlib.sha256(waveforms.tobytes()).hexdigest(),
    }
    accepted.to_csv(args.output_dir / "eligible_stations.csv", index=False)
    events.to_csv(args.output_dir / "event_predictions.csv", index=False)
    np.savez_compressed(args.output_dir / "event_features.npz", event_id=events.event_id.to_numpy(), vae=vae, physics=physics)
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_comparison(events, results, args.output_dir)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
