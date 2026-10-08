from __future__ import annotations

import numpy as np
import torch


COMPONENTS = ("E", "N", "Z")
BANDS_HZ = ((1.0, 4.0), (4.0, 8.0), (8.0, 15.0), (15.0, 25.0))


def feature_names() -> list[str]:
    names: list[str] = []
    for metric in (
        "log_p_to_noise_energy", "log_later_to_p_energy", "log_coda_to_p_energy", "p_impulsiveness",
    ):
        names.extend(f"{metric}_{component}" for component in COMPONENTS)
    for low, high in BANDS_HZ:
        names.extend(f"spectral_fraction_{low:g}_{high:g}Hz_{component}" for component in COMPONENTS)
    for metric in ("spectral_centroid_hz", "spectral_bandwidth_hz", "spectral_entropy", "peak_delay_s"):
        names.extend(f"{metric}_{component}" for component in COMPONENTS)
    names.extend(("abs_corr_EN_p", "abs_corr_EZ_p", "abs_corr_NZ_p", "rectilinearity_p", "planarity_p", "energy_duration_80_s"))
    return names


def extract_station_features(waveforms: torch.Tensor, sample_rate: float = 100.0) -> torch.Tensor:
    """Compute scale-robust physical descriptors from P-5:P+35 normalized ENZ inputs."""
    if waveforms.ndim != 3 or waveforms.shape[1:] != (3, 4000):
        raise ValueError(f"Expected [batch, 3, 4000], got {tuple(waveforms.shape)}")
    x = waveforms.float()
    eps = torch.finfo(x.dtype).eps

    def energy(values: torch.Tensor) -> torch.Tensor:
        return values.square().mean(dim=-1).clamp_min(eps)

    noise = x[..., 200:500]
    p_wave = x[..., 500:800]
    later = x[..., 800:2000]
    coda = x[..., 2000:3500]
    post = x[..., 500:3500]
    e_noise, e_p, e_later, e_coda = map(energy, (noise, p_wave, later, coda))
    features = [
        torch.log10(e_p / e_noise),
        torch.log10(e_later / e_p),
        torch.log10(e_coda / e_p),
        p_wave.abs().amax(dim=-1) / torch.sqrt(e_p),
    ]

    spectrum_input = post - post.mean(dim=-1, keepdim=True)
    power = torch.fft.rfft(spectrum_input, dim=-1).abs().square()
    frequencies = torch.fft.rfftfreq(post.shape[-1], d=1.0 / sample_rate).to(x.device)
    analysis = (frequencies >= 1.0) & (frequencies <= 25.0)
    analysis_power = power[..., analysis]
    analysis_frequency = frequencies[analysis]
    total_power = analysis_power.sum(dim=-1).clamp_min(eps)
    for low, high in BANDS_HZ:
        band = (analysis_frequency >= low) & (analysis_frequency < high if high < 25.0 else analysis_frequency <= high)
        features.append(analysis_power[..., band].sum(dim=-1) / total_power)
    centroid = (analysis_power * analysis_frequency).sum(dim=-1) / total_power
    bandwidth = torch.sqrt(
        (analysis_power * (analysis_frequency - centroid.unsqueeze(-1)).square()).sum(dim=-1) / total_power
    )
    distribution = analysis_power / total_power.unsqueeze(-1)
    entropy = -(distribution * torch.log(distribution.clamp_min(eps))).sum(dim=-1) / np.log(analysis_power.shape[-1])
    peak_delay = post.abs().argmax(dim=-1).float() / sample_rate
    features.extend((centroid, bandwidth, entropy, peak_delay))

    standardized = p_wave - p_wave.mean(dim=-1, keepdim=True)
    standardized = standardized / standardized.std(dim=-1, keepdim=True).clamp_min(1e-6)
    correlations = torch.matmul(standardized, standardized.transpose(1, 2)) / standardized.shape[-1]
    features.append(torch.stack((correlations[:, 0, 1].abs(), correlations[:, 0, 2].abs(), correlations[:, 1, 2].abs()), dim=1))
    covariance = torch.matmul(standardized, standardized.transpose(1, 2)) / standardized.shape[-1]
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(eps)
    l3, l2, l1 = eigenvalues[:, 0], eigenvalues[:, 1], eigenvalues[:, 2]
    rectilinearity = 1.0 - (l2 + l3) / (2.0 * l1)
    planarity = 1.0 - 2.0 * l3 / (l1 + l2)
    features.append(torch.stack((rectilinearity, planarity), dim=1))

    component_scaled_energy = post.square() / post.square().sum(dim=-1, keepdim=True).clamp_min(eps)
    cumulative = component_scaled_energy.sum(dim=1).cumsum(dim=-1) / 3.0
    t10 = (cumulative >= 0.10).float().argmax(dim=-1)
    t90 = (cumulative >= 0.90).float().argmax(dim=-1)
    features.append(((t90 - t10).float() / sample_rate).unsqueeze(1))

    output = torch.cat(features, dim=1)
    if output.shape[1] != len(feature_names()):
        raise RuntimeError(f"Feature count mismatch: {output.shape[1]} vs {len(feature_names())}")
    return torch.nan_to_num(output, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 40.0)


def aggregate_event_features(station_features: np.ndarray, event_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return median and IQR station summaries in sorted event-id order."""
    order = np.argsort(event_ids.astype(str), kind="stable")
    ids = event_ids[order]
    values = station_features[order]
    unique_ids, starts = np.unique(ids, return_index=True)
    rows = []
    for start, stop in zip(starts, np.r_[starts[1:], len(ids)]):
        group = values[start:stop]
        median = np.median(group, axis=0)
        iqr = np.percentile(group, 75, axis=0) - np.percentile(group, 25, axis=0)
        rows.append(np.concatenate((median, iqr)).astype(np.float32))
    return np.stack(rows), unique_ids


def aggregated_feature_names() -> list[str]:
    base = feature_names()
    return [f"median_{name}" for name in base] + [f"iqr_{name}" for name in base]
