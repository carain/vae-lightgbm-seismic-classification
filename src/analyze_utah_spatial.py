from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, f1_score, roc_auc_score,
)


ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "outputs" / "utah_physics_fusion_v1"
OUT = ROOT / "outputs" / "utah_paired_spatial_analysis_v1"
CATALOG = ROOT / "data" / "utah_catalog.csv"
R_EARTH_KM = 6371.0088
SEED = 42
N_BOOT = 10000
N_PERM = 20000


def metrics(y, probability, prediction):
    return {
        "accuracy": accuracy_score(y, prediction),
        "balanced_accuracy": balanced_accuracy_score(y, prediction),
        "macro_f1": f1_score(y, prediction, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y, probability),
        "blast_average_precision": average_precision_score(y, probability),
        "brier_score": brier_score_loss(y, probability),
        "earthquake_recall": np.mean(prediction[y == 0] == 0),
        "blast_recall": np.mean(prediction[y == 1] == 1),
    }


def exact_mcnemar(correct_a, correct_b):
    a_wins = int(np.sum(correct_a & ~correct_b))
    b_wins = int(np.sum(~correct_a & correct_b))
    discordant = a_wins + b_wins
    p = float(binomtest(min(a_wins, b_wins), discordant, 0.5).pvalue) if discordant else 1.0
    return {"physical_only_wins": a_wins, "combined_wins": b_wins,
            "discordant_events": discordant, "two_sided_exact_p": p}


def holm_adjust(p_values):
    names = list(p_values)
    order = sorted(range(len(names)), key=lambda i: p_values[names[i]])
    adjusted = np.empty(len(names), float)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, p_values[names[index]] * (len(names) - rank))
        running = max(running, value)
        adjusted[index] = running
    return {name: float(adjusted[i]) for i, name in enumerate(names)}


def paired_resampling(events):
    y = events.label.to_numpy(np.int8)
    p0 = events.p_blast_physics_only.to_numpy(float)
    p1 = events.p_blast_vae_physics_fusion.to_numpy(float)
    z0 = events.prediction_physics_only.to_numpy(np.int8)
    z1 = events.prediction_vae_physics_fusion.to_numpy(np.int8)
    m0, m1 = metrics(y, p0, z0), metrics(y, p1, z1)
    observed = {key: float(m1[key] - m0[key]) for key in m0}
    rng = np.random.default_rng(SEED)
    class_indices = [np.flatnonzero(y == label) for label in (0, 1)]
    boot = {key: np.empty(N_BOOT, float) for key in observed}
    for b in range(N_BOOT):
        idx = np.concatenate([rng.choice(values, len(values), replace=True) for values in class_indices])
        a, c = metrics(y[idx], p0[idx], z0[idx]), metrics(y[idx], p1[idx], z1[idx])
        for key in boot:
            boot[key][b] = c[key] - a[key]
    intervals = {key: {"delta_combined_minus_physical": observed[key],
                       "bootstrap_95_ci": [float(v) for v in np.percentile(boot[key], [2.5, 97.5])]}
                 for key in observed}

    exceed = {key: 0 for key in observed}
    for _ in range(N_PERM):
        swap = rng.random(len(y)) < 0.5
        pa = np.where(swap, p1, p0); pb = np.where(swap, p0, p1)
        za = np.where(swap, z1, z0); zb = np.where(swap, z0, z1)
        a, c = metrics(y, pa, za), metrics(y, pb, zb)
        for key in exceed:
            if abs(c[key] - a[key]) >= abs(observed[key]) - 1e-15:
                exceed[key] += 1
    raw_p = {key: (exceed[key] + 1) / (N_PERM + 1) for key in exceed}
    adjusted = holm_adjust(raw_p)
    for key in intervals:
        intervals[key]["paired_permutation_p"] = float(raw_p[key])
        intervals[key]["holm_adjusted_p_8_metrics"] = float(adjusted[key])
    return m0, m1, intervals


def spatial_groups(events):
    output = events.copy()
    ex_mask = output.label.eq(1).to_numpy()
    ex_coords = np.radians(output.loc[ex_mask, ["evla", "evlo"]].to_numpy())
    ex_labels = DBSCAN(eps=5.0 / R_EARTH_KM, min_samples=3, metric="haversine").fit_predict(ex_coords)
    ex_frame = output.loc[ex_mask, ["event_id", "evla", "evlo"]].copy()
    ex_frame["raw"] = ex_labels
    centers = ex_frame.groupby("raw").evla.mean().sort_values(ascending=False)
    site_names = {label: f"EX-site-{chr(65 + i)}" for i, label in enumerate(centers.index)}
    output["explosion_site_5km"] = "not_applicable"
    output.loc[ex_mask, "explosion_site_5km"] = [site_names[value] for value in ex_labels]

    coords = np.radians(output[["evla", "evlo"]].to_numpy())
    labels = DBSCAN(eps=20.0 / R_EARTH_KM, min_samples=5, metric="haversine").fit_predict(coords)
    counts = pd.Series(labels[labels >= 0]).value_counts()
    order = {label: f"SC-{i + 1:02d}" for i, label in enumerate(counts.index)}
    output["spatial_cluster_20km"] = [order[value] if value >= 0 else f"noise-{event_id}"
                                             for value, event_id in zip(labels, output.event_id)]
    return output


def group_table(events, column):
    rows = []
    for group, frame in events.groupby(column, sort=True):
        if group == "not_applicable":
            continue
        y = frame.label.to_numpy(np.int8)
        p0 = frame.prediction_physics_only.to_numpy(np.int8)
        p1 = frame.prediction_vae_physics_fusion.to_numpy(np.int8)
        row = {
            "group": group, "n_events": len(frame), "n_earthquake": int(np.sum(y == 0)),
            "n_blast": int(np.sum(y == 1)), "center_latitude": frame.evla.mean(),
            "center_longitude": frame.evlo.mean(), "physical_accuracy": np.mean(p0 == y),
            "combined_accuracy": np.mean(p1 == y), "combined_minus_physical_accuracy": np.mean(p1 == y) - np.mean(p0 == y),
        }
        if np.any(y == 0):
            row.update(physical_earthquake_recall=np.mean(p0[y == 0] == 0), combined_earthquake_recall=np.mean(p1[y == 0] == 0))
        if np.any(y == 1):
            row.update(physical_blast_recall=np.mean(p0[y == 1] == 1), combined_blast_recall=np.mean(p1[y == 1] == 1))
        row.update(exact_mcnemar(p0 == y, p1 == y))
        rows.append(row)
    return pd.DataFrame(rows)


def cluster_sensitivity(events):
    rows = []
    for label, name in ((1, "explosion_only"), (None, "all_events")):
        frame = events if label is None else events[events.label == label]
        for eps in ((2, 5, 10, 20, 30) if label == 1 else (10, 15, 20, 30, 50)):
            minimum = 3 if label == 1 else 5
            clusters = DBSCAN(eps=eps / R_EARTH_KM, min_samples=minimum, metric="haversine").fit_predict(
                np.radians(frame[["evla", "evlo"]].to_numpy()))
            sizes = pd.Series(clusters[clusters >= 0]).value_counts().sort_values(ascending=False).tolist()
            rows.append({"scope": name, "eps_km": eps, "min_samples": minimum,
                         "n_clusters": len(sizes), "n_noise": int(np.sum(clusters < 0)),
                         "cluster_sizes": ";".join(map(str, sizes))})
    return pd.DataFrame(rows)


def build_event_inventory(events):
    requested = pd.read_csv(BASE / "requested_records.csv", dtype={"event_id": str})
    eligible = pd.read_csv(BASE / "eligible_stations.csv", dtype={"event_id": str})
    requested.event_id = requested.event_id.astype(str)
    eligible.event_id = eligible.event_id.astype(str)
    catalog = pd.read_csv(CATALOG, low_memory=False, dtype={"id": str})
    catalog["event_id"] = catalog.id.str.replace(r"\.0$", "", regex=True)
    parent = catalog[["event_id", "Typ", "Date", "Time", "evla", "evlo", "Depth", "Mag"]].drop_duplicates("event_id")
    base = requested.sort_values(["event_id", "network", "station"]).drop_duplicates("event_id")[
        ["event_id", "source_type", "label", "Date", "Time", "evla", "evlo", "Depth", "Mag", "Typ"]].copy()
    req = requested.groupby("event_id").agg(
        n_requested_stations=("station", "size"),
        requested_networks=("network", lambda x: ";".join(sorted(set(map(str, x))))),
        requested_stations=("station", lambda x: ";".join(sorted(set(map(str, x))))),
    )
    elig = eligible.groupby("event_id").agg(
        n_eligible_stations=("station", "size"),
        eligible_networks=("network", lambda x: ";".join(sorted(set(map(str, x))))),
        eligible_stations=("station", lambda x: ";".join(sorted(set(map(str, x))))),
    )
    inventory = base.merge(req, on="event_id").merge(elig, on="event_id", how="left")
    inventory["n_eligible_stations"] = inventory.n_eligible_stations.fillna(0).astype(int)
    inventory["analysis_status"] = np.where(inventory.n_eligible_stations > 0, "included", "excluded_all_records_failed_QC")
    inventory["label_definition"] = inventory.source_type.map({"le": "local earthquake", "ex": "generic explosion"})
    prediction_columns = [column for column in events.columns if column.startswith(("p_blast_", "prediction_"))]
    inventory = inventory.merge(events[["event_id", "explosion_site_5km", "spatial_cluster_20km"] + prediction_columns],
                                on="event_id", how="left", validate="one_to_one")
    validation = {
        "requested_event_count": int(inventory.event_id.nunique()),
        "included_event_count": int((inventory.analysis_status == "included").sum()),
        "excluded_event_count": int((inventory.analysis_status != "included").sum()),
        "filename_label_matches_catalog": bool((requested.source_type == requested.Typ).all()),
        "inventory_label_matches_parent_catalog": bool((base.merge(parent[["event_id", "Typ"]], on="event_id", suffixes=("_test", "_parent")).Typ_test ==
                                                         base.merge(parent[["event_id", "Typ"]], on="event_id", suffixes=("_test", "_parent")).Typ_parent).all()),
        "parent_catalog_events": int(parent.event_id.nunique()),
        "parent_catalog_event_types": {str(k): int(v) for k, v in parent.Typ.value_counts().items()},
        "parent_catalog_date_range": [str(parent.Date.min()), str(parent.Date.max())],
    }
    return inventory, validation


def make_figure(events, paired, site_table, output):
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 7, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none", "axes.linewidth": 0.7})
    physical_color, combined_color = "#8B95A5", "#2B6F9F"
    fig = plt.figure(figsize=(7.2, 5.6), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.15))
    ax_a = fig.add_subplot(grid[0, 0]); ax_b = fig.add_subplot(grid[0, 1]); ax_c = fig.add_subplot(grid[1, :])
    keys = ["balanced_accuracy", "macro_f1", "roc_auc", "blast_average_precision", "brier_score"]
    labels = ["Balanced accuracy", "Macro-F1", "ROC AUC", "Blast AP", "Brier reduction"]
    values, lower, upper = [], [], []
    for key in keys:
        delta = paired[2][key]["delta_combined_minus_physical"]
        interval = paired[2][key]["bootstrap_95_ci"]
        if key == "brier_score":
            delta, interval = -delta, [-interval[1], -interval[0]]
        values.append(100 * delta); lower.append(100 * (delta - interval[0])); upper.append(100 * (interval[1] - delta))
    y_pos = np.arange(len(keys))[::-1]
    colors = [combined_color if value >= 0 else physical_color for value in values]
    ax_a.errorbar(values, y_pos, xerr=np.vstack((lower, upper)), fmt="none", ecolor="#374151", capsize=2, lw=0.9)
    ax_a.scatter(values, y_pos, c=colors, s=24, zorder=3)
    ax_a.axvline(0, color="#6B7280", lw=0.7, ls="--")
    ax_a.set_yticks(y_pos, labels); ax_a.set_xlabel("Improvement with combined model (percentage points)")
    ax_a.set_xlim(-3.5, 12); ax_a.set_title("a  Paired effects with 95% bootstrap CIs", loc="left", weight="bold")
    ax_a.grid(axis="x", color="#DDE3EA", lw=0.6); ax_a.set_axisbelow(True)

    transition = np.array([[11, 37], [21, 297]])
    image = ax_b.imshow(transition, cmap="Blues", vmin=0, vmax=transition.max())
    for i in range(2):
        for j in range(2):
            ax_b.text(j, i, str(transition[i, j]), ha="center", va="center",
                      color="white" if transition[i, j] > transition.max()/2 else "#182230", fontsize=9, weight="bold")
    ax_b.set_xticks([0, 1], ["Incorrect", "Correct"]); ax_b.set_yticks([0, 1], ["Incorrect", "Correct"])
    ax_b.set_xlabel("Combined"); ax_b.set_ylabel("Physical-only")
    ax_b.set_title("b  Event-level correctness transitions", loc="left", weight="bold")
    ax_b.spines[:].set_visible(False)

    eq = events[events.label == 0]
    ex = events[events.label == 1]
    ax_c.scatter(eq.evlo, eq.evla, s=10, c="#637E9B", alpha=0.75, linewidths=0, label=f"Earthquake (n={len(eq)})")
    site_colors = {"EX-site-A": "#D55E00", "EX-site-B": "#E6A700"}
    for site, frame in ex.groupby("explosion_site_5km"):
        ax_c.scatter(frame.evlo, frame.evla, s=20, marker="^", c=site_colors[site], edgecolors="white", linewidths=0.25,
                     label=f"{site} explosion (n={len(frame)})")
    changed = events[events.prediction_physics_only != events.prediction_vae_physics_fusion]
    ax_c.scatter(changed.evlo, changed.evla, s=36, facecolors="none", edgecolors="#111111", linewidths=0.65,
                 label=f"Models disagree (n={len(changed)})")
    ax_c.set_xlabel("Longitude (°E)"); ax_c.set_ylabel("Latitude (°N)")
    ax_c.set_title("c  Spatial clusters and paired decision changes", loc="left", weight="bold")
    ax_c.legend(loc="lower left", ncol=2, fontsize=6.5)
    ax_c.grid(color="#E3E8EF", lw=0.5); ax_c.set_axisbelow(True)
    inset = ax_c.inset_axes([0.025, 0.52, 0.30, 0.40])
    northern = ex[ex.explosion_site_5km == "EX-site-A"]
    inset.scatter(northern.evlo, northern.evla, s=13, marker="^", c=site_colors["EX-site-A"], linewidths=0)
    disagreement = northern[northern.prediction_physics_only != northern.prediction_vae_physics_fusion]
    inset.scatter(disagreement.evlo, disagreement.evla, s=27, facecolors="none", edgecolors="#111111", linewidths=0.55)
    inset.set_title("EX-site-A detail", fontsize=6.5, loc="left", weight="bold")
    inset.tick_params(labelsize=5.5); inset.grid(color="#E3E8EF", lw=0.4); inset.set_axisbelow(True)
    fig.savefig(output / "utah_paired_spatial_analysis.svg", bbox_inches="tight")
    fig.savefig(output / "utah_paired_spatial_analysis.pdf", bbox_inches="tight")
    fig.savefig(output / "utah_paired_spatial_analysis.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(output / "utah_paired_spatial_analysis.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(BASE / "event_predictions.csv", dtype={"event_id": str})
    events = spatial_groups(events)
    physical, combined, paired = paired_resampling(events)
    correct_physical = events.prediction_physics_only.eq(events.label).to_numpy()
    correct_combined = events.prediction_vae_physics_fusion.eq(events.label).to_numpy()
    mcnemar = {
        "all_events": exact_mcnemar(correct_physical, correct_combined),
        "earthquakes": exact_mcnemar(correct_physical[events.label.eq(0)], correct_combined[events.label.eq(0)]),
        "explosions": exact_mcnemar(correct_physical[events.label.eq(1)], correct_combined[events.label.eq(1)]),
    }
    mcnemar_adjusted = holm_adjust({name: value["two_sided_exact_p"] for name, value in mcnemar.items()})
    for name in mcnemar:
        mcnemar[name]["holm_adjusted_p_3_tests"] = mcnemar_adjusted[name]
    site_table = group_table(events[events.label.eq(1)], "explosion_site_5km")
    cluster_table = group_table(events, "spatial_cluster_20km")
    sensitivity = cluster_sensitivity(events)
    inventory, provenance_validation = build_event_inventory(events)
    events.to_csv(OUT / "utah_events_with_spatial_groups.csv", index=False)
    inventory.to_csv(OUT / "utah_event_inventory_complete.csv", index=False)
    inventory[inventory.analysis_status != "included"].to_csv(OUT / "utah_excluded_events.csv", index=False)
    site_table.to_csv(OUT / "explosion_site_performance.csv", index=False)
    cluster_table.to_csv(OUT / "spatial_cluster_performance.csv", index=False)
    sensitivity.to_csv(OUT / "spatial_cluster_sensitivity.csv", index=False)
    pd.DataFrame([{"metric": k, "physical_only": physical[k], "combined": combined[k], **paired[k]} for k in paired]).to_csv(
        OUT / "paired_metric_comparison.csv", index=False)
    report = {
        "analysis_unit": "event_id", "n_events": len(events), "n_earthquakes": int(np.sum(events.label == 0)),
        "n_explosions": int(np.sum(events.label == 1)), "resampling": {"stratified_paired_bootstrap": N_BOOT, "paired_permutations": N_PERM, "seed": SEED},
        "physical_only": {k: float(v) for k, v in physical.items()}, "combined": {k: float(v) for k, v in combined.items()},
        "paired_differences": paired, "exact_mcnemar": mcnemar,
        "spatial_definition": {"explosion_sites": "class-specific DBSCAN, haversine distance, eps=5 km, min_samples=3",
                               "all_event_clusters": "label-blind DBSCAN, haversine distance, eps=20 km, min_samples=5"},
        "provenance_validation": provenance_validation,
        "source_lineage": {
            "catalog": "University of Utah Seismograph Stations (UUSS) Utah Discrimination Datasets 2012-2020",
            "catalog_doi": "10.31905/RDQW00CT",
            "related_publication_doi": "10.1029/2018GL081119",
            "waveforms": "EarthScope FDSN dataselect service; network/station/channel/time requests archived in requested_records.csv",
            "labels": {"le": "local earthquake", "ex": "generic explosion", "qb": "quarry blast (not present in the test-file cohort)"},
        },
    }
    (OUT / "paired_spatial_statistics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    checksums = {}
    for path in (BASE / "event_predictions.csv", BASE / "requested_records.csv", BASE / "eligible_stations.csv", CATALOG):
        checksums[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT / "input_checksums.json").write_text(json.dumps(checksums, indent=2), encoding="utf-8")
    make_figure(events, (physical, combined, paired), site_table, OUT)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
