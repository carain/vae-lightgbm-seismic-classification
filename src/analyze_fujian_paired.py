from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from analyze_utah_spatial import exact_mcnemar, holm_adjust, paired_resampling


ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "models" / "event_scheme_a_v1"
FUSION = ROOT / "models" / "event_scheme_a_physics_fusion_v1"
OUT = ROOT / "outputs" / "utah_paired_spatial_analysis_v1"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(BASELINE / "event_feature_manifest.csv", dtype={"event_id": str})
    vae = np.load(BASELINE / "event_features.npz")["features"]
    physical = np.load(FUSION / "fujian_event_physics_features.npz")["features"]
    physical_model = joblib.load(FUSION / "physics_only_lightgbm.joblib")
    combined_model = joblib.load(FUSION / "vae_physics_fusion_lightgbm.joblib")
    mask = table.split.eq("test").to_numpy()
    events = table.loc[mask, ["event_id", "source_type", "label", "split", "n_stations"]].reset_index(drop=True)
    events["p_blast_physics_only"] = physical_model.predict_proba(physical[mask])[:, 1]
    events["p_blast_vae_physics_fusion"] = combined_model.predict_proba(np.concatenate((vae[mask], physical[mask]), axis=1))[:, 1]
    # Frozen thresholds selected on the Fujian validation partition and archived in the model report.
    events["prediction_physics_only"] = (events.p_blast_physics_only >= 0.435).astype(np.int8)
    events["prediction_vae_physics_fusion"] = (events.p_blast_vae_physics_fusion >= 0.496).astype(np.int8)
    model0, model1, paired = paired_resampling(events)
    c0 = events.prediction_physics_only.eq(events.label).to_numpy()
    c1 = events.prediction_vae_physics_fusion.eq(events.label).to_numpy()
    mcnemar = {
        "all_events": exact_mcnemar(c0, c1),
        "earthquakes": exact_mcnemar(c0[events.label.eq(0)], c1[events.label.eq(0)]),
        "blasts": exact_mcnemar(c0[events.label.eq(1)], c1[events.label.eq(1)]),
    }
    adjusted = holm_adjust({name: values["two_sided_exact_p"] for name, values in mcnemar.items()})
    for name in mcnemar:
        mcnemar[name]["holm_adjusted_p_3_tests"] = adjusted[name]
    report = {"analysis_unit": "event_id", "n_events": len(events), "physical_only": model0,
              "combined": model1, "paired_differences": paired, "exact_mcnemar": mcnemar}
    events.to_csv(OUT / "fujian_paired_event_predictions.csv", index=False)
    (OUT / "fujian_paired_statistics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame([{"metric": key, "physical_only": model0[key], "combined": model1[key], **paired[key]} for key in paired]).to_csv(
        OUT / "fujian_paired_metric_comparison.csv", index=False)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
