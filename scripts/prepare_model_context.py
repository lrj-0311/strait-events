"""Create lagged news context and entity groups for the dynamic graph model."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FEATURES = (
    "avg_tone", "goldstein_mean", "mention_sum", "conflict_ratio",
    "escalation_index", "external_intervention_index", "regional_linkage_index",
    "core_entity_visibility", "external_entity_dominance",
)


def prepare_context(mechanism: pd.DataFrame, slices: pd.DataFrame) -> pd.DataFrame:
    mechanism = mechanism.sort_values("timestamp_start").copy()
    mechanism["timestamp_start"] = pd.to_datetime(mechanism["timestamp_start"])
    slices = slices.copy()
    slices["timestamp_start"] = pd.to_datetime(slices["timestamp_start"])
    if mechanism["timestamp_start"].duplicated().any():
        raise ValueError("Duplicate mechanism weeks")
    if not mechanism["timestamp_start"].diff().dropna().eq(pd.Timedelta(days=7)).all():
        raise ValueError("Mechanism table must contain consecutive Monday weeks")
    missing = set(FEATURES) - set(mechanism.columns)
    if missing:
        raise ValueError(f"Missing mechanism columns: {sorted(missing)}")

    context = mechanism[["timestamp_start"]].copy()
    for feature in FEATURES:
        values = mechanism[feature].astype(float)
        if feature == "mention_sum":
            values = np.log1p(values)
            name = "mention_sum_log_lag_1"
        else:
            name = f"{feature}_lag_1"
        context[name] = values.shift(1)

    aligned = slices[["timestamp_id", "timestamp_start", "split"]].merge(
        context, on="timestamp_start", how="left", validate="one_to_one"
    ).sort_values("timestamp_id")
    for feature in [column for column in aligned if column.endswith("_lag_1")]:
        train = aligned.loc[aligned["split"].eq("train"), feature]
        mean = float(train.mean()) if train.notna().any() else 0.0
        aligned[feature] = aligned[feature].fillna(mean)
    return aligned


def prepare_groups(entities: pd.DataFrame) -> pd.DataFrame:
    entities = entities.sort_values("id").copy()
    if entities["id"].tolist() != list(range(len(entities))):
        raise ValueError("Entity IDs must be contiguous from zero")
    names = entities["name"].str.lower()
    entities["is_core"] = names.str.contains("china|taiwan", regex=True).astype(int)
    entities["is_external"] = (names.str.contains(
        "united states|japan|south korea|philippines|indonesia", regex=True
    ) & ~entities["is_core"].astype(bool)).astype(int)
    return entities


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mechanism", type=Path, default=ROOT / "data/MECHANISM_FEATURES_WEEKLY/mechanism_features.csv")
    parser.add_argument("--time-slices", type=Path, default=ROOT / "data/ACLED_MUSE_7COUNTRY/time_slice_features.csv")
    parser.add_argument("--entity-map", type=Path, default=ROOT / "data/ACLED_MUSE_7COUNTRY/entity2id.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/model_context")
    args = parser.parse_args()
    context = prepare_context(pd.read_csv(args.mechanism), pd.read_csv(args.time_slices))
    groups = prepare_groups(pd.read_csv(args.entity_map))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    context.to_csv(args.output_dir / "mechanism_context_weekly.csv", index=False)
    groups.to_csv(args.output_dir / "entity_group_features.csv", index=False)
    print(f"Context weeks: {len(context)}; entities: {len(groups)}")


if __name__ == "__main__":
    main()
