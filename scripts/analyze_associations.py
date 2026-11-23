"""Analyze coverage-aware weekly associations without changing model runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
COUNTRIES = ["China", "Taiwan", "United States", "Japan", "South Korea", "Philippines", "Indonesia"]
CORE = ["China", "Taiwan"]
SIGNALS = ["negative_tone_index", "balanced_negative_tone", "core_negative_tone",
           "ct_negative_mean_tone", "us_intervention_index", "militarization_index",
           "escalation_index", "attention_per_event"]
OUTCOMES = ["acled_event_count", "no_us_protests_count", "no_us_count",
            "core_country_count", "violence_count", "fatalities_sum", "strait_text_count"]
LAGS = [-4, -2, -1, 0, 1, 2, 4, 8]


def benjamini_hochberg(p_values: pd.Series) -> np.ndarray:
    values = p_values.to_numpy(dtype=float)
    result = np.full(len(values), np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(values))
    if not len(valid):
        return result
    order = valid[np.argsort(values[valid])]
    ranked = values[order] * len(valid) / np.arange(1, len(valid) + 1)
    result[order] = np.clip(np.minimum.accumulate(ranked[::-1])[::-1], 0, 1)
    return result


def markdown_table(frame: pd.DataFrame) -> str:
    def render(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.4f}"
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(map(str, frame.columns)) + " |",
        "|" + "|".join(["---"] * len(frame.columns)) + "|",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(render(value) for value in row) + " |")
    return "\n".join(lines)


def monday(values):
    dates = pd.to_datetime(values)
    return dates - pd.to_timedelta(dates.dt.weekday, unit="D")


def full_week_bounds(first, last):
    first, last = pd.Timestamp(first).normalize(), pd.Timestamp(last).normalize()
    start = first + pd.Timedelta(days=(-first.weekday()) % 7)
    end = last - pd.Timedelta(days=(last.weekday() + 1) % 7) - pd.Timedelta(days=6)
    return start, end


def add_future_labels(frame, columns, lags=(1, 2, 4, 8)):
    result = frame.copy()
    for column in columns:
        for lag in lags:
            # Reindex by calendar date so missing weeks cannot become adjacent observations.
            dates = result.index + pd.Timedelta(weeks=lag)
            result[f"{column}_future_{lag}w"] = result[column].reindex(dates).to_numpy()
    return result


def load_inputs():
    feature_dir = ROOT / "data" / "MECHANISM_FEATURES_WEEKLY"
    paths = {
        "mechanism": feature_dir / "mechanism_features.csv",
        "nodes": feature_dir / "entity_node_features.csv",
        "edges": feature_dir / "entity_edge_features.csv",
        "events": ROOT / "data/ACLED_MUSE_7COUNTRY/map_layers/events_map_points.csv",
        "time": ROOT / "data/ACLED_MUSE_7COUNTRY/time_slice_features.csv",
    }
    data = {key: pd.read_csv(value, low_memory=False) for key, value in paths.items()}
    for key in ["mechanism", "nodes", "edges", "time"]:
        data[key]["timestamp_start"] = pd.to_datetime(data[key]["timestamp_start"])
    events = data["events"]
    if events["event_id"].duplicated().any():
        raise ValueError("Duplicate source event IDs require explicit reconciliation.")
    events["event_date"] = pd.to_datetime(events["event_date"])
    events["timestamp_start"] = monday(events["event_date"])
    if set(events["country"]) != set(COUNTRIES):
        raise ValueError("Unexpected country scope.")
    data["events"] = events
    return data, paths


def build_panels(data):
    events = data["events"].copy()
    original = data["mechanism"].sort_values("timestamp_start").set_index("timestamp_start")
    calendar = pd.date_range(original.index.min(), original.index.max(), freq="W-MON")
    if original.index.duplicated().any():
        raise ValueError("Duplicated weekly feature rows.")
    panel = original.reindex(calendar).copy()
    panel.index.name = "timestamp_start"
    source_last = events["event_date"].max()
    coverage = events.groupby("country").agg(
        first_event=("event_date", "min"), last_event=("event_date", "max"),
        event_count=("event_id", "size"),
    ).reindex(COUNTRIES).reset_index()
    coverage["first_full_week"] = coverage["first_event"].map(lambda d: full_week_bounds(d, source_last)[0])
    end = full_week_bounds(events["event_date"].min(), source_last)[1]
    coverage["last_full_week"] = end
    coverage["coverage_basis"] = "observed_first_event_and_conservative_snapshot_boundary"
    country_weekly = []
    violence_types = ["Battles", "Explosions/Remote violence", "Violence against civilians"]
    events["is_violence"] = events["event_type"].isin(violence_types).astype(int)
    events["is_protest"] = events["event_type"].eq("Protests").astype(int)
    events["fatalities"] = pd.to_numeric(events["fatalities"], errors="raise")
    for row in coverage.itertuples():
        subset = events.loc[events["country"].eq(row.country)]
        weekly = subset.groupby("timestamp_start").agg(
            event_count=("event_id", "size"), protest_count=("is_protest", "sum"),
            violence_count=("is_violence", "sum"), fatalities_sum=("fatalities", "sum"),
        ).reindex(calendar)
        covered = (calendar >= row.first_full_week) & (calendar <= end)
        weekly = weekly.fillna(0).where(pd.Series(covered, index=calendar), axis=0)
        weekly["country"] = row.country
        weekly["in_observed_full_week_range"] = covered
        country_weekly.append(weekly.rename_axis("timestamp_start").reset_index())
    country_weekly = pd.concat(country_weekly, ignore_index=True)
    values = country_weekly.pivot(index="timestamp_start", columns="country", values="event_count")
    cover_matrix = country_weekly.pivot(index="timestamp_start", columns="country",
                                        values="in_observed_full_week_range").reindex(columns=COUNTRIES)
    panel["covered_country_count"] = cover_matrix.sum(axis=1)
    balanced = cover_matrix.all(axis=1)
    historical_start = full_week_bounds(events["event_date"].min(), source_last)[0]
    historical = (calendar >= historical_start) & (calendar <= end)
    old_behavior = {"acled_event_count", "protest_count", "violence_count",
                    "fatalities_sum", "strategic_development_count", "risk_level"}
    for column in panel:
        if column in old_behavior or column.startswith(("risk_t_plus_", "risk_count_t_plus_")):
            panel[column] = np.nan
        elif "_lag_" in column and (column.startswith(("acled_", "risk_", "attention_per_acled",
                                                       "gdelt_to_acled", "gdelt_conflict_per_acled"))):
            panel[column] = np.nan
    panel["acled_event_count"] = values[COUNTRIES].sum(axis=1, min_count=7)
    panel["available_country_count_total"] = events.groupby("timestamp_start").size().reindex(calendar).fillna(0).where(historical)
    panel["core_country_count"] = values[CORE].sum(axis=1, min_count=2)
    for key in ["protest_count", "violence_count", "fatalities_sum"]:
        wide = country_weekly.pivot(index="timestamp_start", columns="country", values=key)
        panel[key] = wide[COUNTRIES].sum(axis=1, min_count=7)
    us_protests = events.loc[events["country"].eq("United States") & events["is_protest"].eq(1)]
    panel["no_us_protests_count"] = panel["acled_event_count"] - us_protests.groupby("timestamp_start").size().reindex(calendar).fillna(0)
    panel["no_us_count"] = values[[c for c in COUNTRIES if c != "United States"]].sum(axis=1, min_count=6)
    panel["us_event_share"] = values["United States"] / panel["acled_event_count"].replace(0, np.nan)
    text_fields = events[["actor1", "actor2", "notes"]].fillna("").agg(" ".join, axis=1).str.lower()
    china = text_fields.str.contains(r"\b(?:china|chinese|beijing|prc|people.s liberation army)\b", regex=True)
    taiwan = text_fields.str.contains(r"\b(?:taiwan|taiwanese|taipei|kinmen|matsu|penghu)\b", regex=True)
    strait = text_fields.str.contains(r"\b(?:taiwan strait|cross[- ]strait)\b", regex=True) | (china & taiwan)
    events["strait_text_candidate"] = strait
    panel["strait_text_count"] = events.loc[strait].groupby("timestamp_start").size().reindex(calendar).fillna(0).where(balanced)
    panel["acled_observed"] = balanced
    panel["balanced_sample"] = balanced
    panel["historical_available_sample"] = historical
    panel["balanced_2021plus_sample"] = balanced & (calendar >= pd.Timestamp("2021-01-04"))
    panel["gdelt_observed"] = panel["gdelt_event_count"].notna()
    panel["attention_per_event"] = panel["media_attention_index"] / panel["gdelt_event_count"].replace(0, np.nan)
    for feature in ["gdelt_to_acled_event_ratio", "attention_per_acled_event", "gdelt_conflict_per_acled_event",
                    "military_signal_per_acled_event", "external_signal_per_acled_event", "regional_signal_per_acled_event"]:
        if feature in panel:
            panel[feature] = np.nan
    nodes = data["nodes"].pivot(index="timestamp_start", columns="entity", values="entity_negative_tone_index")
    panel["balanced_negative_tone"] = nodes[COUNTRIES].mean(axis=1).where(nodes[COUNTRIES].notna().all(axis=1))
    panel["core_negative_tone"] = nodes[CORE].mean(axis=1).where(nodes[CORE].notna().all(axis=1))
    edges = data["edges"]
    ct_mask = (edges["entity_left"].eq("China") & edges["entity_right"].eq("Taiwan")) | (
        edges["entity_right"].eq("China") & edges["entity_left"].eq("Taiwan"))
    ct = edges.loc[ct_mask].set_index("timestamp_start")
    panel["china_taiwan_edge_weight"] = ct["edge_weight"].reindex(calendar)
    panel["ct_negative_mean_tone"] = (-ct["edge_avg_tone"]).clip(lower=0).reindex(calendar)
    regional = edges.loc[(edges["entity_left"].isin(CORE) & edges["entity_right"].isin(COUNTRIES[3:])) |
                         (edges["entity_right"].isin(CORE) & edges["entity_left"].isin(COUNTRIES[3:]))]
    panel["regional_core_edge_weight"] = regional.groupby("timestamp_start")["edge_weight"].sum().reindex(calendar)
    panel["china_taiwan_edge_change"] = panel["china_taiwan_edge_weight"].diff()
    panel["regional_core_edge_change"] = panel["regional_core_edge_weight"].diff()
    # Keep the series names used by the analysis tables.
    for lag in [1, 2, 4]:
        panel[f"acled_event_count_lag_{lag}"] = panel["acled_event_count"].shift(lag)
    panel = add_future_labels(panel, OUTCOMES)
    for horizon in [1, 2]:
        shifted = [panel["acled_event_count"].shift(-h) for h in range(1, horizon + 1)]
        count = sum(shifted)
        panel[f"risk_count_t_plus_{horizon * 7}"] = count
        panel[f"risk_t_plus_{horizon * 7}"] = count.gt(0).astype(float).where(count.notna())
    return panel, coverage, country_weekly, cover_matrix, events


def scope_frames(panel):
    historical = panel.loc[panel["historical_available_sample"]].copy()
    historical["acled_event_count"] = historical["available_country_count_total"]
    frames = {
        "historical_available": historical,
        "balanced_2020plus": panel.loc[panel["balanced_sample"]].copy(),
        "balanced_2021plus": panel.loc[panel["balanced_2021plus_sample"]].copy(),
    }
    for frame in frames.values():
        frame["china_taiwan_edge_change"] = frame["china_taiwan_edge_weight"].diff()
        frame["regional_core_edge_change"] = frame["regional_core_edge_weight"].diff()
    return frames


def correlation_record(x, y):
    pair = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    result = {"n": len(pair), "rho": np.nan, "p_iid": np.nan, "p_hac4": np.nan, "p_hac8": np.nan}
    if len(pair) < 24 or pair.nunique().min() < 2:
        result["status"] = "insufficient_variation"
        return result
    result["rho"], result["p_iid"] = map(float, spearmanr(pair["x"], pair["y"]))
    ranked = np.column_stack([rankdata(pair[c]) for c in ["x", "y"]])
    ranked = (ranked - ranked.mean(axis=0)) / ranked.std(axis=0)
    for bandwidth in [4, 8]:
        fit = sm.OLS(ranked[:, 1], sm.add_constant(ranked[:, 0])).fit(
            cov_type="HAC", cov_kwds={"maxlags": bandwidth, "use_correction": True})
        result[f"p_hac{bandwidth}"] = float(fit.pvalues[1])
    result["status"] = "ok"
    return result


def adjust_fdr(frame, p_columns, groups):
    for column in p_columns:
        q = column.replace("p_", "q_", 1)
        frame[q] = np.nan
        for _, group in frame.groupby(groups, dropna=False):
            frame.loc[group.index, q] = benjamini_hochberg(group[column])
    return frame


def lag_results(panel):
    rows = []
    for scope, frame in scope_frames(panel).items():
        for signal in SIGNALS:
            for outcome in OUTCOMES:
                for lag in LAGS:
                    rows.append({"scope": scope, "signal": signal, "outcome": outcome,
                                 "lead_weeks": lag,
                                 **correlation_record(frame[signal], frame[outcome].shift(-lag))})
    return adjust_fdr(pd.DataFrame(rows), ["p_iid", "p_hac4", "p_hac8"], ["scope"])


def controlled_results(panel):
    rows = []
    for scope, frame in scope_frames(panel).items():
        if scope == "historical_available":
            continue
        time = np.arange(len(frame), dtype=float)
        for outcome in OUTCOMES:
            y = np.log1p(frame[outcome])
            controls = pd.DataFrame({
                "current_activity": y, "activity_lag1": y.shift(1),
                "activity_lag4": y.shift(4), "trend": time,
                "annual_sin": np.sin(2 * np.pi * time / 52.1775),
                "annual_cos": np.cos(2 * np.pi * time / 52.1775),
                "news_volume": np.log1p(frame["gdelt_event_count"]),
            }, index=frame.index)
            for signal in SIGNALS:
                for lag in [1, 2, 4, 8]:
                    sample = controls.assign(signal=frame[signal], response=y.shift(-lag)).dropna()
                    row = {"scope": scope, "signal": signal, "outcome": outcome,
                           "lead_weeks": lag, "n": len(sample), "beta_per_signal_sd": np.nan,
                           "ci_low": np.nan, "ci_high": np.nan, "p_hac": np.nan}
                    if len(sample) < 40 or sample["signal"].std() == 0 or sample["response"].std() == 0:
                        row["status"] = "insufficient_variation"
                    else:
                        predictors = sample.drop(columns="response")
                        varying = predictors.std().gt(1e-10)
                        predictors = predictors.loc[:, varying]
                        predictors = (predictors - predictors.mean()) / predictors.std()
                        fit = sm.OLS(sample["response"], sm.add_constant(predictors)).fit(
                            cov_type="HAC", cov_kwds={"maxlags": max(8, lag + 1), "use_correction": True})
                        row.update(beta_per_signal_sd=float(fit.params["signal"]),
                                   ci_low=float(fit.conf_int().loc["signal", 0]),
                                   ci_high=float(fit.conf_int().loc["signal", 1]),
                                   p_hac=float(fit.pvalues["signal"]), status="ok")
                    rows.append(row)
    return adjust_fdr(pd.DataFrame(rows), ["p_hac"], ["scope"])


def block_indices(n, block, rng):
    starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
    return np.concatenate([np.arange(s, s + block) for s in starts])[:n]


def bootstrap_correlations(panel, repeats, seed):
    rows = []
    rng = np.random.default_rng(seed)
    for scope, frame in scope_frames(panel).items():
        tests = [("negative_tone_index", outcome, 4) for outcome in
                 ["acled_event_count", "no_us_protests_count", "core_country_count"]]
        tests.append(("regional_core_edge_change", "china_taiwan_edge_change", 0))
        for signal, outcome, lag in tests:
            pair = pd.concat([frame[signal], frame[outcome].shift(-lag)], axis=1).dropna()
            if len(pair) < 24 or pair.nunique().min() < 2:
                continue
            x, y = pair.iloc[:, 0].to_numpy(), pair.iloc[:, 1].to_numpy()
            for block in [4, 8]:
                estimates = []
                for _ in range(repeats):
                    ix = block_indices(len(pair), block, rng)
                    estimates.append(spearmanr(x[ix], y[ix]).statistic)
                rows.append({
                    "scope": scope, "signal": signal, "outcome": outcome, "lead_weeks": lag,
                    "n": len(pair), "block_weeks": block, "bootstrap_repeats": repeats,
                    "rho": float(spearmanr(x, y).statistic),
                    "ci_low": float(np.nanquantile(estimates, 0.025)),
                    "ci_high": float(np.nanquantile(estimates, 0.975)),
                })
    return pd.DataFrame(rows)


def regional_results(panel):
    rows = []
    for scope, frame in scope_frames(panel).items():
        for mode in ["level", "change"]:
            suffix = "weight" if mode == "level" else "change"
            rows.append({"scope": scope, "series": mode,
                         **correlation_record(frame[f"regional_core_edge_{suffix}"],
                                              frame[f"china_taiwan_edge_{suffix}"])})
    return adjust_fdr(pd.DataFrame(rows), ["p_hac4", "p_hac8"], ["scope"])


def forecast_table(frame, outcome, splits):
    y = frame[outcome]
    design = pd.DataFrame({
        "activity_current": np.log1p(y), "activity_lag1": np.log1p(y.shift(1)),
        "activity_lag3": np.log1p(y.shift(3)),
    }, index=frame.index)
    for signal in ["negative_tone_index", "militarization_index", "escalation_index"]:
        for lag in [0, 1, 3]:
            design[f"{signal}_lag{lag}"] = frame[signal].shift(lag)
    design["target_week"] = design.index + pd.Timedelta(weeks=1)
    design["target"] = y.reindex(design["target_week"]).to_numpy()
    design["split"] = design["target_week"].map(splits)
    return design.dropna().rename_axis("issue_week").reset_index()


def forecast_results(panel, time_slices, repeats, seed):
    splits = time_slices.set_index("timestamp_start")["split"]
    metrics, predictions, design_frames, comparisons = [], [], [], []
    rng = np.random.default_rng(seed)
    for scope, frame in scope_frames(panel).items():
        if scope == "balanced_2021plus":
            continue
        for outcome in ["acled_event_count", "no_us_protests_count", "core_country_count"]:
            table = forecast_table(frame, outcome, splits)
            table["scope"], table["outcome"] = scope, outcome
            design_frames.append(table)
            train, valid, test = [table.loc[table["split"].eq(s)] for s in ["train", "valid", "test"]]
            if min(map(len, [train, valid, test])) < 10:
                continue
            history = ["activity_current", "activity_lag1", "activity_lag3"]
            text_features = [c for c in table if "_lag" in c and c not in history]
            predicted = {}
            for name, features in [("history_only", history), ("history_plus_news", history + text_features)]:
                candidates = []
                for alpha in [0.1, 1.0, 10.0, 100.0]:
                    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
                    model.fit(train[features], np.log1p(train["target"]))
                    val_pred = np.maximum(0, model.predict(valid[features]))
                    candidates.append((mean_absolute_error(np.log1p(valid["target"]), val_pred), alpha, model))
                val_mae, alpha, model = min(candidates, key=lambda item: (item[0], item[1]))
                pred_log = np.maximum(0, model.predict(test[features]))
                pred_count = np.expm1(pred_log)
                target = test["target"].to_numpy()
                predicted[name] = pred_count
                metrics.append({
                    "scope": scope, "outcome": outcome, "model": name, "alpha": alpha,
                    "n_train": len(train), "n_valid": len(valid), "n_test": len(test),
                    "train_target_end": train["target_week"].max(),
                    "test_target_start": test["target_week"].min(), "test_target_end": test["target_week"].max(),
                    "validation_log_mae": val_mae,
                    "log_mae": mean_absolute_error(np.log1p(target), pred_log),
                    "count_mae": mean_absolute_error(target, pred_count),
                    "count_rmse": np.sqrt(mean_squared_error(target, pred_count)),
                    "spearman_rho": spearmanr(target, pred_count).statistic,
                })
                records = test[["issue_week", "target_week", "target"]].copy()
                records["scope"], records["outcome"], records["model"] = scope, outcome, name
                records["prediction"] = pred_count
                predictions.append(records)
            if len(predicted) == 2:
                target = test["target"].to_numpy()
                for block in [4, 8]:
                    changes = []
                    for _ in range(repeats):
                        ix = block_indices(len(target), block, rng)
                        changes.append(np.sqrt(np.mean((predicted["history_plus_news"][ix] - target[ix]) ** 2)) -
                                       np.sqrt(np.mean((predicted["history_only"][ix] - target[ix]) ** 2)))
                    comparisons.append({
                        "scope": scope, "outcome": outcome, "block_weeks": block, "n_test": len(target),
                        "rmse_delta_news_minus_history": float(np.sqrt(np.mean((predicted["history_plus_news"] - target) ** 2)) -
                                                              np.sqrt(np.mean((predicted["history_only"] - target) ** 2))),
                        "ci_low": float(np.quantile(changes, .025)), "ci_high": float(np.quantile(changes, .975)),
                    })
    return pd.DataFrame(metrics), pd.concat(predictions, ignore_index=True), pd.concat(design_frames, ignore_index=True), pd.DataFrame(comparisons)


def stage_results(panel):
    from run_stage3_museed_plus import stage_sensitivity
    summaries, details = [], []
    for scope, frame in scope_frames(panel).items():
        if scope == "balanced_2021plus":
            continue
        summary, detail = stage_sensitivity(frame.reset_index())
        summary["scope"], detail["scope"] = scope, scope
        summaries.append(summary)
        details.append(detail)
    return pd.concat(summaries, ignore_index=True), pd.concat(details, ignore_index=True)


def review_candidates(events, seed):
    rows = []
    # Sampling creates a human review queue; no validation labels are fabricated.
    for label, group in events.groupby("strait_text_candidate"):
        size = min(80, len(group))
        sample = group.sample(size, random_state=seed).copy()
        sample["sampling_stratum"] = "text_hit" if label else "text_nonhit"
        sample["stratum_population"] = len(group)
        sample["inclusion_probability"] = size / len(group)
        for field in ["manual_relevance", "manual_event_category", "reviewer", "review_notes"]:
            sample[field] = ""
        rows.append(sample)
    return pd.concat(rows, ignore_index=True)


def make_figures(panel, coverage_matrix, lag, output):
    from matplotlib.colors import ListedColormap
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), gridspec_kw={"height_ratios": [1, 1.35]}, constrained_layout=True)
    ticks = np.arange(0, len(coverage_matrix), 52)
    axes[0].imshow(coverage_matrix[COUNTRIES].to_numpy().T.astype(float), aspect="auto",
                   interpolation="nearest", cmap=ListedColormap(["#e6e6e6", "#398477"]), vmin=0, vmax=1)
    axes[0].set_yticks(range(7), COUNTRIES)
    axes[0].set_xticks(ticks, coverage_matrix.index[ticks].strftime("%Y"))
    axes[0].set_title("A  Observed full-week ranges by country (gray: outside range)", loc="left")
    axes[1].plot(panel.index, panel["available_country_count_total"], color="#777777", lw=1,
                 label="Available-country total")
    axes[1].plot(panel.index, panel["acled_event_count"], color="#2676a5", lw=1, label="Seven-country common range")
    axes[1].plot(panel.index, panel["no_us_protests_count"], color="#a74e3b", lw=1,
                 label="Excluding US protests")
    axes[1].set_title("B  Event counts under different observation scopes", loc="left")
    axes[1].set_ylabel("Weekly events")
    axes[1].legend(frameon=False, fontsize=9)
    fig.savefig(output / "coverage_and_activity.png", dpi=220)
    plt.close(fig)
    selected = lag.loc[lag["signal"].eq("negative_tone_index") & lag["lead_weeks"].eq(4)]
    outcomes = ["acled_event_count", "no_us_protests_count", "core_country_count", "violence_count", "fatalities_sum"]
    names = ["Total events", "Excl. US protests", "China/Taiwan locations", "Violent events", "Fatalities"]
    scopes = ["historical_available", "balanced_2020plus", "balanced_2021plus"]
    fig, ax = plt.subplots(figsize=(9, 4.4), constrained_layout=True)
    for offset, scope, color, label in zip([-.23, 0, .23], scopes, ["#777777", "#2676a5", "#a74e3b"],
                                         ["Historical available", "Common range, 2020+", "Common range, 2021+"]):
        rows = selected.loc[selected["scope"].eq(scope)].set_index("outcome").reindex(outcomes)
        ax.barh(np.arange(len(outcomes)) + offset, rows["rho"], height=.21, color=color, label=label)
    ax.axvline(0, color="#333333", lw=.8)
    ax.set_yticks(range(len(outcomes)), names)
    ax.invert_yaxis()
    ax.set_xlabel("Spearman correlation: news tone at t, activity at t + 4 weeks")
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(output / "tone_scope_comparison.png", dpi=220)
    plt.close(fig)

    primary = panel.loc[panel["balanced_sample"]].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    series = {
        "China-Taiwan news edge": primary["china_taiwan_edge_weight"].rolling(4, min_periods=1).mean(),
        "Regional-core news edges": primary["regional_core_edge_weight"].rolling(4, min_periods=1).mean(),
    }
    for label, values in series.items():
        standardized = (values - values.mean()) / values.std(ddof=0)
        axes[0].plot(primary.index, standardized, lw=1.25, label=label)
    axes[0].axhline(0, color="#777777", lw=.7)
    axes[0].set_title("A  Standardized four-week moving averages", loc="left")
    axes[0].set_ylabel("Standard deviations")
    axes[0].legend(frameon=False, fontsize=9)

    pair = primary[["china_taiwan_edge_change", "regional_core_edge_change"]].dropna()
    axes[1].scatter(pair["china_taiwan_edge_change"], pair["regional_core_edge_change"],
                    s=17, alpha=.55, color="#2676a5", edgecolors="none")
    slope, intercept = np.polyfit(pair["china_taiwan_edge_change"], pair["regional_core_edge_change"], 1)
    xline = np.linspace(pair["china_taiwan_edge_change"].min(), pair["china_taiwan_edge_change"].max(), 100)
    axes[1].plot(xline, intercept + slope * xline, color="#a74e3b", lw=1.3)
    rho, _ = spearmanr(pair["china_taiwan_edge_change"], pair["regional_core_edge_change"])
    axes[1].set_title(f"B  Weekly first differences (Spearman rho={rho:.3f})", loc="left")
    axes[1].set_xlabel("Change in China-Taiwan news edge")
    axes[1].set_ylabel("Change in regional-core news edges")
    fig.savefig(output / "regional_core_news_synchrony.png", dpi=220)
    plt.close(fig)

def write_report(output, panel, coverage, lag, controlled, auxiliary, regional, audit):
    focus = lag.loc[lag["signal"].eq("negative_tone_index") & lag["lead_weeks"].eq(4),
                    ["scope", "outcome", "rho", "n", "q_iid", "q_hac8"]]
    ctrl = controlled.loc[controlled["signal"].eq("negative_tone_index") & controlled["lead_weeks"].eq(4),
                          ["scope", "outcome", "beta_per_signal_sd", "ci_low", "ci_high", "q_hac", "n"]]
    text = [
        "# Coverage-aware analysis", "",
        "## Observation scope", "",
        markdown_table(coverage), "",
        "The coverage intervals are inferred conservatively from the observed extract. "
        "They do not establish that every event within an interval was reported. "
        "The last complete week ends before the extract's last event date. "
        "Weeks outside a country's range retain missing outcomes.", "",
        "The historical_available scope has changing country composition and is descriptive only. "
        "The balanced_2020plus scope is the primary cross-source analysis. "
        "balanced_2021plus is a predeclared sensitivity excluding the whole of 2020.", "",
        "## Negative tone and future outcomes (four-week lead)", "", markdown_table(focus), "",
        "HAC p-values use standardized-rank regression with 4/8-week Newey-West covariance. "
        "BH adjustment includes all signals, outcomes and eight lead/lag values within each scope. "
        "These are exploratory associations with serial-dependence corrections, not causal effects.", "",
        "## Conditional associations", "", markdown_table(ctrl), "",
        "Models control current outcome, outcome lags 1/4, a linear trend, annual sine/cosine, "
        "and log GDELT event volume. The response is log(1 + future outcome); coefficients "
        "refer to one sample standard deviation of the signal. HAC bandwidth=max(8, lead+1).", "",
        "## Regional synchrony", "", markdown_table(regional), "",
        "Both edges are measured in GDELT and can share reporting shocks; synchrony describes "
        "news-network co-variation.", "",
        "## Auxiliary forecasts", "", markdown_table(auxiliary), "",
        "Forecast origin: end of completed week t; target: week t+1. "
        "Splits are assigned by target week using the original calendar boundaries. "
        "Scaling is fitted on training only. Ridge alpha is selected by validation log-MAE "
        "from 0.1/1/10/100; training stays frozen for test evaluation. "
        "Future labels outside observed full-week ranges are excluded.", "",
        "## Measurement limits", "",
        "- Negative tone is an event-record average of article-tone negativity; no topic/stance coding is claimed.",
        "- Equal-country and equal-core tone means reduce composition imbalance but share some source events.",
        "- Per-event attention controls event volume; it does not deduplicate articles or normalize the global media universe.",
        "- core_country_count includes domestic events in China/Taiwan and does not identify bilateral interaction.",
        "- strait_text_count is a keyword candidate series, pending manual relevance review.",
        "- Seven-country event totals are regional observation-background measures.",
        "- Historical neural MRR, seeds and ablations have not been rerun on the revised panel.",
        "## Validation", "", "~~~json", json.dumps(audit, ensure_ascii=False, indent=2), "~~~", "",
        "## Reproduction", "",
        "Run scripts/analyze_associations.py with --output-dir pointing to a fresh directory. "
        "Existing outputs are not overwritten. The manifest records hashes and versions.", "",
    ]
    (output / "coverage_analysis_report.md").write_text("\n".join(text), encoding="utf-8")


def validate_outputs(panel, coverage, country_weekly, designs, data):
    covered = panel["balanced_sample"]
    target_columns = [c for c in panel if c.startswith("acled_event_count_future_")]
    assert panel.loc[~covered, "acled_event_count"].isna().all()
    assert panel.iloc[-1][target_columns].isna().all()
    assert country_weekly.loc[~country_weekly["in_observed_full_week_range"], "event_count"].isna().all()
    assert designs["target"].notna().all()
    assert (designs["target_week"] - designs["issue_week"]).eq(pd.Timedelta(weeks=1)).all()
    assert designs["target_week"].max() <= panel.index[covered].max()
    train_end = data["time"].loc[data["time"]["split"].eq("train"), "timestamp_start"].max()
    assert designs.loc[designs["split"].eq("train"), "target_week"].max() <= train_end
    dates = panel.index[covered]
    return {
        "source_events": len(data["events"]), "weekly_rows": len(panel),
        "primary_weeks": int(covered.sum()), "primary_first_week": str(dates.min().date()),
        "primary_last_week": str(dates.max().date()),
        "historical_complete_weeks": int(panel["historical_available_sample"].sum()),
        "post_acled_calendar_weeks_masked": int((panel.index > data["time"]["timestamp_start"].max()).sum()),
        "partial_end_week_excluded": str(data["time"]["timestamp_start"].max().date()),
        "target_outside_observed_range": 0, "training_labels_after_training_cutoff": 0,
        "country_coverage_verified_externally": False, "keyword_relevance_manually_verified": False,
        "all_integrity_checks_passed": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/coverage_analysis")
    parser.add_argument("--bootstrap-repeats", type=int, default=999)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--refresh", action="store_true", help="Replace this run's generated outputs.")
    args = parser.parse_args()
    output = args.output_dir
    if (output / "manifest.json").exists() and not args.refresh:
        raise FileExistsError("Completed output exists; choose a fresh --output-dir.")
    output.mkdir(parents=True, exist_ok=True)
    print("Loading existing observations...", flush=True)
    data, paths = load_inputs()
    panel, coverage, country_weekly, matrix, events = build_panels(data)
    print(f"Primary common range: {int(panel.balanced_sample.sum())} complete weeks.", flush=True)
    lag = lag_results(panel)
    print("Lag comparisons complete; fitting conditional associations...", flush=True)
    controlled = controlled_results(panel)
    regional = regional_results(panel)
    boot = bootstrap_correlations(panel, args.bootstrap_repeats, args.seed)
    auxiliary, predictions, designs, comparisons = forecast_results(panel, data["time"], args.bootstrap_repeats, args.seed)
    print("Forecasts complete; recomputing stage sensitivity...", flush=True)
    stages, stage_details = stage_results(panel)
    candidates = review_candidates(events, args.seed)
    audit = validate_outputs(panel, coverage, country_weekly, designs, data)
    outputs = {
        "weekly_panel": panel.reset_index(), "country_coverage": coverage,
        "country_weekly": country_weekly, "lag_associations": lag,
        "conditional_associations": controlled, "regional_synchrony": regional,
        "block_bootstrap_correlations": boot, "auxiliary_forecast_metrics": auxiliary,
        "auxiliary_forecast_predictions": predictions, "forecast_design": designs,
        "forecast_rmse_comparison": comparisons, "stage_sensitivity": stages,
        "stage_segments": stage_details, "event_relevance_review_queue": candidates,
    }
    for name, frame in outputs.items():
        frame.to_csv(output / f"{name}.csv", index=False, encoding="utf-8-sig")
    make_figures(panel, matrix, lag, output)
    write_report(output, panel, coverage, lag, controlled, auxiliary, regional, audit)
    manifest_paths = paths
    manifest = {
        "seed": args.seed, "bootstrap_repeats": args.bootstrap_repeats,
        "primary_scope_rule": "latest first full week across observed country records to last globally complete week",
        "forecast_split_rule": "target_week mapped to original train/valid/test calendar",
        "ridge_alpha_grid": [0.1, 1.0, 10.0, 100.0],
        "selection_metric": "validation log MAE",
        "lag_grid": LAGS, "hac_bandwidths": [4, 8],
        "conditional_controls": ["outcome_t", "outcome_lag1", "outcome_lag4", "trend",
                                 "annual_sin", "annual_cos", "log_news_volume"],
        "inputs": {name: {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                   for name, p in manifest_paths.items()},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {"python": sys.version, "pandas": pd.__version__, "numpy": np.__version__,
                     "scipy": scipy.__version__, "statsmodels": sm.__version__},
        "audit": audit,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    print(f"Completed: {output}", flush=True)


if __name__ == "__main__":
    main()
