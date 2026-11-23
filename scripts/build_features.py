from __future__ import annotations

import argparse
import csv
import itertools
import math
import re
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GDELT_DIR = WORKSPACE_ROOT / "data" / "raw" / "gdelt"
DEFAULT_ACLED_PATH = WORKSPACE_ROOT / "data" / "raw" / "acled.csv"
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "data" / "MECHANISM_FEATURES_WEEKLY"
DEFAULT_START_DATE = "20160101"
DEFAULT_END_DATE = "20260528"

ENTITIES = {
    "China": {"actor": "CHN", "geo": "CH"},
    "Taiwan": {"actor": "TWN", "geo": "TW"},
    "Japan": {"actor": "JPN", "geo": "JA"},
    "South Korea": {"actor": "KOR", "geo": "KS"},
    "Philippines": {"actor": "PHL", "geo": "RP"},
    "Indonesia": {"actor": "IDN", "geo": "ID"},
    "United States": {"actor": "USA", "geo": "US"},
}

CORE_ENTITIES = {"China", "Taiwan"}
REGIONAL_ENTITIES = {"Japan", "South Korea", "Philippines", "Indonesia"}
EXTERNAL_ENTITIES = {"Japan", "South Korea", "Philippines", "Indonesia", "United States"}

ACLED_RELATED_FIELDS = (
    "country",
    "actor1",
    "actor2",
    "assoc_actor_1",
    "assoc_actor_2",
    "admin1",
    "admin2",
    "admin3",
    "location",
    "notes",
    "tags",
    "source",
)

ACLED_ENTITY_KEYWORDS = {
    "China": ("china", "chinese", "prc", "beijing", "pla", "people's liberation army", "peoples liberation army"),
    "Taiwan": ("taiwan", "taiwanese", "taipei", "roc", "taiwan strait", "kinmen", "matsu", "penghu"),
    "Japan": ("japan", "japanese", "tokyo"),
    "South Korea": ("south korea", "korea", "korean", "seoul", "rok"),
    "Philippines": ("philippines", "philippine", "filipino", "manila"),
    "Indonesia": ("indonesia", "indonesian", "jakarta"),
    "United States": ("united states", "u.s.", "usa", "america", "american", "washington"),
}

MILITARY_KEYWORDS = (
    "military",
    "drill",
    "exercise",
    "warship",
    "missile",
    "fighter",
    "navy",
    "air force",
    "coast guard",
    "patrol",
    "blockade",
)

GDELT_COLUMNS = [
    "SQLDATE",
    "Actor1CountryCode",
    "Actor1Type1Code",
    "Actor2CountryCode",
    "Actor2Type1Code",
    "EventRootCode",
    "QuadClass",
    "GoldsteinScale",
    "NumMentions",
    "NumSources",
    "NumArticles",
    "AvgTone",
    "ActionGeo_CountryCode",
]

MILITARY_PATTERN = "|".join(re.escape(keyword) for keyword in MILITARY_KEYWORDS)


def clean(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def safe_float(value: object, default: float = 0.0) -> float:
    text = clean(value)
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def safe_int(value: object, default: int = 0) -> int:
    text = clean(value)
    if not text:
        return default
    try:
        return int(float(text))
    except ValueError:
        return default


def parse_yyyymmdd(value: object) -> date | None:
    text = clean(value)
    if len(text) < 8:
        return None
    try:
        return datetime.strptime(text[:8], "%Y%m%d").date()
    except ValueError:
        return None


def parse_iso_date(value: object) -> date | None:
    text = clean(value)
    if not text:
        return None
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        return None


def bucket_date(day: date, granularity: str) -> date:
    if granularity == "day":
        return day
    if granularity == "week":
        return day - timedelta(days=day.weekday())
    if granularity == "month":
        return date(day.year, day.month, 1)
    raise ValueError(f"Unsupported granularity: {granularity}")


def gdelt_entities(row: pd.Series) -> set[str]:
    actor1 = clean(row.get("Actor1CountryCode"))
    actor2 = clean(row.get("Actor2CountryCode"))
    geo = clean(row.get("ActionGeo_CountryCode"))
    entities = set()
    for entity, codes in ENTITIES.items():
        if actor1 == codes["actor"] or actor2 == codes["actor"] or geo == codes["geo"]:
            entities.add(entity)
    return entities


def row_text(row: pd.Series) -> str:
    fields = [
        row.get("Actor1Name"),
        row.get("Actor2Name"),
        row.get("ActionGeo_FullName"),
        row.get("SOURCEURL"),
        row.get("EventCode"),
        row.get("EventBaseCode"),
        row.get("EventRootCode"),
    ]
    return " ".join(clean(field) for field in fields).lower()


def acled_row_text(row: Dict[str, str]) -> str:
    return " ".join(clean(row.get(field)) for field in ACLED_RELATED_FIELDS).lower()


def is_acled_related_to_entities(row: Dict[str, str], entities: set[str]) -> bool:
    if clean(row.get("country")) in entities:
        return True
    text = acled_row_text(row)
    for entity in entities:
        if any(keyword in text for keyword in ACLED_ENTITY_KEYWORDS.get(entity, ())):
            return True
    return False


def is_military_related(row: pd.Series) -> bool:
    text = row_text(row)
    return any(keyword in text for keyword in MILITARY_KEYWORDS)


def is_conflict_quad(quad_class: int) -> bool:
    return quad_class in {3, 4}


def is_cooperation_quad(quad_class: int) -> bool:
    return quad_class in {1, 2}


def iter_gdelt_frames(gdelt_dir: Path) -> Iterator[pd.DataFrame]:
    files = sorted(path for path in gdelt_dir.glob("gdelt_7entities_*.parquet") if not path.name.endswith(".partial"))
    for path in files:
        try:
            import pyarrow.parquet as pq

            parquet = pq.ParquetFile(path)
            available_columns = [column for column in GDELT_COLUMNS if column in parquet.schema.names]
            for idx in range(parquet.num_row_groups):
                yield parquet.read_row_group(idx, columns=available_columns).to_pandas()
        except Exception:
            yield pd.read_parquet(path, columns=GDELT_COLUMNS)


def blank_bucket_stats() -> Dict[str, float]:
    return defaultdict(float)


def add_gdelt_row(
    row: pd.Series,
    granularity: str,
    bucket_stats: Dict[date, Dict[str, float]],
    node_stats: Dict[tuple[date, str], Dict[str, float]],
    edge_stats: Dict[tuple[date, str, str], Dict[str, float]],
) -> None:
    event_day = parse_yyyymmdd(row.get("SQLDATE"))
    if event_day is None:
        return
    bucket = bucket_date(event_day, granularity)
    entities = gdelt_entities(row)
    if not entities:
        return

    mentions = safe_float(row.get("NumMentions"))
    sources = safe_float(row.get("NumSources"))
    articles = safe_float(row.get("NumArticles"))
    tone = safe_float(row.get("AvgTone"))
    goldstein = safe_float(row.get("GoldsteinScale"))
    quad_class = safe_int(row.get("QuadClass"))
    negative_tone = max(0.0, -tone)
    conflict = 1.0 if is_conflict_quad(quad_class) else 0.0
    cooperation = 1.0 if is_cooperation_quad(quad_class) else 0.0
    verbal_conflict = 1.0 if quad_class == 3 else 0.0
    material_conflict = 1.0 if quad_class == 4 else 0.0
    military = 1.0 if is_military_related(row) else 0.0
    attention = math.log1p(mentions) + math.log1p(articles) + math.log1p(sources)

    stats = bucket_stats.setdefault(bucket, blank_bucket_stats())
    stats["gdelt_event_count"] += 1
    stats["mention_sum"] += mentions
    stats["article_sum"] += articles
    stats["source_sum"] += sources
    stats["media_attention_index"] += attention
    stats["tone_sum"] += tone
    stats["tone_sq_sum"] += tone * tone
    stats["negative_tone_sum"] += negative_tone
    stats["goldstein_sum"] += goldstein
    stats["conflict_event_count"] += conflict
    stats["cooperation_event_count"] += cooperation
    stats["verbal_conflict_count"] += verbal_conflict
    stats["material_conflict_count"] += material_conflict
    stats["military_event_count"] += military
    stats["core_related_events"] += 1 if entities & CORE_ENTITIES else 0
    stats["regional_related_events"] += 1 if entities & REGIONAL_ENTITIES else 0
    stats["external_related_events"] += 1 if entities & EXTERNAL_ENTITIES else 0
    stats["us_related_events"] += 1 if "United States" in entities else 0
    stats["china_taiwan_events"] += 1 if {"China", "Taiwan"} <= entities else 0

    for entity in entities:
        node = node_stats.setdefault((bucket, entity), blank_bucket_stats())
        node["event_count"] += 1
        node["mention_sum"] += mentions
        node["negative_tone_sum"] += negative_tone
        node["conflict_event_count"] += conflict

    for left, right in itertools.combinations(sorted(entities), 2):
        edge = edge_stats.setdefault((bucket, left, right), blank_bucket_stats())
        edge["edge_event_count"] += 1
        edge["edge_mentions_sum"] += mentions
        edge["edge_tone_sum"] += tone
        edge["edge_goldstein_sum"] += goldstein
        edge["edge_conflict_count"] += conflict
        edge["edge_negative_count"] += 1 if tone < 0 else 0


def finalize_gdelt_features(bucket_stats: Dict[date, Dict[str, float]], edge_stats: Dict[tuple[date, str, str], Dict[str, float]]) -> pd.DataFrame:
    edge_by_bucket: Dict[date, list[Dict[str, float]]] = defaultdict(list)
    for (bucket, _, _), values in edge_stats.items():
        edge_by_bucket[bucket].append(values)

    rows = []
    for bucket, values in sorted(bucket_stats.items()):
        total = values["gdelt_event_count"]
        tone_mean = values["tone_sum"] / total if total else 0.0
        tone_var = values["tone_sq_sum"] / total - tone_mean * tone_mean if total else 0.0
        verbal = values["verbal_conflict_count"]
        material = values["material_conflict_count"]
        edges = edge_by_bucket.get(bucket, [])
        edge_total = sum(edge["edge_event_count"] for edge in edges)
        edge_conflict = sum(edge["edge_conflict_count"] for edge in edges)
        edge_negative = sum(edge["edge_negative_count"] for edge in edges)

        rows.append(
            {
                "timestamp_start": bucket.isoformat(),
                "gdelt_event_count": int(total),
                "mention_sum": values["mention_sum"],
                "article_sum": values["article_sum"],
                "source_sum": values["source_sum"],
                "media_attention_index": values["media_attention_index"],
                "avg_tone": tone_mean,
                "negative_tone_index": values["negative_tone_sum"] / total if total else 0.0,
                "tone_volatility": math.sqrt(max(0.0, tone_var)),
                "goldstein_mean": values["goldstein_sum"] / total if total else 0.0,
                "conflict_event_count": int(values["conflict_event_count"]),
                "cooperation_event_count": int(values["cooperation_event_count"]),
                "verbal_conflict_count": int(verbal),
                "material_conflict_count": int(material),
                "conflict_ratio": values["conflict_event_count"] / total if total else 0.0,
                "escalation_index": material / (verbal + 1.0),
                "military_event_count": int(values["military_event_count"]),
                "militarization_index": values["military_event_count"] / total if total else 0.0,
                "us_intervention_index": values["us_related_events"] / total if total else 0.0,
                "external_intervention_index": values["external_related_events"] / total if total else 0.0,
                "regional_linkage_index": values["regional_related_events"] / total if total else 0.0,
                "core_entity_visibility": values["core_related_events"] / total if total else 0.0,
                "external_entity_dominance": values["external_related_events"] / (values["core_related_events"] + 1.0),
                "core_relation_ratio": values["china_taiwan_events"] / total if total else 0.0,
                "active_edge_density": len(edges) / 21.0,
                "conflict_edge_ratio": edge_conflict / edge_total if edge_total else 0.0,
                "negative_edge_ratio": edge_negative / edge_total if edge_total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def finalize_node_features(node_stats: Dict[tuple[date, str], Dict[str, float]], edge_stats: Dict[tuple[date, str, str], Dict[str, float]]) -> pd.DataFrame:
    weighted_degree: Dict[tuple[date, str], float] = defaultdict(float)
    neighbors: Dict[tuple[date, str], set[str]] = defaultdict(set)
    conflict_degree: Dict[tuple[date, str], float] = defaultdict(float)

    for (bucket, left, right), values in edge_stats.items():
        weight = values["edge_event_count"]
        conflict = values["edge_conflict_count"]
        weighted_degree[(bucket, left)] += weight
        weighted_degree[(bucket, right)] += weight
        conflict_degree[(bucket, left)] += conflict
        conflict_degree[(bucket, right)] += conflict
        neighbors[(bucket, left)].add(right)
        neighbors[(bucket, right)].add(left)

    rows = []
    buckets = sorted({bucket for bucket, _ in node_stats} | {bucket for bucket, _ in weighted_degree})
    for bucket in buckets:
        for entity in ENTITIES:
            values = node_stats.get((bucket, entity), {})
            event_count = values.get("event_count", 0.0)
            rows.append(
                {
                    "timestamp_start": bucket.isoformat(),
                    "entity": entity,
                    "entity_event_count": int(event_count),
                    "entity_mentions_sum": values.get("mention_sum", 0.0),
                    "entity_negative_tone_index": values.get("negative_tone_sum", 0.0) / event_count if event_count else 0.0,
                    "degree_centrality": len(neighbors.get((bucket, entity), set())) / 6.0,
                    "weighted_degree": weighted_degree.get((bucket, entity), 0.0),
                    "conflict_centrality": conflict_degree.get((bucket, entity), 0.0),
                }
            )
    return pd.DataFrame(rows)


def finalize_edge_features(edge_stats: Dict[tuple[date, str, str], Dict[str, float]]) -> pd.DataFrame:
    rows = []
    for (bucket, left, right), values in sorted(edge_stats.items()):
        count = values["edge_event_count"]
        rows.append(
            {
                "timestamp_start": bucket.isoformat(),
                "entity_left": left,
                "entity_right": right,
                "edge_event_count": int(count),
                "edge_mentions_sum": values["edge_mentions_sum"],
                "edge_avg_tone": values["edge_tone_sum"] / count if count else 0.0,
                "edge_goldstein_mean": values["edge_goldstein_sum"] / count if count else 0.0,
                "edge_conflict_ratio": values["edge_conflict_count"] / count if count else 0.0,
                "edge_negative_ratio": values["edge_negative_count"] / count if count else 0.0,
                "edge_weight": math.log1p(count),
                "edge_risk_weight": math.log1p(values["edge_mentions_sum"]) * (values["edge_conflict_count"] / count if count else 0.0),
            }
        )
    return pd.DataFrame(rows)


def build_gdelt_features(
    gdelt_dir: Path,
    granularity: str,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    bucket_frames: list[pd.DataFrame] = []
    node_frames: list[pd.DataFrame] = []
    edge_frames: list[pd.DataFrame] = []
    frame_count = 0
    row_count = 0

    for frame in iter_gdelt_frames(gdelt_dir):
        frame_count += 1
        row_count += len(frame)
        bucket_frame, node_frame, edge_frame = aggregate_gdelt_frame(frame, granularity, start_date, end_date)
        if not bucket_frame.empty:
            bucket_frames.append(bucket_frame)
        if not node_frame.empty:
            node_frames.append(node_frame)
        if not edge_frame.empty:
            edge_frames.append(edge_frame)
        if frame_count == 1 or frame_count % 20 == 0:
            print(f"Processed GDELT row groups: {frame_count}, rows: {row_count}", flush=True)

    print(f"Finished GDELT aggregation input: {frame_count} row groups, rows: {row_count}", flush=True)

    bucket_agg = aggregate_grouped_frames(bucket_frames, ["bucket_start"])
    node_agg = aggregate_grouped_frames(node_frames, ["bucket_start", "entity"])
    edge_agg = aggregate_grouped_frames(edge_frames, ["bucket_start", "entity_left", "entity_right"])

    return (
        finalize_gdelt_features_from_frames(bucket_agg, edge_agg),
        finalize_node_features_from_frames(node_agg, edge_agg),
        finalize_edge_features_from_frames(edge_agg),
        {"gdelt_row_groups": frame_count, "gdelt_input_rows": row_count},
    )


def aggregate_grouped_frames(frames: list[pd.DataFrame], group_keys: Sequence[str]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    value_columns = [column for column in merged.columns if column not in group_keys]
    return merged.groupby(list(group_keys), as_index=False)[value_columns].sum()


def bucket_timestamp_series(days: pd.Series, granularity: str) -> pd.Series:
    if granularity == "day":
        return days.dt.normalize()
    if granularity == "week":
        return (days - pd.to_timedelta(days.dt.weekday, unit="D")).dt.normalize()
    if granularity == "month":
        return days.dt.to_period("M").dt.to_timestamp()
    raise ValueError(f"Unsupported granularity: {granularity}")


def aggregate_gdelt_frame(
    frame: pd.DataFrame,
    granularity: str,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df = frame.copy()
    event_day = pd.to_datetime(df["SQLDATE"].astype(str).str[:8], format="%Y%m%d", errors="coerce")
    start_ts = pd.to_datetime(start_date, format="%Y%m%d")
    end_ts = pd.to_datetime(end_date, format="%Y%m%d")
    valid_dates = event_day.notna() & (event_day >= start_ts) & (event_day <= end_ts)
    df = df.loc[valid_dates].copy()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df["bucket_start"] = bucket_timestamp_series(event_day.loc[df.index], granularity)
    df["mentions"] = pd.to_numeric(df["NumMentions"], errors="coerce").fillna(0.0)
    df["sources"] = pd.to_numeric(df["NumSources"], errors="coerce").fillna(0.0)
    df["articles"] = pd.to_numeric(df["NumArticles"], errors="coerce").fillna(0.0)
    df["tone"] = pd.to_numeric(df["AvgTone"], errors="coerce").fillna(0.0)
    df["goldstein"] = pd.to_numeric(df["GoldsteinScale"], errors="coerce").fillna(0.0)
    df["quad_class"] = pd.to_numeric(df["QuadClass"], errors="coerce").fillna(0).astype(int)

    df["negative_tone"] = (-df["tone"]).clip(lower=0.0)
    df["conflict"] = df["quad_class"].isin([3, 4]).astype(int)
    df["cooperation"] = df["quad_class"].isin([1, 2]).astype(int)
    df["verbal_conflict"] = (df["quad_class"] == 3).astype(int)
    df["material_conflict"] = (df["quad_class"] == 4).astype(int)
    df["media_attention"] = np.log1p(df["mentions"]) + np.log1p(df["articles"]) + np.log1p(df["sources"])
    event_root = pd.to_numeric(df["EventRootCode"], errors="coerce").fillna(0).astype(int)
    actor1_type = df["Actor1Type1Code"].fillna("").astype(str)
    actor2_type = df["Actor2Type1Code"].fillna("").astype(str)
    # CAMEO root 15 captures military posture; MIL actor codes catch military-linked actors.
    df["military"] = ((event_root == 15) | (actor1_type == "MIL") | (actor2_type == "MIL")).astype(int)

    actor1 = df["Actor1CountryCode"].fillna("").astype(str)
    actor2 = df["Actor2CountryCode"].fillna("").astype(str)
    geo = df["ActionGeo_CountryCode"].fillna("").astype(str)

    entity_masks: Dict[str, pd.Series] = {}
    any_entity = pd.Series(False, index=df.index)
    for entity, codes in ENTITIES.items():
        mask = (actor1 == codes["actor"]) | (actor2 == codes["actor"]) | (geo == codes["geo"])
        entity_masks[entity] = mask
        any_entity = any_entity | mask

    df = df.loc[any_entity].copy()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    entity_masks = {entity: mask.loc[df.index] for entity, mask in entity_masks.items()}

    df["core_related_events"] = pd.concat([entity_masks[entity] for entity in CORE_ENTITIES], axis=1).any(axis=1).astype(int)
    df["regional_related_events"] = pd.concat([entity_masks[entity] for entity in REGIONAL_ENTITIES], axis=1).any(axis=1).astype(int)
    df["external_related_events"] = pd.concat([entity_masks[entity] for entity in EXTERNAL_ENTITIES], axis=1).any(axis=1).astype(int)
    df["us_related_events"] = entity_masks["United States"].astype(int)
    df["china_taiwan_events"] = (entity_masks["China"] & entity_masks["Taiwan"]).astype(int)
    df["tone_sq"] = df["tone"] * df["tone"]

    bucket_frame = (
        df.groupby("bucket_start", as_index=False)
        .agg(
            gdelt_event_count=("bucket_start", "size"),
            mention_sum=("mentions", "sum"),
            article_sum=("articles", "sum"),
            source_sum=("sources", "sum"),
            media_attention_index=("media_attention", "sum"),
            tone_sum=("tone", "sum"),
            tone_sq_sum=("tone_sq", "sum"),
            negative_tone_sum=("negative_tone", "sum"),
            goldstein_sum=("goldstein", "sum"),
            conflict_event_count=("conflict", "sum"),
            cooperation_event_count=("cooperation", "sum"),
            verbal_conflict_count=("verbal_conflict", "sum"),
            material_conflict_count=("material_conflict", "sum"),
            military_event_count=("military", "sum"),
            core_related_events=("core_related_events", "sum"),
            regional_related_events=("regional_related_events", "sum"),
            external_related_events=("external_related_events", "sum"),
            us_related_events=("us_related_events", "sum"),
            china_taiwan_events=("china_taiwan_events", "sum"),
        )
    )

    node_frames: list[pd.DataFrame] = []
    for entity, mask in entity_masks.items():
        sub = df.loc[mask, ["bucket_start", "mentions", "negative_tone", "conflict"]]
        if sub.empty:
            continue
        grouped = (
            sub.groupby("bucket_start", as_index=False)
            .agg(
                event_count=("bucket_start", "size"),
                mention_sum=("mentions", "sum"),
                negative_tone_sum=("negative_tone", "sum"),
                conflict_event_count=("conflict", "sum"),
            )
        )
        grouped["entity"] = entity
        node_frames.append(grouped)

    edge_frames: list[pd.DataFrame] = []
    for left, right in itertools.combinations(sorted(ENTITIES), 2):
        pair_mask = entity_masks[left] & entity_masks[right]
        sub = df.loc[pair_mask, ["bucket_start", "mentions", "tone", "goldstein", "conflict"]]
        if sub.empty:
            continue
        grouped = (
            sub.groupby("bucket_start", as_index=False)
            .agg(
                edge_event_count=("bucket_start", "size"),
                edge_mentions_sum=("mentions", "sum"),
                edge_tone_sum=("tone", "sum"),
                edge_goldstein_sum=("goldstein", "sum"),
                edge_conflict_count=("conflict", "sum"),
            )
        )
        grouped["edge_negative_count"] = (
            sub.assign(negative=(sub["tone"] < 0).astype(int))
            .groupby("bucket_start")["negative"]
            .sum()
            .reindex(grouped["bucket_start"])
            .to_numpy()
        )
        grouped["entity_left"] = left
        grouped["entity_right"] = right
        edge_frames.append(grouped)

    node_frame = pd.concat(node_frames, ignore_index=True) if node_frames else pd.DataFrame()
    edge_frame = pd.concat(edge_frames, ignore_index=True) if edge_frames else pd.DataFrame()
    return bucket_frame, node_frame, edge_frame


def finalize_gdelt_features_from_frames(bucket_frame: pd.DataFrame, edge_frame: pd.DataFrame) -> pd.DataFrame:
    if bucket_frame.empty:
        return pd.DataFrame()

    edge_totals = pd.DataFrame(columns=["bucket_start", "edge_event_count_sum", "edge_conflict_count_sum", "edge_negative_count_sum"])
    if not edge_frame.empty:
        edge_totals = (
            edge_frame.groupby("bucket_start", as_index=False)
            .agg(
                edge_event_count_sum=("edge_event_count", "sum"),
                edge_conflict_count_sum=("edge_conflict_count", "sum"),
                edge_negative_count_sum=("edge_negative_count", "sum"),
            )
        )

    merged = pd.merge(bucket_frame, edge_totals, on="bucket_start", how="left").fillna(0)
    rows = []
    for _, item in merged.sort_values("bucket_start").iterrows():
        total = item["gdelt_event_count"]
        tone_mean = item["tone_sum"] / total if total else 0.0
        tone_var = item["tone_sq_sum"] / total - tone_mean * tone_mean if total else 0.0
        verbal = item["verbal_conflict_count"]
        material = item["material_conflict_count"]
        edge_total = item["edge_event_count_sum"]
        rows.append(
            {
                "timestamp_start": item["bucket_start"].date().isoformat(),
                "gdelt_event_count": int(total),
                "mention_sum": item["mention_sum"],
                "article_sum": item["article_sum"],
                "source_sum": item["source_sum"],
                "media_attention_index": item["media_attention_index"],
                "avg_tone": tone_mean,
                "negative_tone_index": item["negative_tone_sum"] / total if total else 0.0,
                "tone_volatility": math.sqrt(max(0.0, tone_var)),
                "goldstein_mean": item["goldstein_sum"] / total if total else 0.0,
                "conflict_event_count": int(item["conflict_event_count"]),
                "cooperation_event_count": int(item["cooperation_event_count"]),
                "verbal_conflict_count": int(verbal),
                "material_conflict_count": int(material),
                "conflict_ratio": item["conflict_event_count"] / total if total else 0.0,
                "escalation_index": material / (verbal + 1.0),
                "military_event_count": int(item["military_event_count"]),
                "militarization_index": item["military_event_count"] / total if total else 0.0,
                "us_intervention_index": item["us_related_events"] / total if total else 0.0,
                "external_intervention_index": item["external_related_events"] / total if total else 0.0,
                "regional_linkage_index": item["regional_related_events"] / total if total else 0.0,
                "core_entity_visibility": item["core_related_events"] / total if total else 0.0,
                "external_entity_dominance": item["external_related_events"] / (item["core_related_events"] + 1.0),
                "core_relation_ratio": item["china_taiwan_events"] / total if total else 0.0,
                "active_edge_density": len(edge_frame.loc[edge_frame["bucket_start"] == item["bucket_start"]]) / 21.0 if not edge_frame.empty else 0.0,
                "conflict_edge_ratio": item["edge_conflict_count_sum"] / edge_total if edge_total else 0.0,
                "negative_edge_ratio": item["edge_negative_count_sum"] / edge_total if edge_total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def finalize_node_features_from_frames(node_frame: pd.DataFrame, edge_frame: pd.DataFrame) -> pd.DataFrame:
    if node_frame.empty:
        return pd.DataFrame()

    weighted_degree: Dict[tuple[pd.Timestamp, str], float] = defaultdict(float)
    neighbors: Dict[tuple[pd.Timestamp, str], set[str]] = defaultdict(set)
    conflict_degree: Dict[tuple[pd.Timestamp, str], float] = defaultdict(float)
    for _, item in edge_frame.iterrows():
        bucket = item["bucket_start"]
        left = item["entity_left"]
        right = item["entity_right"]
        weight = item["edge_event_count"]
        conflict = item["edge_conflict_count"]
        weighted_degree[(bucket, left)] += weight
        weighted_degree[(bucket, right)] += weight
        conflict_degree[(bucket, left)] += conflict
        conflict_degree[(bucket, right)] += conflict
        neighbors[(bucket, left)].add(right)
        neighbors[(bucket, right)].add(left)

    rows = []
    for _, item in node_frame.sort_values(["bucket_start", "entity"]).iterrows():
        bucket = item["bucket_start"]
        entity = item["entity"]
        event_count = item["event_count"]
        rows.append(
            {
                "timestamp_start": bucket.date().isoformat(),
                "entity": entity,
                "entity_event_count": int(event_count),
                "entity_mentions_sum": item["mention_sum"],
                "entity_negative_tone_index": item["negative_tone_sum"] / event_count if event_count else 0.0,
                "degree_centrality": len(neighbors.get((bucket, entity), set())) / 6.0,
                "weighted_degree": weighted_degree.get((bucket, entity), 0.0),
                "conflict_centrality": conflict_degree.get((bucket, entity), 0.0),
            }
        )
    return pd.DataFrame(rows)


def finalize_edge_features_from_frames(edge_frame: pd.DataFrame) -> pd.DataFrame:
    if edge_frame.empty:
        return pd.DataFrame()

    rows = []
    for _, item in edge_frame.sort_values(["bucket_start", "entity_left", "entity_right"]).iterrows():
        count = item["edge_event_count"]
        rows.append(
            {
                "timestamp_start": item["bucket_start"].date().isoformat(),
                "entity_left": item["entity_left"],
                "entity_right": item["entity_right"],
                "edge_event_count": int(count),
                "edge_mentions_sum": item["edge_mentions_sum"],
                "edge_avg_tone": item["edge_tone_sum"] / count if count else 0.0,
                "edge_goldstein_mean": item["edge_goldstein_sum"] / count if count else 0.0,
                "edge_conflict_ratio": item["edge_conflict_count"] / count if count else 0.0,
                "edge_negative_ratio": item["edge_negative_count"] / count if count else 0.0,
                "edge_weight": math.log1p(count),
                "edge_risk_weight": math.log1p(item["edge_mentions_sum"]) * (item["edge_conflict_count"] / count if count else 0.0),
            }
        )
    return pd.DataFrame(rows)


def acled_event_type_counts(rows: Sequence[Dict[str, str]]) -> Dict[str, int]:
    counts = {
        "violence_count": 0,
        "protest_count": 0,
        "strategic_development_count": 0,
    }
    for row in rows:
        event_type = clean(row.get("event_type"))
        disorder_type = clean(row.get("disorder_type"))
        if event_type in {"Battles", "Explosions/Remote violence", "Riots", "Violence against civilians"} or disorder_type == "Political violence":
            counts["violence_count"] += 1
        if event_type == "Protests":
            counts["protest_count"] += 1
        if event_type == "Strategic developments":
            counts["strategic_development_count"] += 1
    return counts


ACLED_SIGNAL_COLUMNS = [
    "acled_event_count",
    "fatalities_sum",
    "violence_count",
    "protest_count",
    "strategic_development_count",
    "risk_level",
]


def acled_label_columns(granularity: str) -> tuple[str, str, str, str]:
    if granularity == "week":
        return "risk_t_plus_7", "risk_t_plus_14", "risk_count_t_plus_7", "risk_count_t_plus_14"
    return "risk_t_plus_1", "risk_t_plus_2", "risk_count_t_plus_1", "risk_count_t_plus_2"


def continuous_time_index(start_value: str, end_value: str, granularity: str) -> pd.DatetimeIndex:
    start = pd.to_datetime(start_value)
    end = pd.to_datetime(end_value)
    if granularity == "day":
        return pd.date_range(start=start, end=end, freq="D")
    if granularity == "week":
        return pd.date_range(start=start, end=end, freq="W-MON")
    if granularity == "month":
        return pd.date_range(start=start, end=end, freq="MS")
    raise ValueError(f"Unsupported granularity: {granularity}")


def complete_acled_time_axis(frame: pd.DataFrame, granularity: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    full_index = continuous_time_index(frame["timestamp_start"].min(), frame["timestamp_start"].max(), granularity)
    frame = frame.copy()
    frame["_timestamp"] = pd.to_datetime(frame["timestamp_start"])
    frame = frame.set_index("_timestamp").drop(columns=["timestamp_start"]).reindex(full_index)
    for column in ACLED_SIGNAL_COLUMNS:
        if column not in frame.columns:
            frame[column] = 0
    frame[ACLED_SIGNAL_COLUMNS] = frame[ACLED_SIGNAL_COLUMNS].fillna(0)
    frame[ACLED_SIGNAL_COLUMNS] = frame[ACLED_SIGNAL_COLUMNS].astype(int)
    frame = frame.reset_index(names="timestamp_start")
    frame["timestamp_start"] = frame["timestamp_start"].dt.date.astype(str)
    return frame


def build_acled_features(acled_path: Path, granularity: str, acled_filter_mode: str = "country") -> pd.DataFrame:
    buckets: Dict[date, list[Dict[str, str]]] = defaultdict(list)
    countries = set(ENTITIES)

    with acled_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter=";")
        for row in reader:
            if acled_filter_mode == "country":
                include = clean(row.get("country")) in countries
            elif acled_filter_mode == "related":
                include = is_acled_related_to_entities(row, countries)
            else:
                raise ValueError(f"Unsupported ACLED filter mode: {acled_filter_mode}")
            if not include:
                continue
            event_day = parse_iso_date(row.get("event_date"))
            if event_day is None:
                continue
            buckets[bucket_date(event_day, granularity)].append(row)

    rows = []
    for bucket, bucket_rows in sorted(buckets.items()):
        event_count = len(bucket_rows)
        fatalities_sum = sum(safe_int(row.get("fatalities")) for row in bucket_rows)
        counts = acled_event_type_counts(bucket_rows)
        if event_count == 0:
            risk_level = 0
        elif fatalities_sum > 0 or counts["violence_count"] > 0:
            risk_level = 3
        elif event_count >= 3 or counts["strategic_development_count"] > 0:
            risk_level = 2
        else:
            risk_level = 1

        rows.append(
            {
                "timestamp_start": bucket.isoformat(),
                "acled_event_count": event_count,
                "fatalities_sum": fatalities_sum,
                **counts,
                "risk_level": risk_level,
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    frame = complete_acled_time_axis(frame, granularity)
    risk_1, risk_2, count_1, count_2 = acled_label_columns(granularity)
    next_risk = frame["risk_level"].shift(-1)
    second_risk = frame["risk_level"].shift(-2)
    frame[risk_1] = next_risk.gt(0).astype(float).where(next_risk.notna())
    frame[risk_2] = (next_risk.gt(0) | second_risk.gt(0)).astype(float).where(
        next_risk.notna() & second_risk.notna()
    )
    frame[count_1] = frame["acled_event_count"].shift(-1)
    frame[count_2] = frame["acled_event_count"].shift(-1) + frame["acled_event_count"].shift(-2)
    frame["acled_observed"] = True
    return frame


def add_lag_features(frame: pd.DataFrame, columns: Sequence[str], lags: Sequence[int]) -> pd.DataFrame:
    frame = frame.sort_values("timestamp_start").reset_index(drop=True)
    for column in columns:
        if column not in frame.columns:
            continue
        for lag in lags:
            frame[f"{column}_lag_{lag}"] = frame[column].shift(lag)
    return frame


def merge_features(gdelt_frame: pd.DataFrame, acled_frame: pd.DataFrame) -> pd.DataFrame:
    if gdelt_frame.empty:
        merged = acled_frame.copy()
    elif acled_frame.empty:
        merged = gdelt_frame.copy()
    else:
        merged = pd.merge(gdelt_frame, acled_frame, on="timestamp_start", how="outer")

    if merged.empty:
        return merged
    merged = merged.sort_values("timestamp_start").reset_index(drop=True)
    # An unmatched source window is unobserved; zero denotes observed absence.
    merged["acled_observed"] = merged.get(
        "acled_event_count", pd.Series(np.nan, index=merged.index)
    ).notna()
    merged["gdelt_observed"] = merged.get(
        "gdelt_event_count", pd.Series(np.nan, index=merged.index)
    ).notna()

    # Synchronization features compare same-window news volume with ACLED behavior volume.
    # They are retained for mechanism/synchronous analysis. Prediction uses lagged
    # versions only through load_mechanism_features(..., lag_only=True).
    if "gdelt_event_count" in merged.columns and "acled_event_count" in merged.columns:
        denom = merged["acled_event_count"] + 1.0
        merged["gdelt_to_acled_event_ratio"] = merged["gdelt_event_count"] / denom
        if "media_attention_index" in merged.columns:
            merged["attention_per_acled_event"] = merged["media_attention_index"] / denom
        if "conflict_event_count" in merged.columns:
            merged["gdelt_conflict_per_acled_event"] = merged["conflict_event_count"] / denom
        if "military_event_count" in merged.columns:
            merged["military_signal_per_acled_event"] = merged["military_event_count"] / denom
        if "external_related_events" in merged.columns:
            merged["external_signal_per_acled_event"] = merged["external_related_events"] / denom
        if "regional_related_events" in merged.columns:
            merged["regional_signal_per_acled_event"] = merged["regional_related_events"] / denom

    merged = add_lag_features(
        merged,
        [
            "media_attention_index",
            "negative_tone_index",
            "conflict_ratio",
            "escalation_index",
            "militarization_index",
            "us_intervention_index",
            "external_intervention_index",
            "regional_linkage_index",
            "core_entity_visibility",
            "external_entity_dominance",
            "active_edge_density",
            "gdelt_to_acled_event_ratio",
            "attention_per_acled_event",
            "gdelt_conflict_per_acled_event",
            "acled_event_count",
            "risk_level",
        ],
        [1, 2, 4],
    )
    return merged


def build_prediction_feature_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Build a leakage-controlled table for forecasting experiments.

    The full mechanism table keeps same-window ACLED fields for descriptive and
    synchronous mechanism analysis. This table removes current-window ACLED
    outcomes and same-window GDELT/ACLED ratio features, while retaining labels
    and lagged history features.
    """
    label_cols = {
        column for column in frame.columns
        if column.startswith(("risk_t_plus_", "risk_count_t_plus_"))
    }
    forbidden_current_acled_cols = {
        "acled_event_count",
        "fatalities_sum",
        "violence_count",
        "protest_count",
        "strategic_development_count",
        "risk_level",
        "gdelt_to_acled_event_ratio",
        "attention_per_acled_event",
        "gdelt_conflict_per_acled_event",
        "military_signal_per_acled_event",
        "external_signal_per_acled_event",
        "regional_signal_per_acled_event",
    }

    keep_cols = []
    seen = set()

    def keep(column: str) -> None:
        if column in frame.columns and column not in seen:
            keep_cols.append(column)
            seen.add(column)

    keep("timestamp_start")
    for column in frame.columns:
        if column == "timestamp_start":
            continue
        if column in label_cols:
            keep(column)
        elif column.endswith(("_lag_1", "_lag_2", "_lag_4")):
            keep(column)
        elif column not in forbidden_current_acled_cols and not column.startswith(("risk_t_plus", "risk_count_t_plus")):
            keep(column)

    return frame[keep_cols].copy()


def write_alignment_report(
    output_dir: Path,
    args: argparse.Namespace,
    gdelt_summary: Dict[str, int],
    gdelt_frame: pd.DataFrame,
    acled_frame: pd.DataFrame,
    mechanism_frame: pd.DataFrame,
    node_frame: pd.DataFrame,
    edge_frame: pd.DataFrame,
) -> None:
    gdelt_windows = set(gdelt_frame.get("timestamp_start", pd.Series(dtype=str)).astype(str))
    acled_windows = set(acled_frame.get("timestamp_start", pd.Series(dtype=str)).astype(str))
    acled_active_windows = set(
        acled_frame.loc[acled_frame.get("acled_event_count", pd.Series(dtype=float)) > 0, "timestamp_start"].astype(str)
    ) if "timestamp_start" in acled_frame.columns and "acled_event_count" in acled_frame.columns else set()
    overlap_windows = gdelt_windows & acled_windows
    missing_acled_windows = gdelt_windows - acled_windows
    missing_gdelt_windows = acled_windows - gdelt_windows

    def positive_ratio(frame: pd.DataFrame, column: str) -> float:
        if column not in frame.columns or len(frame) == 0:
            return 0.0
        return float((frame[column] > 0).mean())

    def zero_count(frame: pd.DataFrame, column: str) -> int:
        if column not in frame.columns:
            return 0
        return int((frame[column] == 0).sum())

    label_lines = []
    for column in [
        column for column in acled_frame.columns
        if column.startswith(("risk_t_plus_", "risk_count_t_plus_"))
    ]:
        label_lines.append(f"- {column} positive ratio: {positive_ratio(acled_frame, column):.6f}")

    rows = [
        "# Alignment Report",
        "",
        "This report records how GDELT news-signal features and ACLED behavior labels are aligned.",
        "",
        "## Settings",
        "",
        f"- granularity: `{args.granularity}`",
        f"- GDELT SQLDATE window: `{args.start_date}` to `{args.end_date}`",
        f"- ACLED filter mode: `{args.acled_filter_mode}`",
        "",
        "## Coverage",
        "",
        f"- GDELT row groups: {gdelt_summary.get('gdelt_row_groups', 0)}",
        f"- GDELT input rows: {gdelt_summary.get('gdelt_input_rows', 0)}",
        f"- GDELT active windows: {len(gdelt_windows)}",
        f"- ACLED calendar windows: {len(acled_windows)}",
        f"- ACLED active windows: {len(acled_active_windows)}",
        f"- overlap calendar windows: {len(overlap_windows)}",
        f"- GDELT windows without ACLED calendar: {len(missing_acled_windows)}",
        f"- ACLED calendar windows without GDELT: {len(missing_gdelt_windows)}",
        f"- ACLED empty calendar windows: {len(acled_windows - acled_active_windows)}",
        f"- mechanism rows: {len(mechanism_frame)}",
        f"- node rows: {len(node_frame)}",
        f"- edge rows: {len(edge_frame)}",
        f"- mechanism rows with zero GDELT events: {zero_count(mechanism_frame, 'gdelt_event_count')}",
        f"- mechanism rows with zero ACLED events: {zero_count(mechanism_frame, 'acled_event_count')}",
        "",
        "## Label Balance",
        "",
        *(label_lines or ["- no future label columns found"]),
        "",
        "## Modeling Note",
        "",
        "`mechanism_features.csv` is for mechanism and synchronous analysis. "
        "`prediction_features_strict.csv` removes current-window ACLED outcomes and same-window "
        "GDELT/ACLED ratio features for forecasting experiments.",
        "",
        "ACLED future labels are generated after completing the calendar time axis. "
        "For weekly tables, `risk_t_plus_7` is the next calendar week and `risk_t_plus_14` "
        "covers the next two calendar weeks; month/day tables use `risk_t_plus_1` and "
        "`risk_t_plus_2` naming.",
        "",
    ]
    (output_dir / "alignment_report.md").write_text("\n".join(rows), encoding="utf-8")


def write_feature_card(output_dir: Path, args: argparse.Namespace, summary: Dict[str, int]) -> None:
    lines = [
        "# Mechanism Features",
        "",
        "Feature tables generated for MUSE-ED cross-strait event forecasting and mechanism analysis.",
        "",
        "## Inputs",
        "",
        f"- GDELT directory: `{Path(args.gdelt_dir).resolve()}`",
        f"- ACLED file: `{Path(args.acled).resolve()}`",
        f"- granularity: `{args.granularity}`",
        f"- GDELT SQLDATE window: `{args.start_date}` to `{args.end_date}`",
        f"- ACLED filter mode: `{args.acled_filter_mode}`",
        "",
        "## Outputs",
        "",
        "- `mechanism_features.csv`: time-window features and ACLED future-risk labels.",
        "- `prediction_features_strict.csv`: leakage-controlled forecasting features.",
        "- `entity_node_features.csv`: seven-entity node features and centrality proxies.",
        "- `entity_edge_features.csv`: pairwise seven-entity relation features.",
        "- `alignment_report.md`: source coverage, overlap, missing windows, and label balance.",
        "",
        "## Summary",
        "",
    ]
    for key, value in summary.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    (output_dir / "feature_card.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GDELT + ACLED mechanism features for seven-entity cross-strait analysis.")
    parser.add_argument("--gdelt-dir", default=str(DEFAULT_GDELT_DIR), help="Directory containing yearly GDELT parquet files.")
    parser.add_argument("--acled", default=str(DEFAULT_ACLED_PATH), help="ACLED semicolon-delimited CSV file.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for feature tables.")
    parser.add_argument("--granularity", choices=["day", "week", "month"], default="week", help="Temporal aggregation level.")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, help="Inclusive GDELT SQLDATE lower bound, YYYYMMDD.")
    parser.add_argument("--end-date", default=DEFAULT_END_DATE, help="Inclusive GDELT SQLDATE upper bound, YYYYMMDD.")
    parser.add_argument("--acled-filter-mode", choices=["country", "related"], default="country", help="ACLED filtering: country-only or country/actor/text related.")
    parser.add_argument("--skip-gdelt", action="store_true", help="Build only ACLED-side features.")
    parser.add_argument("--skip-acled", action="store_true", help="Build only GDELT-side features.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gdelt_frame = pd.DataFrame()
    node_frame = pd.DataFrame()
    edge_frame = pd.DataFrame()
    gdelt_summary = {"gdelt_row_groups": 0, "gdelt_input_rows": 0}
    if not args.skip_gdelt:
        gdelt_frame, node_frame, edge_frame, gdelt_summary = build_gdelt_features(
            Path(args.gdelt_dir),
            args.granularity,
            args.start_date,
            args.end_date,
        )

    acled_frame = pd.DataFrame()
    if not args.skip_acled:
        acled_frame = build_acled_features(Path(args.acled), args.granularity, args.acled_filter_mode)

    mechanism_frame = merge_features(gdelt_frame, acled_frame)
    mechanism_frame.to_csv(output_dir / "mechanism_features.csv", index=False, encoding="utf-8-sig")
    prediction_frame = build_prediction_feature_table(mechanism_frame)
    prediction_frame.to_csv(output_dir / "prediction_features_strict.csv", index=False, encoding="utf-8-sig")
    node_frame.to_csv(output_dir / "entity_node_features.csv", index=False, encoding="utf-8-sig")
    edge_frame.to_csv(output_dir / "entity_edge_features.csv", index=False, encoding="utf-8-sig")

    summary = {
        "mechanism_rows": len(mechanism_frame),
        "node_rows": len(node_frame),
        "edge_rows": len(edge_frame),
    }
    write_feature_card(output_dir, args, summary)
    write_alignment_report(output_dir, args, gdelt_summary, gdelt_frame, acled_frame, mechanism_frame, node_frame, edge_frame)
    print("Mechanism features generated.")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"output_dir: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
