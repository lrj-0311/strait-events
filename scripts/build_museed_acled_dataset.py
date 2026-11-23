from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


DEFAULT_SEVEN_COUNTRIES = ["China", "Taiwan", "Japan", "South Korea", "Philippines", "Indonesia", "United States"]


KEYWORD_FIELDS = (
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

ACTOR_KEYWORDS = (
    "china",
    "chinese",
    "taiwan",
    "taiwanese",
    "pla",
    "people's liberation army",
    "peoples liberation army",
    "prc",
    "roc",
    "taiwan strait",
    "kinmen",
    "quemoy",
    "matsu",
    "penghu",
)

STRAIT_KEYWORDS = (
    "taiwan strait",
    "kinmen",
    "quemoy",
    "matsu",
    "penghu",
    "pescadores",
    "fujian",
    "xiamen",
    "quanzhou",
    "fuzhou",
    "coast of fujian",
    "taipei",
    "kaohsiung",
    "taichung",
    "tainan",
)

COUNTRY_RELATED_KEYWORDS = {
    "China": ("china", "chinese", "prc", "beijing", "pla", "people's liberation army", "peoples liberation army"),
    "Taiwan": ("taiwan", "taiwanese", "taipei", "roc", "taiwan strait", "kinmen", "matsu", "penghu"),
    "Japan": ("japan", "japanese", "tokyo"),
    "South Korea": ("south korea", "korea", "korean", "seoul", "rok"),
    "Philippines": ("philippines", "philippine", "filipino", "manila"),
    "Indonesia": ("indonesia", "indonesian", "jakarta"),
    "United States": ("united states", "u.s.", "usa", "america", "american", "washington"),
}


@dataclass(frozen=True)
class RawFact:
    event_id: str
    event_date: date
    bucket_start: date
    head: str
    relation: str
    tail: str
    fact_type: str
    country: str
    admin1: str
    location: str


def clean(value: str | None) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def typed(kind: str, value: str) -> str:
    return f"{kind}::{clean(value)}"


def parse_event_date(value: str) -> date:
    return datetime.strptime(clean(value), "%Y-%m-%d").date()


def bucket_date(day: date, granularity: str) -> date:
    if granularity == "day":
        return day
    if granularity == "week":
        return day - timedelta(days=day.weekday())
    if granularity == "month":
        return date(day.year, day.month, 1)
    raise ValueError(f"Unsupported granularity: {granularity}")


def row_text(row: Dict[str, str], fields: Iterable[str] = KEYWORD_FIELDS) -> str:
    return " ".join(clean(row.get(field, "")) for field in fields).lower()


def has_any_keyword(row: Dict[str, str], keywords: Sequence[str]) -> bool:
    text = row_text(row)
    return any(keyword in text for keyword in keywords)


def include_row(row: Dict[str, str], scope: str) -> bool:
    country = clean(row.get("country"))
    is_core = country in {"China", "Taiwan"}

    if scope == "core":
        return is_core

    actor_keyword_hit = has_any_keyword(row, ACTOR_KEYWORDS)
    if scope == "actor-keyword":
        return is_core or actor_keyword_hit

    strait_hit = has_any_keyword(row, STRAIT_KEYWORDS)
    if scope == "strait-focus":
        return strait_hit or (is_core and actor_keyword_hit)

    raise ValueError(f"Unsupported scope: {scope}")


def normalize_countries(countries: Sequence[str] | None) -> set[str]:
    return {clean(country) for country in (countries or []) if clean(country)}


def country_related_hit(row: Dict[str, str], countries: set[str]) -> bool:
    if clean(row.get("country")) in countries:
        return True
    text = row_text(row)
    for country in countries:
        if any(keyword in text for keyword in COUNTRY_RELATED_KEYWORDS.get(country, ())):
            return True
    return False


def include_row_with_countries(row: Dict[str, str], scope: str, countries: set[str], country_filter_mode: str = "country") -> bool:
    if countries:
        if country_filter_mode == "country":
            return clean(row.get("country")) in countries
        if country_filter_mode == "related":
            return country_related_hit(row, countries)
        raise ValueError(f"Unsupported country filter mode: {country_filter_mode}")
    return include_row(row, scope)


def precision_ok(row: Dict[str, str], max_time_precision: int, max_geo_precision: int) -> bool:
    try:
        time_precision = int(clean(row.get("time_precision")) or "99")
    except ValueError:
        time_precision = 99

    try:
        geo_precision = int(clean(row.get("geo_precision")) or "99")
    except ValueError:
        geo_precision = 99

    return time_precision <= max_time_precision and geo_precision <= max_geo_precision


def relation_name(row: Dict[str, str]) -> str:
    event_type = clean(row.get("event_type"))
    sub_event_type = clean(row.get("sub_event_type")) or event_type
    if event_type and sub_event_type and event_type != sub_event_type:
        return f"{event_type}::{sub_event_type}"
    return sub_event_type


def row_to_facts(row: Dict[str, str], granularity: str) -> List[RawFact]:
    event_id = clean(row.get("event_id_cnty"))
    event_day = parse_event_date(clean(row.get("event_date")))
    bucket = bucket_date(event_day, granularity)

    actor1 = clean(row.get("actor1"))
    actor2 = clean(row.get("actor2"))
    country = clean(row.get("country"))
    admin1 = clean(row.get("admin1"))
    location = clean(row.get("location"))
    base_relation = relation_name(row)

    if not actor1 or not base_relation:
        return []

    head = typed("ACTOR", actor1)
    facts: List[RawFact] = []

    def add(relation: str, tail: str, fact_type: str) -> None:
        if not tail:
            return
        facts.append(
            RawFact(
                event_id=event_id,
                event_date=event_day,
                bucket_start=bucket,
                head=head,
                relation=relation,
                tail=tail,
                fact_type=fact_type,
                country=country,
                admin1=admin1,
                location=location,
            )
        )

    add(f"REL::{base_relation}", typed("ACTOR", actor2), "actor_actor")
    add(f"REL::{base_relation}@LOCATION", typed("LOCATION", location), "actor_location")
    add(f"REL::{base_relation}@ADMIN1", typed("ADMIN1", admin1), "actor_admin1")
    add(f"REL::{base_relation}@COUNTRY", typed("COUNTRY", country), "actor_country")

    return facts


def dedupe_facts(facts: Iterable[RawFact]) -> List[RawFact]:
    seen = set()
    output = []
    for fact in facts:
        key = (fact.head, fact.relation, fact.tail, fact.bucket_start)
        if key in seen:
            continue
        seen.add(key)
        output.append(fact)
    return output


def split_timestamps(timestamps: Sequence[date], split: Sequence[float]) -> Tuple[set[date], set[date], set[date]]:
    if len(timestamps) < 3:
        raise ValueError("Need at least 3 timestamps to create train/valid/test splits.")

    if not math.isclose(sum(split), 1.0, abs_tol=1e-6):
        raise ValueError("--split values must sum to 1.0")

    n_total = len(timestamps)
    n_train = max(1, int(n_total * split[0]))
    n_valid = max(1, int(n_total * split[1]))
    if n_train + n_valid >= n_total:
        n_train = max(1, n_total - 2)
        n_valid = 1

    train_times = set(timestamps[:n_train])
    valid_times = set(timestamps[n_train : n_train + n_valid])
    test_times = set(timestamps[n_train + n_valid :])
    return train_times, valid_times, test_times


def write_mapping(path: Path, id_to_name: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "name"])
        for idx, name in enumerate(id_to_name):
            writer.writerow([idx, name])


def write_quadruples(path: Path, facts: Sequence[RawFact], entity_to_id: Dict[str, int], relation_to_id: Dict[str, int], time_to_id: Dict[date, int]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        for fact in sorted(
            facts,
            key=lambda item: (
                time_to_id[item.bucket_start],
                relation_to_id[item.relation],
                entity_to_id[item.head],
                entity_to_id[item.tail],
            ),
        ):
            file.write(
                f"{entity_to_id[fact.head]} {relation_to_id[fact.relation]} "
                f"{entity_to_id[fact.tail]} {time_to_id[fact.bucket_start]}\n"
            )


def write_sources(path: Path, facts: Sequence[RawFact], entity_to_id: Dict[str, int], relation_to_id: Dict[str, int], time_to_id: Dict[date, int], split_by_time: Dict[date, str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "event_id",
                "event_date",
                "split",
                "timestamp_id",
                "timestamp_start",
                "head_id",
                "relation_id",
                "tail_id",
                "head",
                "relation",
                "tail",
                "fact_type",
                "country",
                "admin1",
                "location",
            ]
        )
        for fact in sorted(facts, key=lambda item: (time_to_id[item.bucket_start], item.event_id, item.fact_type)):
            writer.writerow(
                [
                    fact.event_id,
                    fact.event_date.isoformat(),
                    split_by_time[fact.bucket_start],
                    time_to_id[fact.bucket_start],
                    fact.bucket_start.isoformat(),
                    entity_to_id[fact.head],
                    relation_to_id[fact.relation],
                    entity_to_id[fact.tail],
                    fact.head,
                    fact.relation,
                    fact.tail,
                    fact.fact_type,
                    fact.country,
                    fact.admin1,
                    fact.location,
                ]
            )


def write_hyperedges(path: Path, rows: Sequence[Dict[str, str]], time_to_id: Dict[date, int], split_by_time: Dict[date, str]) -> None:
    role_fields = [
        ("ACTOR", "actor1", "actor1"),
        ("ACTOR", "actor2", "actor2"),
        ("ACTOR", "assoc_actor_1", "assoc_actor_1"),
        ("ACTOR", "assoc_actor_2", "assoc_actor_2"),
        ("INTER", "inter1", "inter1"),
        ("INTER", "inter2", "inter2"),
        ("INTERACTION", "interaction", "interaction"),
        ("DISORDER", "disorder_type", "disorder_type"),
        ("EVENT_TYPE", "event_type", "event_type"),
        ("SUB_EVENT_TYPE", "sub_event_type", "sub_event_type"),
        ("COUNTRY", "country", "country"),
        ("ADMIN1", "admin1", "admin1"),
        ("ADMIN2", "admin2", "admin2"),
        ("ADMIN3", "admin3", "admin3"),
        ("LOCATION", "location", "location"),
        ("SOURCE", "source_scale", "source_scale"),
    ]

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["event_id", "event_date", "split", "timestamp_id", "timestamp_start", "node_type", "node_name", "role"])
        for row in rows:
            bucket = parse_event_date(row["_bucket_start"])
            for node_type, field, role in role_fields:
                value = clean(row.get(field))
                if not value:
                    continue
                writer.writerow(
                    [
                        clean(row.get("event_id_cnty")),
                        clean(row.get("event_date")),
                        split_by_time[bucket],
                        time_to_id[bucket],
                        bucket.isoformat(),
                        node_type,
                        value,
                        role,
                    ]
                )


def write_time_features(path: Path, rows: Sequence[Dict[str, str]], time_to_id: Dict[date, int], split_by_time: Dict[date, str]) -> None:
    buckets: Dict[date, List[Dict[str, str]]] = {}
    for row in rows:
        bucket = parse_event_date(row["_bucket_start"])
        buckets.setdefault(bucket, []).append(row)

    tracked_event_types = [
        "Battles",
        "Explosions/Remote violence",
        "Political violence",
        "Protests",
        "Riots",
        "Strategic developments",
        "Violence against civilians",
    ]

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        header = [
            "timestamp_id",
            "timestamp_start",
            "split",
            "event_count",
            "fatalities_sum",
            "unique_actor1",
            "unique_actor2",
            "unique_locations",
            "unique_admin1",
        ] + [f"count::{name}" for name in tracked_event_types]
        writer.writerow(header)

        for bucket in sorted(buckets):
            bucket_rows = buckets[bucket]
            fatalities_sum = 0
            for row in bucket_rows:
                try:
                    fatalities_sum += int(float(clean(row.get("fatalities")) or "0"))
                except ValueError:
                    continue

            counts = []
            for event_type in tracked_event_types:
                counts.append(sum(1 for row in bucket_rows if clean(row.get("event_type")) == event_type or clean(row.get("disorder_type")) == event_type))

            writer.writerow(
                [
                    time_to_id[bucket],
                    bucket.isoformat(),
                    split_by_time[bucket],
                    len(bucket_rows),
                    fatalities_sum,
                    len({clean(row.get("actor1")) for row in bucket_rows if clean(row.get("actor1"))}),
                    len({clean(row.get("actor2")) for row in bucket_rows if clean(row.get("actor2"))}),
                    len({clean(row.get("location")) for row in bucket_rows if clean(row.get("location"))}),
                    len({clean(row.get("admin1")) for row in bucket_rows if clean(row.get("admin1"))}),
                ]
                + counts
            )


def write_dataset_card(path: Path, args: argparse.Namespace, stats: Dict[str, int | str]) -> None:
    lines = [
        f"# {args.dataset}",
        "",
        "Generated from ACLED for MUSE-ED multi-factor event forecasting experiments.",
        "",
        "## Build options",
        "",
        f"- input: `{Path(args.input).resolve()}`",
        f"- countries: `{', '.join(sorted(normalize_countries(args.countries))) if args.countries else 'scope-based'}`",
        f"- country filter mode: `{args.country_filter_mode}`",
        f"- scope: `{args.scope}`",
        f"- time granularity: `{args.time_granularity}`",
        f"- split: `{args.split[0]} {args.split[1]} {args.split[2]}`",
        f"- max time precision: `{args.max_time_precision}`",
        f"- max geo precision: `{args.max_geo_precision}`",
        f"- keep duplicates: `{args.keep_duplicates}`",
        "",
        "## Statistics",
        "",
    ]
    for key, value in stats.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_dataset(args: argparse.Namespace) -> Dict[str, int | str]:
    input_path = Path(args.input)
    output_dir = Path(args.output_root) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    countries = normalize_countries(args.countries)

    facts: List[RawFact] = []
    kept_event_rows: List[Dict[str, str]] = []
    rows_seen = 0
    rows_kept = 0

    with input_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter=";")
        for row in reader:
            rows_seen += 1
            if not include_row_with_countries(row, args.scope, countries, args.country_filter_mode):
                continue
            if not precision_ok(row, args.max_time_precision, args.max_geo_precision):
                continue
            rows_kept += 1
            try:
                generated_facts = row_to_facts(row, args.time_granularity)
            except ValueError:
                continue
            if not generated_facts:
                continue
            facts.extend(generated_facts)
            row_copy = dict(row)
            row_copy["_bucket_start"] = generated_facts[0].bucket_start.isoformat()
            kept_event_rows.append(row_copy)

    if not args.keep_duplicates:
        facts = dedupe_facts(facts)

    if not facts:
        raise ValueError("No facts were generated. Check scope and precision filters.")

    timestamps = sorted({fact.bucket_start for fact in facts})
    train_times, valid_times, test_times = split_timestamps(timestamps, args.split)

    split_by_time: Dict[date, str] = {}
    split_by_time.update({time: "train" for time in train_times})
    split_by_time.update({time: "valid" for time in valid_times})
    split_by_time.update({time: "test" for time in test_times})

    train_facts = [fact for fact in facts if fact.bucket_start in train_times]
    valid_facts = [fact for fact in facts if fact.bucket_start in valid_times]
    test_facts = [fact for fact in facts if fact.bucket_start in test_times]

    entities = sorted({fact.head for fact in facts} | {fact.tail for fact in facts})
    relations = sorted({fact.relation for fact in facts})
    entity_to_id = {name: idx for idx, name in enumerate(entities)}
    relation_to_id = {name: idx for idx, name in enumerate(relations)}
    time_to_id = {time: idx for idx, time in enumerate(timestamps)}

    write_quadruples(output_dir / "train.txt", train_facts, entity_to_id, relation_to_id, time_to_id)
    write_quadruples(output_dir / "valid.txt", valid_facts, entity_to_id, relation_to_id, time_to_id)
    write_quadruples(output_dir / "test.txt", test_facts, entity_to_id, relation_to_id, time_to_id)

    (output_dir / "stat.txt").write_text(f"{len(entities)} {len(relations)}\n", encoding="utf-8")
    write_mapping(output_dir / "entity2id.csv", entities)
    write_mapping(output_dir / "relation2id.csv", relations)
    write_sources(output_dir / "event_id_to_quadruples.csv", facts, entity_to_id, relation_to_id, time_to_id, split_by_time)
    write_hyperedges(output_dir / "event_hyperedges.csv", kept_event_rows, time_to_id, split_by_time)
    write_time_features(output_dir / "time_slice_features.csv", kept_event_rows, time_to_id, split_by_time)

    stats: Dict[str, int | str] = {
        "country_filter": ", ".join(sorted(countries)) if countries else "scope_based",
        "country_filter_mode": args.country_filter_mode,
        "rows_seen": rows_seen,
        "rows_kept": rows_kept,
        "events_with_facts": len(kept_event_rows),
        "facts": len(facts),
        "entities": len(entities),
        "relations": len(relations),
        "timestamps": len(timestamps),
        "train_facts": len(train_facts),
        "valid_facts": len(valid_facts),
        "test_facts": len(test_facts),
        "first_timestamp": timestamps[0].isoformat(),
        "last_timestamp": timestamps[-1].isoformat(),
    }
    write_dataset_card(output_dir / "dataset_card.md", args, stats)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a MUSE-ED multi-factor temporal event dataset from ACLED CSV.")
    parser.add_argument("--input", default="data/raw/acled.csv", help="Path to the ACLED semicolon-delimited CSV file.")
    parser.add_argument("--dataset", default="ACLED_MUSE_7COUNTRY", help="Output dataset name under --output-root.")
    parser.add_argument("--output-root", default="data", help="Output root that contains MUSE-ED dataset folders.")
    parser.add_argument("--countries", nargs="*", default=DEFAULT_SEVEN_COUNTRIES, help="ACLED country names. Defaults to the seven-entity regional network.")
    parser.add_argument("--country-filter-mode", choices=["country", "related"], default="country", help="When --countries is used, filter by country only or by country/actor/text relevance.")
    parser.add_argument("--scope", choices=["core", "actor-keyword", "strait-focus"], default="core", help="ACLED row filtering scope.")
    parser.add_argument("--time-granularity", choices=["day", "week", "month"], default="week", help="Timestamp bucket size.")
    parser.add_argument("--split", nargs=3, type=float, default=[0.8, 0.1, 0.1], metavar=("TRAIN", "VALID", "TEST"), help="Chronological split ratios.")
    parser.add_argument("--max-time-precision", type=int, default=3, help="Keep rows with ACLED time_precision <= this value.")
    parser.add_argument("--max-geo-precision", type=int, default=3, help="Keep rows with ACLED geo_precision <= this value.")
    parser.add_argument("--keep-duplicates", action="store_true", help="Keep duplicate quadruples within the same timestamp.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = build_dataset(args)
    print("ACLED TKG dataset generated successfully.")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
