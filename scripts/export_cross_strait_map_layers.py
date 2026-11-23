from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, Tuple

from build_museed_acled_dataset import bucket_date, clean, include_row_with_countries, normalize_countries, parse_event_date, relation_name


DEFAULT_SEVEN_COUNTRIES = ["China", "Taiwan", "Japan", "South Korea", "Philippines", "Indonesia", "United States"]


HIGH_RISK_EVENT_TYPES = {
    "Battles",
    "Explosions/Remote violence",
    "Political violence",
    "Riots",
    "Strategic developments",
    "Violence against civilians",
}


def safe_float(value: str | None) -> float | None:
    text = clean(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def safe_int(value: str | None) -> int:
    text = clean(value)
    if not text:
        return 0
    try:
        return int(float(text))
    except ValueError:
        return 0


def risk_score(row: Dict[str, str]) -> float:
    event_type = clean(row.get("event_type"))
    fatalities = safe_int(row.get("fatalities"))
    civilian_targeting = 1 if clean(row.get("civilian_targeting")) else 0
    high_risk = 1 if event_type in HIGH_RISK_EVENT_TYPES else 0
    return 1.0 + high_risk + civilian_targeting + min(fatalities, 10) * 0.25


def iter_filtered_rows(input_path: Path, scope: str, granularity: str, countries: set[str], country_filter_mode: str = "country") -> Iterable[Tuple[Dict[str, str], date, date]]:
    with input_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter=";")
        for row in reader:
            if not include_row_with_countries(row, scope, countries, country_filter_mode):
                continue
            lat = safe_float(row.get("latitude"))
            lon = safe_float(row.get("longitude"))
            if lat is None or lon is None:
                continue
            try:
                event_day = parse_event_date(clean(row.get("event_date")))
            except ValueError:
                continue
            yield row, event_day, bucket_date(event_day, granularity)


def write_points(rows: Iterable[Tuple[Dict[str, str], date, date]], path: Path) -> Tuple[int, Dict[Tuple[str, str], Dict[str, float]]]:
    weekly_location: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(lambda: {"event_count": 0, "risk_sum": 0, "fatalities_sum": 0})
    count = 0
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "event_id",
                "event_date",
                "year",
                "timestamp_start",
                "country",
                "admin1",
                "admin2",
                "location",
                "latitude",
                "longitude",
                "event_type",
                "sub_event_type",
                "relation",
                "actor1",
                "actor2",
                "fatalities",
                "civilian_targeting",
                "risk_score",
                "source_scale",
                "notes",
            ]
        )
        for row, event_day, bucket in rows:
            lat = safe_float(row.get("latitude"))
            lon = safe_float(row.get("longitude"))
            if lat is None or lon is None:
                continue
            score = risk_score(row)
            fatalities = safe_int(row.get("fatalities"))
            writer.writerow(
                [
                    clean(row.get("event_id_cnty")),
                    event_day.isoformat(),
                    event_day.year,
                    bucket.isoformat(),
                    clean(row.get("country")),
                    clean(row.get("admin1")),
                    clean(row.get("admin2")),
                    clean(row.get("location")),
                    lat,
                    lon,
                    clean(row.get("event_type")),
                    clean(row.get("sub_event_type")),
                    relation_name(row),
                    clean(row.get("actor1")),
                    clean(row.get("actor2")),
                    fatalities,
                    clean(row.get("civilian_targeting")),
                    round(score, 4),
                    clean(row.get("source_scale")),
                    clean(row.get("notes")),
                ]
            )
            key = (bucket.isoformat(), clean(row.get("location")) or clean(row.get("admin1")))
            weekly_location[key]["event_count"] += 1
            weekly_location[key]["risk_sum"] += score
            weekly_location[key]["fatalities_sum"] += fatalities
            count += 1
    return count, weekly_location


def write_weekly_location_risk(weekly_location: Dict[Tuple[str, str], Dict[str, float]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp_start", "location_key", "event_count", "risk_sum", "fatalities_sum"])
        for (timestamp_start, location_key), values in sorted(weekly_location.items()):
            writer.writerow(
                [
                    timestamp_start,
                    location_key,
                    int(values["event_count"]),
                    round(values["risk_sum"], 4),
                    int(values["fatalities_sum"]),
                ]
            )


def write_admin_yearly(rows: Iterable[Tuple[Dict[str, str], date, date]], path: Path) -> None:
    admin_yearly: Dict[Tuple[int, str, str], Dict[str, float]] = defaultdict(lambda: {"event_count": 0, "risk_sum": 0, "fatalities_sum": 0})
    for row, event_day, _ in rows:
        key = (event_day.year, clean(row.get("country")), clean(row.get("admin1")) or "UNKNOWN")
        admin_yearly[key]["event_count"] += 1
        admin_yearly[key]["risk_sum"] += risk_score(row)
        admin_yearly[key]["fatalities_sum"] += safe_int(row.get("fatalities"))

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["year", "country", "admin1", "event_count", "risk_sum", "fatalities_sum"])
        for (year, country, admin1), values in sorted(admin_yearly.items()):
            writer.writerow([year, country, admin1, int(values["event_count"]), round(values["risk_sum"], 4), int(values["fatalities_sum"])])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export normalized ACLED event locations and weekly records.")
    parser.add_argument("--input", default="data/raw/acled.csv", help="Path to the ACLED semicolon-delimited CSV file.")
    parser.add_argument("--output-dir", default="data/ACLED_MUSE_7COUNTRY/map_layers", help="Directory for event CSV layers.")
    parser.add_argument("--countries", nargs="*", default=DEFAULT_SEVEN_COUNTRIES, help="ACLED country names. Defaults to the seven-entity regional network.")
    parser.add_argument("--country-filter-mode", choices=["country", "related"], default="country", help="When --countries is used, filter by country only or by country/actor/text relevance.")
    parser.add_argument("--scope", choices=["core", "actor-keyword", "strait-focus"], default="core", help="ACLED row filtering scope.")
    parser.add_argument("--time-granularity", choices=["day", "week", "month"], default="week", help="Timestamp bucket size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    countries = normalize_countries(args.countries)

    rows = list(iter_filtered_rows(input_path, args.scope, args.time_granularity, countries, args.country_filter_mode))
    count, weekly_location = write_points(rows, output_dir / "events_map_points.csv")
    write_weekly_location_risk(weekly_location, output_dir / "weekly_location_risk.csv")
    write_admin_yearly(rows, output_dir / "admin1_yearly_risk.csv")

    print("Map layers exported.")
    print(f"country_filter: {', '.join(sorted(countries)) if countries else 'scope_based'}")
    print(f"country_filter_mode: {args.country_filter_mode}")
    print(f"events: {count}")
    print(f"output_dir: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
