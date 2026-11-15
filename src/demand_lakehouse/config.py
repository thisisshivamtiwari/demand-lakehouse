from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
LAKE = ROOT / "data" / "lake"
GOLD_SNAPSHOT = ROOT / "data" / "gold" / "zone_day.parquet"
RESULTS = ROOT / "results"

BRONZE = LAKE / "bronze" / "trips"
SILVER = LAKE / "silver" / "trips"
QUARANTINE = LAKE / "silver" / "quarantine"
GOLD = LAKE / "gold" / "zone_day"

# 12 months of NYC TLC yellow-taxi records (public): Oct 2025 to Sep 2026
MONTHS = [f"2025-{m:02d}" for m in (10, 11, 12)] + [f"2026-{m:02d}" for m in range(1, 10)]
TLC_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{month}.parquet"
ZONES_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
