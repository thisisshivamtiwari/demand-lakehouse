"""Bronze -> silver -> gold on a hand-made fixture with the problems seen in the real files."""

import datetime as dt

import pandas as pd
import pytest

from demand_lakehouse import bronze, gold, silver


def raw_month(month: str, drift: bool) -> pd.DataFrame:
    y, m = map(int, month.split("-"))
    t = dt.datetime(y, m, 2, 8, 0)
    rows = [
        # good trips: zone 1 twice on day 2, zone 2 once on day 4
        dict(pu=1, start=t, mins=12, dist=2.0, fare=10.0, total=14.0),
        dict(pu=1, start=t + dt.timedelta(hours=3), mins=20, dist=4.5, fare=18.0, total=22.0),
        dict(pu=2, start=t + dt.timedelta(days=2), mins=8, dist=1.1, fare=7.0, total=9.0),
        # bad trips, one per rule
        dict(pu=1, start=dt.datetime(2009, 1, 1, 0, 0), mins=10, dist=1.0, fare=5.0, total=6.0),  # outside the file's month
        dict(pu=1, start=t, mins=-5, dist=1.0, fare=5.0, total=6.0),  # dropoff before pickup
        dict(pu=2, start=t, mins=0.5, dist=0.2, fare=3.0, total=4.0),  # 30-second trip
        dict(pu=2, start=t, mins=15, dist=0.0, fare=8.0, total=9.0),  # zero distance
        dict(pu=1, start=t, mins=15, dist=3.0, fare=-12.0, total=-15.0),  # reversal
        dict(pu=264, start=t, mins=15, dist=3.0, fare=12.0, total=15.0),  # unknown zone
        # flex-fare trip with a negative fare: a real pickup, kept and counted by a warn rule
        dict(pu=2, start=t + dt.timedelta(days=2), mins=9, dist=1.4, fare=-4.0, total=-3.0, pt=0),
    ]
    df = pd.DataFrame(
        {
            "VendorID": 2,
            "tpep_pickup_datetime": [r["start"] for r in rows],
            "tpep_dropoff_datetime": [r["start"] + dt.timedelta(minutes=r["mins"]) for r in rows],
            "passenger_count": 1.0 if drift else 1,
            "trip_distance": [r["dist"] for r in rows],
            "PULocationID": [r["pu"] for r in rows],
            "DOLocationID": 3,
            "payment_type": [r.get("pt", 1) for r in rows],
            "fare_amount": [r["fare"] for r in rows],
            "total_amount": [r["total"] for r in rows],
        }
    )
    if drift:  # later files renamed a column and added a fee
        df = df.rename(columns={"VendorID": "vendorid"})
        df["cbd_congestion_fee"] = 0.75
    return df


@pytest.fixture(scope="module")
def lake(spark, tmp_path_factory):
    root = tmp_path_factory.mktemp("lake")
    raw = root / "raw"
    raw.mkdir()
    months = ["2026-01", "2026-02"]
    for i, m in enumerate(months):
        raw_month(m, drift=i == 1).to_parquet(raw / f"yellow_tripdata_{m}.parquet", coerce_timestamps="us")
    pd.DataFrame({"LocationID": [1, 2, 264], "Borough": ["Manhattan", "Queens", "Unknown"], "Zone": ["a", "b", "c"]}).to_csv(
        raw / "zones.csv", index=False
    )
    return root, raw, months


def test_bronze_records_drift_and_is_idempotent(spark, lake):
    root, raw, months = lake
    first = bronze.run(spark, raw, months, root / "bronze")
    again = bronze.run(spark, raw, months, root / "bronze")  # rerun must replace, not append
    assert first["rows_by_month"] == again["rows_by_month"] == {"2026-01": 10, "2026-02": 10}
    drift = {(d["column"], d["change"]) for d in first["months"][1]["drift"]}
    assert ("passenger_count", "cast") in drift
    assert first["months"][1]["unused_columns"] == ["cbd_congestion_fee"]
    assert ("VendorID", "renamed") in drift  # found case-insensitively, recorded as drift


def test_silver_quarantines_one_row_per_rule(spark, lake):
    root, raw, months = lake
    lenient = {k: (cond, None if limit is None else 1.0) for k, (cond, limit) in silver.default_rules().items()}
    rep = silver.run(spark, root / "bronze", root / "silver", root / "quarantine", rules=lenient)
    assert rep["silver_rows"] == 8 and rep["quarantined_rows"] == 12
    assert rep["rules"]["flex_fare_amount_non_negative"]["failed"] == 2 and not rep["breaches"]
    q = spark.read.format("delta").load(str(root / "quarantine")).groupBy("failed_rule").count().collect()
    assert {r["failed_rule"]: r["count"] for r in q} == {k: 2 for k, (_, limit) in lenient.items() if limit is not None}


def test_silver_stops_the_run_above_threshold(spark, lake):
    root, *_ = lake
    with pytest.raises(silver.ExpectationFailed, match="known_pickup_zone"):
        silver.run(spark, root / "bronze", root / "s2", root / "q2")


def test_gold_zero_fills_every_zone_day(spark, lake):
    root, raw, _ = lake
    stats = gold.run(spark, root / "silver", raw / "zones.csv", root / "gold")
    df = spark.read.format("delta").load(str(root / "gold")).toPandas()
    assert stats["zones"] == 2 and stats["trips"] == 8
    assert len(df) == 2 * stats["days"]  # every zone on every day, zeros included
    jan2 = df[(df["date"].astype(str) == "2026-01-02")].set_index("zone")["trips"]
    assert jan2.to_dict() == {1: 2, 2: 0}
