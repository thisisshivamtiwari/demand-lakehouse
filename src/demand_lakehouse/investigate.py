"""Why the silver gate stopped Oct-Nov 2025: are negative-fare rows reversals of real trips, or trips of their own?

A reversal has a positive twin in the same file: same pickup and dropoff times and zones, opposite fare."""

from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

KEY = ["pickup_ts", "dropoff_ts", "pu_zone", "do_zone"]


def negative_fares(spark: SparkSession, bronze: Path, months: list[str]) -> dict:
    df = spark.read.format("delta").load(str(bronze)).where(F.col("src_month").isin(months))
    neg = df.where(F.col("fare_amount") < 0).select("src_month", *KEY, "payment_type", "fare_amount", "trip_distance")
    pos = df.where(F.col("fare_amount") > 0).select("src_month", *KEY, F.col("fare_amount").alias("twin_fare"))
    pos = pos.dropDuplicates(["src_month", *KEY])
    joined = neg.join(pos, ["src_month", *KEY], "left")
    rows = (
        joined.groupBy("src_month", "payment_type")
        .agg(
            F.count(F.lit(1)).alias("negative_rows"),
            F.sum((F.col("twin_fare") == -F.col("fare_amount")).cast("int")).alias("exact_positive_twin"),
            F.percentile_approx("trip_distance", 0.5).alias("median_miles"),
        )
        .orderBy("src_month", "payment_type")
        .collect()
    )
    totals = {r["src_month"]: r["count"] for r in df.groupBy("src_month").count().collect()}
    return {"months": months, "rows_in_month": totals, "by_payment_type": [r.asDict() for r in rows]}
