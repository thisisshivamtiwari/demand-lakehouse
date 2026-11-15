"""Gold: pickups per zone per day, zero-filled, with the zone's borough.

Zero-filling matters: a zone with no pickups on a day has demand 0, not a missing row, and lag features count rows.
"""

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


def zones(spark: SparkSession, lookup_csv: Path) -> DataFrame:
    z = spark.read.option("header", True).csv(str(lookup_csv))
    return z.select(F.col("LocationID").cast("int").alias("zone"), F.col("Borough").alias("borough")).where(
        F.col("zone").between(1, 263)
    )


def build(silver: DataFrame, zone_df: DataFrame) -> DataFrame:
    counts = silver.groupBy(F.col("pu_zone").alias("zone"), F.col("pickup_date").alias("date")).agg(
        F.count(F.lit(1)).alias("trips")
    )
    bounds = silver.agg(F.min("pickup_date").alias("lo"), F.max("pickup_date").alias("hi"))
    dates = bounds.select(F.explode(F.sequence("lo", "hi")).alias("date"))
    grid = zone_df.crossJoin(dates)
    return grid.join(counts, ["zone", "date"], "left").fillna(0, ["trips"]).select("zone", "borough", "date", "trips")


def run(spark: SparkSession, silver: Path, lookup_csv: Path, gold: Path, snapshot: Path | None = None) -> dict:
    df = build(spark.read.format("delta").load(str(silver)), zones(spark, lookup_csv))
    df.write.format("delta").mode("overwrite").save(str(gold))
    out = spark.read.format("delta").load(str(gold)).orderBy("zone", "date")
    if snapshot is not None:
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        out.toPandas().to_parquet(snapshot, index=False)
    stats = out.agg(
        F.countDistinct("zone").alias("zones"), F.countDistinct("date").alias("days"), F.sum("trips").alias("trips")
    ).first()
    return stats.asDict()
