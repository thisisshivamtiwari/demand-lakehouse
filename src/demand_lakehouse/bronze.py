"""Bronze: raw monthly files into one Delta table with a canonical schema.

TLC files drift between months (column names change case, integer columns arrive as doubles, new fee columns
appear). Each file is read on its own, mapped to the canonical schema, and the casts are recorded as drift.
A month is written with replaceWhere, so re-running a month replaces it instead of duplicating it.
"""

from pathlib import Path

from delta.tables import DeltaTable
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

# canonical column: (source column, Spark type)
CANONICAL = {
    "vendor_id": ("VendorID", "int"),
    "pickup_ts": ("tpep_pickup_datetime", "timestamp"),
    "dropoff_ts": ("tpep_dropoff_datetime", "timestamp"),
    "passenger_count": ("passenger_count", "int"),
    "trip_distance": ("trip_distance", "double"),
    "pu_zone": ("PULocationID", "int"),
    "do_zone": ("DOLocationID", "int"),
    "payment_type": ("payment_type", "int"),
    "fare_amount": ("fare_amount", "double"),
    "total_amount": ("total_amount", "double"),
}


def to_canonical(df: DataFrame, source: str) -> tuple[DataFrame, list[dict]]:
    """Map one raw file to the canonical schema; return the frame and the drift found (casts, missing columns)."""
    types = {f.name.lower(): (f.name, f.dataType.simpleString()) for f in df.schema.fields}
    cols, drift = [], []
    for name, (src, typ) in CANONICAL.items():
        found = types.get(src.lower())
        if found is None:
            cols.append(F.lit(None).cast(typ).alias(name))
            drift.append({"file": source, "column": src, "change": "missing", "to": typ})
            continue
        actual, actual_type = found
        if actual != src:
            drift.append({"file": source, "column": src, "change": "renamed", "from": actual})
        if actual_type.replace("timestamp_ntz", "timestamp") != typ:
            drift.append({"file": source, "column": actual, "change": "cast", "from": actual_type, "to": typ})
        cols.append(F.col(actual).cast(typ).alias(name))
    return df.select(*cols), drift


def ingest_month(spark: SparkSession, path: Path, month: str, table: Path) -> dict:
    raw = spark.read.parquet(str(path))
    df, drift = to_canonical(raw, path.name)
    df = df.withColumn("src_month", F.lit(month)).withColumn("_source_file", F.lit(path.name))
    df = df.withColumn("_ingested_at", F.current_timestamp())
    writer = df.write.format("delta").mode("overwrite")
    if DeltaTable.isDeltaTable(spark, str(table)):
        writer = writer.option("replaceWhere", f"src_month = '{month}'")
    else:
        writer = writer.partitionBy("src_month")
    writer.save(str(table))
    extra = sorted(c for c in raw.columns if c.lower() not in {src.lower() for src, _ in CANONICAL.values()})
    return {"month": month, "file": path.name, "drift": drift, "unused_columns": extra}


def run(spark: SparkSession, raw_dir: Path, months: list[str], table: Path) -> dict:
    report = [ingest_month(spark, raw_dir / f"yellow_tripdata_{m}.parquet", m, table) for m in months]
    rows = spark.read.format("delta").load(str(table)).groupBy("src_month").count().orderBy("src_month").collect()
    return {"months": report, "rows_by_month": {r["src_month"]: r["count"] for r in rows}}
