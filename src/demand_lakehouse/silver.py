"""Silver: data-quality expectations on bronze trips.

Each rule says what a good row looks like. A row failing any rule goes to the quarantine table with the first rule it
failed; the rest become silver. A rule whose failure rate passes its threshold fails the whole run, so a broken
month never reaches the forecasts (the same idea as Delta Live Tables' expect_or_drop and expect_or_fail).
"""

from pathlib import Path

from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F


def default_rules() -> dict[str, tuple[Column, float]]:
    """name: (good-row condition, maximum share of rows allowed to fail before the run stops).

    Built in a function because Spark columns need a running session."""
    duration = F.unix_timestamp("dropoff_ts") - F.unix_timestamp("pickup_ts")
    return {
        "pickup_in_file_month": (F.date_format("pickup_ts", "yyyy-MM") == F.col("src_month"), 0.005),
        "dropoff_after_pickup": (F.col("dropoff_ts") > F.col("pickup_ts"), 0.02),
        "duration_1min_to_4h": (duration.between(60, 4 * 3600), 0.05),
        "distance_0_to_100mi": ((F.col("trip_distance") > 0) & (F.col("trip_distance") <= 100), 0.05),
        "amounts_non_negative": ((F.col("fare_amount") >= 0) & (F.col("total_amount") >= 0), 0.05),
        "known_pickup_zone": (F.col("pu_zone").between(1, 263), 0.05),
    }


class ExpectationFailed(RuntimeError):
    pass


def failed_rule(rules: dict) -> Column:
    """Name of the first rule a row fails, or null. A null in a condition counts as a failure."""
    expr = F.lit(None).cast("string")
    for name, (good, _) in reversed(list(rules.items())):
        expr = F.when(~F.coalesce(good, F.lit(False)), F.lit(name)).otherwise(expr)
    return expr


def check(df: DataFrame, rules: dict) -> dict:
    """Failure count and rate per rule (rules are counted independently), and which rules break their threshold."""
    aggs = [F.count(F.lit(1)).alias("rows")]
    aggs += [F.sum(F.when(~F.coalesce(good, F.lit(False)), 1).otherwise(0)).alias(name) for name, (good, _) in rules.items()]
    row = df.agg(*aggs).first()
    total = row["rows"]
    out = {"rows": total, "rules": {}, "breaches": []}
    for name, (_, limit) in rules.items():
        rate = row[name] / total if total else 0.0
        out["rules"][name] = {"failed": row[name], "rate": round(rate, 6), "max_rate": limit}
        if rate > limit:
            out["breaches"].append(name)
    return out


def run(spark: SparkSession, bronze: Path, silver: Path, quarantine: Path, rules: dict | None = None) -> dict:
    rules = rules or default_rules()
    df = spark.read.format("delta").load(str(bronze))
    report = check(df, rules)
    if report["breaches"]:
        raise ExpectationFailed(f"data-quality thresholds broken: {report['breaches']}")
    flagged = df.withColumn("failed_rule", failed_rule(rules))
    (
        flagged.where(F.col("failed_rule").isNull())
        .select("pickup_ts", "dropoff_ts", "pu_zone", "do_zone", "trip_distance", "total_amount", "src_month")
        .withColumn("pickup_date", F.to_date("pickup_ts"))
        .write.format("delta")
        .mode("overwrite")
        .partitionBy("src_month")
        .save(str(silver))
    )
    flagged.where(F.col("failed_rule").isNotNull()).write.format("delta").mode("overwrite").save(str(quarantine))
    kept = spark.read.format("delta").load(str(silver)).count()
    report["silver_rows"] = kept
    report["quarantined_rows"] = report["rows"] - kept
    return report
