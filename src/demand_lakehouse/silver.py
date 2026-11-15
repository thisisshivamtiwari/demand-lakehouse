"""Silver: data-quality expectations on bronze trips.

Each rule says what a good row looks like. A row failing any drop rule goes to the quarantine table with the first
rule it failed; the rest become silver. A drop rule whose failure rate passes its threshold in any month fails the
whole run, so a broken month never reaches the forecasts. Warn rules (threshold None) only count rows. This is the
same idea as Delta Live Tables' expect (warn), expect_or_drop and expect_or_fail.
"""

from pathlib import Path

from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F


def default_rules() -> dict[str, tuple[Column, float | None]]:
    """name: (good-row condition, maximum share of a month's rows allowed to fail before the run stops; None = warn only).

    Built in a function because Spark columns need a running session."""
    duration = F.unix_timestamp("dropoff_ts") - F.unix_timestamp("pickup_ts")
    return {
        "pickup_in_file_month": (F.date_format("pickup_ts", "yyyy-MM") == F.col("src_month"), 0.005),
        "dropoff_after_pickup": (F.col("dropoff_ts") > F.col("pickup_ts"), 0.02),
        "duration_1min_to_4h": (duration.between(60, 4 * 3600), 0.05),
        "distance_0_to_100mi": ((F.col("trip_distance") > 0) & (F.col("trip_distance") <= 100), 0.05),
        # A negative row paid by card, cash, no-charge or dispute is a reversal: it mirrors a real trip, so drop it.
        # Flex-fare rows (payment_type 0) with a negative fare have no mirror; they are trips with a bad amount
        # (found when this rule, first written as "amounts >= 0", stopped the Oct-Nov 2025 months). Keep them, count them.
        "not_a_reversal": (((F.col("fare_amount") >= 0) & (F.col("total_amount") >= 0)) | (F.col("payment_type") == 0), 0.05),
        "flex_fare_amount_non_negative": (~((F.col("payment_type") == 0) & (F.col("fare_amount") < 0)), None),
        "known_pickup_zone": (F.col("pu_zone").between(1, 263), 0.05),
    }


class ExpectationFailed(RuntimeError):
    def __init__(self, message: str, report: dict):
        super().__init__(message)
        self.report = report


def failed_rule(rules: dict) -> Column:
    """Name of the first rule a row fails, or null. A null in a condition counts as a failure."""
    expr = F.lit(None).cast("string")
    for name, (good, limit) in reversed(list(rules.items())):
        if limit is None:
            continue
        expr = F.when(~F.coalesce(good, F.lit(False)), F.lit(name)).otherwise(expr)
    return expr


def check(df: DataFrame, rules: dict) -> dict:
    """Failure count and rate per rule (rules counted independently), overall and in the worst month.

    The threshold applies per month, so one broken file cannot hide inside a year of good ones."""
    aggs = [F.count(F.lit(1)).alias("rows")]
    aggs += [F.sum(F.when(~F.coalesce(good, F.lit(False)), 1).otherwise(0)).alias(name) for name, (good, _) in rules.items()]
    months = df.groupBy("src_month").agg(*aggs).collect()
    total = sum(m["rows"] for m in months)
    out = {"rows": total, "rules": {}, "breaches": []}
    for name, (_, limit) in rules.items():
        failed = sum(m[name] for m in months)
        worst = max(months, key=lambda m: m[name] / m["rows"])
        worst_rate = worst[name] / worst["rows"]
        out["rules"][name] = {
            "failed": failed,
            "rate": round(failed / total, 6) if total else 0.0,
            "worst_month": worst["src_month"],
            "worst_month_rate": round(worst_rate, 6),
            "max_rate": limit,
        }
        if limit is not None and worst_rate > limit:
            out["breaches"].append(f"{name} ({worst['src_month']})")
    return out


def run(spark: SparkSession, bronze: Path, silver: Path, quarantine: Path, rules: dict | None = None) -> dict:
    rules = rules or default_rules()
    df = spark.read.format("delta").load(str(bronze))
    report = check(df, rules)
    if report["breaches"]:
        raise ExpectationFailed(f"data-quality thresholds broken: {report['breaches']}", report)
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
