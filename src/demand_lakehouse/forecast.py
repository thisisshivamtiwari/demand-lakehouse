"""Forecasting and the rolling-origin backtest, all on Spark.

A row is (series, origin, horizon): standing at the end of day `origin`, forecast demand on day origin + h.
Features use only days up to the origin, so nothing peeks at the future. Methods:
- seasonal_naive: the last observed value on the same weekday;
- seasonal_mean4: the mean of the last 4 same-weekday values;
- gbt_global: one Spark MLlib gradient-boosted tree model for every series at a level (log scale);
- ridge_per_series: one small ridge regression per series, trained in parallel with applyInPandas ("many models").
Zone forecasts summed up the hierarchy (bottom-up) are compared with direct borough and city forecasts.
"""

import datetime as dt

import numpy as np
import pandas as pd
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

HORIZON = 14
N_ORIGINS = 12
# US federal holidays and the days around Thanksgiving and Christmas, Oct 2025 to Sep 2026
HOLIDAYS = [
    "2025-10-13", "2025-11-11", "2025-11-27", "2025-11-28", "2025-12-24", "2025-12-25", "2025-12-31",
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-05-25", "2026-06-19", "2026-07-03", "2026-07-04", "2026-09-07",
]  # fmt: skip
LAGS = [7, 14, 21, 28, 35]
FEATURES = ["dow", "h", "holiday", "sw1", "sw2", "sw3", "sw4", "last", "m7", "m28"]
GBT_PARAMS = {"maxIter": 60, "maxDepth": 5, "stepSize": 0.1, "seed": 7}


def series_frames(gold: DataFrame) -> dict[str, DataFrame]:
    """(series, date, y) at each level of the hierarchy, plus the zone -> borough map for bottom-up sums."""
    zone = gold.select(F.col("zone").cast("string").alias("series"), "date", F.col("trips").cast("double").alias("y"))
    borough = gold.groupBy(F.col("borough").alias("series"), "date").agg(F.sum("trips").cast("double").alias("y"))
    city = gold.groupBy("date").agg(F.sum("trips").cast("double").alias("y")).withColumn("series", F.lit("NYC"))
    return {"zone": zone, "borough": borough, "city": city.select("series", "date", "y")}


def origins(last_day: dt.date, n: int = N_ORIGINS, horizon: int = HORIZON) -> list[dt.date]:
    """Weekly backtest origins; the latest one leaves exactly `horizon` days of actuals after it."""
    final = last_day - dt.timedelta(days=horizon)
    return [final - dt.timedelta(days=7 * k) for k in reversed(range(n))]


def features(ys: DataFrame, horizon: int = HORIZON) -> DataFrame:
    """One row per (series, target date t, horizon h). Same-weekday lags come from t - 7k with t - 7k <= origin."""
    w = Window.partitionBy("series").orderBy("date")
    base = ys
    for lag in LAGS:
        base = base.withColumn(f"lag{lag}", F.lag("y", lag).over(w))
    base = base.withColumn("m7", F.avg("y").over(w.rowsBetween(-6, 0))).withColumn("m28", F.avg("y").over(w.rowsBetween(-27, 0)))
    base = base.withColumn("n", F.row_number().over(w))
    at_origin = base.select(
        "series", F.col("date").alias("origin"), F.col("y").alias("last"), "m7", "m28", F.col("n").alias("n_o")
    )
    targets = base.select("series", "date", "y", *[f"lag{lag}" for lag in LAGS]).withColumn(
        "h", F.explode(F.sequence(F.lit(1), F.lit(horizon)))
    )
    targets = targets.withColumn("origin", F.date_sub("date", F.col("h")))
    df = targets.join(at_origin, ["series", "origin"])
    week2 = F.col("h") > 7  # beyond a week, the latest same weekday at or before the origin is t - 14
    for j in range(4):
        df = df.withColumn(f"sw{j + 1}", F.when(week2, F.col(f"lag{LAGS[j + 1]}")).otherwise(F.col(f"lag{LAGS[j]}")))
    df = df.withColumn("dow", F.dayofweek("date").cast("double")).withColumn("h", F.col("h").cast("double"))
    df = df.withColumn("holiday", F.col("date").cast("string").isin(HOLIDAYS).cast("double"))
    # enough history: four same-weekday lags and a full 28-day window at the origin
    return df.where(F.col("sw4").isNotNull() & (F.col("n_o") >= 28)).select(
        "series", "origin", "date", "h", "y", *FEATURES[2:], "dow"
    )


def _log(df: DataFrame) -> DataFrame:
    for c in ["sw1", "sw2", "sw3", "sw4", "last", "m7", "m28"]:
        df = df.withColumn(f"l_{c}", F.log1p(c))
    return df.withColumn("label", F.log1p("y"))


LOG_FEATURES = ["dow", "h", "holiday", "l_sw1", "l_sw2", "l_sw3", "l_sw4", "l_last", "l_m7", "l_m28"]


def gbt_forecast(train: DataFrame, test: DataFrame) -> DataFrame:
    assembler = VectorAssembler(inputCols=LOG_FEATURES, outputCol="features")
    # fixed hash partitioning and order, so the sampled split candidates are the same on any machine
    tr = assembler.transform(_log(train)).repartition(8, "series", "date", "h").sortWithinPartitions("series", "date", "h")
    model = GBTRegressor(featuresCol="features", labelCol="label", **GBT_PARAMS).fit(tr)
    pred = model.transform(assembler.transform(_log(test)))
    return pred.select("series", "origin", "date", "h", F.greatest(F.expm1("prediction"), F.lit(0.0)).alias("yhat"))


RIDGE_COLS = ["l_sw1", "l_sw2", "l_sw3", "l_sw4", "l_last", "l_m7", "l_m28", "holiday"]


def _ridge(pdf: pd.DataFrame) -> pd.DataFrame:
    """Per-series ridge on log features plus weekday dummies; trained on this series' training rows only."""

    def design(d):
        dows = np.stack([(d["dow"].to_numpy() == k).astype(float) for k in range(2, 8)], axis=1)
        return np.column_stack([np.ones(len(d)), d[RIDGE_COLS].to_numpy(), dows])

    tr, te = pdf[pdf["is_test"] == 0], pdf[pdf["is_test"] == 1]
    out = te[["series", "origin", "date", "h"]].copy()
    if len(tr) < 30 or len(te) == 0:
        out["yhat"] = np.expm1(te["l_sw1"].to_numpy())
        return out
    x, y = design(tr), tr["label"].to_numpy()
    lam = 1.0 * np.eye(x.shape[1])
    lam[0, 0] = 0.0
    beta = np.linalg.solve(x.T @ x + lam, x.T @ y)
    out["yhat"] = np.clip(np.expm1(design(te) @ beta), 0, None)
    return out


def ridge_forecast(train: DataFrame, test: DataFrame) -> DataFrame:
    cols = ["series", "origin", "date", "h", "dow", "label", *RIDGE_COLS]
    both = (
        _log(train)
        .select(*cols)
        .withColumn("is_test", F.lit(0))
        .unionByName(_log(test).select(*cols).withColumn("is_test", F.lit(1)))
    )
    schema = "series string, origin date, date date, h double, yhat double"
    return both.groupBy("series").applyInPandas(_ridge, schema)


def forecasts(ys: DataFrame, test_origins: list[dt.date]) -> DataFrame:
    """Every method's forecasts for the backtest origins at one level, with actuals.

    Columns: series, origin, date, h, y, method, yhat."""
    feats = features(ys).cache()
    first = min(test_origins)
    train = feats.where(F.col("date") <= F.lit(first))  # labels known at the first backtest origin
    test = feats.where(F.col("origin").isin(test_origins))
    actual = test.select("series", "origin", "date", "h", "y")
    keys = ["series", "origin", "date", "h"]
    mean4 = (F.col("sw1") + F.col("sw2") + F.col("sw3") + F.col("sw4")) / 4
    preds = [
        test.select(*keys, F.lit("seasonal_naive").alias("method"), F.col("sw1").alias("yhat")),
        test.select(*keys, F.lit("seasonal_mean4").alias("method"), mean4.alias("yhat")),
        gbt_forecast(train, test).withColumn("method", F.lit("gbt_global")),
        ridge_forecast(train, test).withColumn("method", F.lit("ridge_per_series")),
    ]
    out = preds[0]
    for p in preds[1:]:
        out = out.unionByName(p.select(*out.columns))
    return out.join(actual, ["series", "origin", "date", "h"])


def scale(ys: DataFrame, first_origin: dt.date) -> DataFrame:
    """MASE denominator per series: in-sample mean absolute error of the weekly seasonal naive before the backtest."""
    w = Window.partitionBy("series").orderBy("date")
    d = ys.where(F.col("date") <= F.lit(first_origin)).withColumn("e", F.abs(F.col("y") - F.lag("y", 7).over(w)))
    return d.groupBy("series").agg(F.avg("e").alias("scale"))


def metrics(fc: DataFrame, scales: DataFrame) -> dict:
    """WAPE, bias and mean MASE per method (MASE over series whose naive error is not zero)."""
    agg = fc.groupBy("method").agg(
        (F.sum(F.abs(F.col("yhat") - F.col("y"))) / F.sum("y")).alias("wape"),
        (F.sum(F.col("yhat") - F.col("y")) / F.sum("y")).alias("bias"),
        F.countDistinct("series").alias("series"),
        F.count(F.lit(1)).alias("rows"),
    )
    per = fc.groupBy("method", "series").agg(F.avg(F.abs(F.col("yhat") - F.col("y"))).alias("mae")).join(scales, "series")
    mase = per.where(F.col("scale") > 0).groupBy("method").agg(F.avg(F.col("mae") / F.col("scale")).alias("mase"))
    rows = agg.join(mase, "method", "left").collect()
    return {
        r["method"]: {
            k: (round(r[k], 4) if isinstance(r[k], float) else r[k]) for k in ["wape", "bias", "mase", "series", "rows"]
        }
        for r in rows
    }


def by_horizon(fc: DataFrame) -> dict:
    """WAPE for week 1 (h 1-7) and week 2 (h 8-14) per method."""
    wk = F.when(F.col("h") <= 7, "h1_7").otherwise("h8_14")
    rows = (
        fc.groupBy("method", wk.alias("wk")).agg((F.sum(F.abs(F.col("yhat") - F.col("y"))) / F.sum("y")).alias("wape")).collect()
    )
    out: dict = {}
    for r in rows:
        out.setdefault(r["method"], {})[r["wk"]] = round(r["wape"], 4)
    return out


def backtest(spark: SparkSession, gold: DataFrame) -> dict:
    levels = series_frames(gold)
    last_day = gold.agg(F.max("date")).first()[0]
    test_origins = origins(last_day)
    zone_map = gold.select(F.col("zone").cast("string").alias("series"), "borough").distinct()
    out = {
        "config": {
            "first_day": str(gold.agg(F.min("date")).first()[0]),
            "last_day": str(last_day),
            "origins": [str(o) for o in test_origins],
            "horizon_days": HORIZON,
            "gbt": GBT_PARAMS,
            "features": FEATURES,
        },
        "levels": {},
        "by_horizon": {},
        "bottom_up": {},
    }
    zone_fc = None
    for level, ys in levels.items():
        fc = forecasts(ys, test_origins).cache()
        out["levels"][level] = metrics(fc, scale(ys, test_origins[0]))
        out["by_horizon"][level] = by_horizon(fc)
        if level == "zone":
            zone_fc = fc
        else:
            # bottom-up: sum the zone forecasts (each method) to this level and score against this level's actuals
            up = zone_fc.join(zone_map, "series")
            key = F.col("borough") if level == "borough" else F.lit("NYC")
            summed = up.groupBy(key.alias("series"), "origin", "date", "h", "method").agg(F.sum("yhat").alias("yhat"))
            truth = fc.select("series", "origin", "date", "h", "y").distinct()
            out["bottom_up"][level] = metrics(summed.join(truth, ["series", "origin", "date", "h"]), scale(ys, test_origins[0]))
    return out
