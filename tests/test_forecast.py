"""The backtest must never see the future: every feature comes from days up to the origin."""

import datetime as dt

import pandas as pd

from demand_lakehouse import forecast

START = dt.date(2026, 1, 1)


def test_origins_are_weekly_and_leave_a_full_horizon():
    o = forecast.origins(dt.date(2026, 9, 30))
    assert len(o) == forecast.N_ORIGINS and o[-1] == dt.date(2026, 9, 16)
    assert all((b - a).days == 7 for a, b in zip(o, o[1:], strict=False))


def test_features_use_only_days_up_to_the_origin(spark):
    # y on day k is k, so every feature value names the day it came from
    days = [START + dt.timedelta(days=k) for k in range(80)]
    ys = spark.createDataFrame(pd.DataFrame({"series": "a", "date": days, "y": [float(k) for k in range(80)]}))
    f = forecast.features(ys).toPandas()
    day = {d: k for k, d in enumerate(days)}
    t, o = f["date"].map(day), f["origin"].map(day)
    assert ((t - o) == f["h"]).all() and f["h"].between(1, 14).all()
    assert (f["last"] == o).all()
    assert (f["m28"] == o - 13.5).all() and (f["m7"] == o - 3).all()
    week = (f["h"] > 7).astype(int)
    assert (f["sw1"] == t - 7 * (1 + week)).all() and (f["sw4"] == t - 7 * (4 + week)).all()
    assert (f["sw1"] <= o).all()  # the latest same-weekday value is never after the origin


def test_metrics_perfect_and_biased(spark):
    rows = [("a", START, START, 1.0, 10.0, m, yhat) for m, yhat in [("perfect", 10.0), ("high", 12.0)]]
    fc = spark.createDataFrame(rows, "series string, origin date, date date, h double, y double, method string, yhat double")
    scales = spark.createDataFrame([("a", 2.0)], "series string, scale double")
    m = forecast.metrics(fc, scales)
    assert m["perfect"]["wape"] == 0 and m["perfect"]["mase"] == 0
    assert m["high"]["wape"] == 0.2 and m["high"]["bias"] == 0.2 and m["high"]["mase"] == 1.0
