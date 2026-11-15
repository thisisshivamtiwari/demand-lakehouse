"""Command line: download | run [--only task ...] | ci | scaling | report | investigate."""

import argparse
import json
import sys
import time
import urllib.request

from . import config as C


def save(name: str, data: dict) -> None:
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    (C.RESULTS / f"{name}.json").write_text(json.dumps(data, indent=1, default=str) + "\n")


def download() -> None:
    C.RAW.mkdir(parents=True, exist_ok=True)
    targets = [(C.TLC_URL.format(month=m), C.RAW / f"yellow_tripdata_{m}.parquet") for m in C.MONTHS]
    targets.append((C.ZONES_URL, C.RAW / "taxi_zone_lookup.csv"))
    for url, path in targets:
        if not path.exists():
            print("downloading", path.name, flush=True)
            urllib.request.urlretrieve(url, path)


def actions(spark) -> dict:
    from pyspark.sql import functions as F

    from . import bronze, forecast, gold, report, silver

    def timed(name, fn):
        def go():
            t0 = time.perf_counter()
            out = fn()
            out["seconds"] = round(time.perf_counter() - t0, 1)
            save(name, out)
            return out

        return go

    def silver_run():
        try:
            return silver.run(spark, C.BRONZE, C.SILVER, C.QUARANTINE)
        except silver.ExpectationFailed as e:
            save("dq_failed", e.report)  # keep the evidence of the stopped run
            raise

    def backtest():
        g = spark.read.format("delta").load(str(C.GOLD)).withColumn("date", F.col("date").cast("date"))
        return forecast.backtest(spark, g)

    return {
        "bronze": timed("bronze", lambda: bronze.run(spark, C.RAW, C.MONTHS, C.BRONZE)),
        "silver": timed("dq", silver_run),
        "gold": timed("gold", lambda: gold.run(spark, C.SILVER, C.RAW / "taxi_zone_lookup.csv", C.GOLD, C.GOLD_SNAPSHOT)),
        "backtest": timed("backtest", backtest),
        "report": lambda: report.write(C.RESULTS, C.ROOT / "README.md") or {},
    }


def ci() -> int:
    """Replay the backtest from the committed gold snapshot; it must reproduce results/backtest.json and beat the baseline."""
    from . import forecast
    from .spark import get_spark

    spark = get_spark("local[2]", partitions=8)
    gold = spark.read.parquet(str(C.GOLD_SNAPSHOT))
    got = forecast.backtest(spark, gold)
    want = json.loads((C.RESULTS / "backtest.json").read_text())
    problems = []
    for section in ["levels", "bottom_up"]:
        for level, methods in want[section].items():
            for method, m in methods.items():
                for k in ["wape", "mase", "bias"]:
                    a, b = got[section][level][method][k], m[k]
                    if a is None or b is None or abs(a - b) > 2e-3:
                        problems.append(f"{section}.{level}.{method}.{k}: got {a}, recorded {b}")
    z = got["levels"]["zone"]
    if not z["gbt_global"]["wape"] < z["seasonal_naive"]["wape"]:
        problems.append("gbt_global does not beat seasonal_naive at zone level")
    print(json.dumps(got["levels"]["zone"], indent=1))
    for p in problems:
        print("FAIL", p)
    print("eval gate:", "failed" if problems else "passed")
    return 1 if problems else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="demand_lakehouse")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download")
    r = sub.add_parser("run")
    r.add_argument("--only", nargs="*")
    r.add_argument("--master", default="local[*]")
    sub.add_parser("ci")
    s = sub.add_parser("scaling")
    s.add_argument("--cores", nargs="*", type=int, default=[1, 2, 4, 8])
    sub.add_parser("report")
    sub.add_parser("investigate", help="negative fares in Oct-Nov 2025: reversals vs flex-fare rows")
    a = p.parse_args(argv)
    if a.cmd == "download":
        download()
        return 0
    if a.cmd == "ci":
        return ci()
    if a.cmd == "report":
        from . import report

        report.write(C.RESULTS, C.ROOT / "README.md")
        return 0
    if a.cmd == "investigate":
        from .investigate import negative_fares
        from .spark import get_spark

        save("negative_fares", negative_fares(get_spark(), C.BRONZE, ["2025-10", "2025-11"]))
        return 0
    if a.cmd == "scaling":
        from . import scaling

        save("scaling", scaling.run(a.cores))
        return 0
    from . import runner
    from .spark import get_spark

    spark = get_spark(a.master)
    rec = runner.run(C.ROOT / "jobs.yaml", actions(spark), only=a.only, log=C.RESULTS / "run.json")
    print(json.dumps(rec, indent=1))
    return 0 if rec["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
