"""Wall time of the silver and gold stages (45M+ rows in) by core count and shuffle partitions.

Each configuration runs in its own process, because a Spark session's master is fixed once it starts.
Output goes to a scratch area of the lake so the real silver and gold tables are untouched.
"""

import json
import subprocess
import sys
import time

from . import config as C

STAGE = """
import json, sys, time
from demand_lakehouse import config as C, gold, silver
from demand_lakehouse.spark import get_spark
cores, parts = int(sys.argv[1]), int(sys.argv[2])
spark = get_spark(f"local[{cores}]", partitions=parts)
tmp = C.LAKE / "scaling"
t0 = time.perf_counter()
silver.run(spark, C.BRONZE, tmp / "silver", tmp / "quarantine")
t1 = time.perf_counter()
gold.run(spark, tmp / "silver", C.RAW / "taxi_zone_lookup.csv", tmp / "gold")
t2 = time.perf_counter()
print(json.dumps({"silver_s": round(t1 - t0, 1), "gold_s": round(t2 - t1, 1)}))
"""


def one(cores: int, partitions: int) -> dict:
    t0 = time.perf_counter()
    out = subprocess.run([sys.executable, "-c", STAGE, str(cores), str(partitions)], capture_output=True, text=True, check=True)
    stages = json.loads(out.stdout.strip().splitlines()[-1])
    return {"cores": cores, "shuffle_partitions": partitions, **stages, "process_s": round(time.perf_counter() - t0, 1)}


def run(cores: list[int]) -> dict:
    runs = [one(n, 16) for n in cores]
    runs += [one(max(cores), p) for p in (4, 200)]
    base = runs[0]["silver_s"] + runs[0]["gold_s"]
    for r in runs:
        r["speedup_vs_first"] = round(base / (r["silver_s"] + r["gold_s"]), 2)
    rows = sum(json.loads((C.RESULTS / "bronze.json").read_text())["rows_by_month"].values())
    return {
        "input_rows": rows,
        "runs": runs,
        "note": "one laptop (Apple M2, 8 cores, 8 GB); local mode; times include Delta writes",
    }
