"""A paced terminal walk-through of the recorded results (results/*.json), for the demo video. No Spark needed."""

import json
import time

from . import config as C


def _load(name: str) -> dict:
    return json.loads((C.RESULTS / f"{name}.json").read_text())


def say(line: str = "", pause: float = 0.25) -> None:
    print(line, flush=True)
    time.sleep(pause)


def run(pause: float = 0.25) -> None:
    bz, dq, neg, sel, bt = (_load(n) for n in ["bronze", "dq", "negative_fares", "model_selection", "backtest"])
    rows = sum(bz["rows_by_month"].values())
    say(f"bronze: {len(bz['months'])} monthly files, {rows:,} trips", pause)
    new = [m["month"] for m in bz["months"] if "request_source" in m["unused_columns"]]
    say(f"schema drift: new column request_source from {new[0]}", pause)
    say()
    nov = [r for r in neg["by_payment_type"] if r["src_month"] == "2025-11"]
    total = neg["rows_in_month"]["2025-11"]
    share = sum(r["negative_rows"] for r in nov) / total
    say(f"silver gate, first rule 'amounts >= 0': Nov 2025 negative fares {share:.1%}", pause)
    if share > 0.05:
        say("RUN STOPPED: amounts_non_negative (2025-11) above its 5% limit", pause * 4)
    say()
    say("payment type   negative rows   mirror of a real trip", pause)
    for r in nov:
        say(f"{r['payment_type']:>12}   {r['negative_rows']:>13,}   {r['exact_positive_twin']:>21,}", pause)
    flex = dq["rules"]["flex_fare_amount_non_negative"]
    say(
        f"flex fare (type 0): real trips with a bad amount, kept: {flex['worst_month_rate']:.1%} of {flex['worst_month']}",
        pause * 4,
    )
    say(f"silver: {dq['silver_rows']:,} trips kept, {dq['quarantined_rows']:,} quarantined", pause)
    say()
    v = sel["levels"]["zone"]
    say(
        f"validation, zone WAPE: GBT log target {v['gbt_log_level']['wape']:.1%}, relative {v['gbt_relative']['wape']:.1%}",
        pause,
    )
    say(f"chosen on validation: {sel['chosen']}", pause * 4)
    say()
    z = bt["levels"]["zone"]
    say(f"test backtest, {len(bt['config']['origins'])} weekly origins, 263 zones", pause)
    for m in ["seasonal_naive", "seasonal_mean4", "ridge_per_series", "gbt_global"]:
        say(f"  {m:<18} WAPE {z[m]['wape']:.1%}   MASE {z[m]['mase']:.2f}", pause)
    b, u = bt["levels"]["borough"]["gbt_global"], bt["bottom_up"]["borough"]["gbt_global"]
    say(f"borough level, GBT: direct {b['wape']:.1%} vs zones summed {u['wape']:.1%}", pause * 4)
