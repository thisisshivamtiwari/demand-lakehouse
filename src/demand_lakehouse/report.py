"""README tables, generated from results/*.json (between the results markers)."""

import json
import re
from pathlib import Path

START, END = "<!-- results:start -->", "<!-- results:end -->"
METHODS = ["seasonal_naive", "seasonal_mean4", "ridge_per_series", "gbt_global"]


def _load(results: Path, name: str) -> dict | None:
    p = results / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _pct(x) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def backtest_table(bt: dict) -> str:
    lines = ["| Level | Method | WAPE | MASE | Bias | Series |", "|---|---|---|---|---|---|"]
    for level in ["zone", "borough", "city"]:
        for m in METHODS:
            r = bt["levels"][level][m]
            lines.append(f"| {level} | {m} | {_pct(r['wape'])} | {r['mase']:.3f} | {_pct(r['bias'])} | {r['series']} |")
        if level in bt["bottom_up"]:
            for m in ["gbt_global", "ridge_per_series"]:
                r = bt["bottom_up"][level][m]
                name = f"{m}, zones summed (bottom-up)"
                lines.append(f"| {level} | {name} | {_pct(r['wape'])} | {r['mase']:.3f} | {_pct(r['bias'])} | {r['series']} |")
    return "\n".join(lines)


def horizon_table(bt: dict) -> str:
    lines = ["| Method | Zone WAPE, days 1-7 | Zone WAPE, days 8-14 |", "|---|---|---|"]
    for m in METHODS:
        r = bt["by_horizon"]["zone"][m]
        lines.append(f"| {m} | {_pct(r['h1_7'])} | {_pct(r['h8_14'])} |")
    return "\n".join(lines)


def dq_table(dq: dict) -> str:
    lines = ["| Rule | Rows failing | Rate | Worst month | Run fails if a month is above |", "|---|---|---|---|---|"]
    for name, r in dq["rules"].items():
        worst = f"{100 * r['worst_month_rate']:.2f}% ({r['worst_month']})"
        limit = "warn only (kept)" if r["max_rate"] is None else f"{100 * r['max_rate']:.1f}%"
        lines.append(f"| {name} | {r['failed']:,} | {100 * r['rate']:.2f}% | {worst} | {limit} |")
    lines.append(
        f"\n{dq['rows']:,} bronze rows; {dq['silver_rows']:,} reached silver and {dq['quarantined_rows']:,} went to quarantine."
    )
    return "\n".join(lines)


def drift_lines(bz: dict) -> str:
    """Casts applied to every file, and columns that appear or vanish between monthly files."""
    months = bz["months"]
    out = []
    casts = {(d["column"], d.get("from"), d.get("to")) for m in months for d in m["drift"] if d["change"] == "cast"}
    for col, src, dst in sorted(casts):
        n = sum(any(d["column"] == col and d["change"] == "cast" for d in m["drift"]) for m in months)
        out.append(f"- `{col}` arrives as {src}, cast to {dst} ({n} of {len(months)} files)")
    for m in months:
        for d in m["drift"]:
            if d["change"] != "cast":
                out.append(
                    f"- {m['month']}: `{d['column']}` {d['change']} " + (f"(arrived as `{d['from']}`)" if d.get("from") else "")
                )
    for prev, cur in zip(months, months[1:], strict=False):
        new = sorted(set(cur["unused_columns"]) - set(prev["unused_columns"]))
        gone = sorted(set(prev["unused_columns"]) - set(cur["unused_columns"]))
        if new:
            out.append(
                f"- {cur['month']}: new column(s) " + ", ".join(f"`{c}`" for c in new) + " (kept in the raw file, not used)"
            )
        if gone:
            out.append(f"- {cur['month']}: column(s) dropped " + ", ".join(f"`{c}`" for c in gone))
    return "\n".join(out) or "No schema drift across the files."


def run_table(run: dict, stage_s: dict) -> str:
    lines = ["| Task | Status | Seconds |", "|---|---|---|"]
    for t in run["tasks"]:
        lines.append(f"| {t['task_key']} | {t['status']} | {stage_s.get(t['task_key'], t.get('seconds', '-'))} |")
    return "\n".join(lines)


def scaling_table(sc: dict) -> str:
    lines = ["| Cores | Shuffle partitions | Silver (s) | Gold (s) | Speed-up vs 1 core |", "|---|---|---|---|---|"]
    for r in sc["runs"]:
        lines.append(f"| {r['cores']} | {r['shuffle_partitions']} | {r['silver_s']} | {r['gold_s']} | {r['speedup_vs_first']}x |")
    return "\n".join(lines) + f"\n\nInput: {sc['input_rows']:,} bronze rows. {sc['note']}."


def render(results: Path) -> str:
    bt, dq, bz, run, sc = (_load(results, n) for n in ["backtest", "dq", "bronze", "run", "scaling"])
    parts = []
    if bt:
        c = bt["config"]
        parts += [
            f"**Backtest**: {len(c['origins'])} weekly origins from {c['origins'][0]} to {c['origins'][-1]}, "
            f"forecasting days 1-{c['horizon_days']} after each origin. WAPE = total absolute error / total demand.\n",
            backtest_table(bt),
            "\n**Error by horizon (zone level)**\n",
            horizon_table(bt),
        ]
    if dq:
        parts += ["\n**Data-quality expectations (silver)**\n", dq_table(dq)]
    if bz:
        parts += ["\n**Schema drift found in bronze**\n", drift_lines(bz)]
    if run:
        stage_s = {}
        for name, key in [("bronze", "bronze"), ("dq", "silver"), ("gold", "gold"), ("backtest", "backtest")]:
            d = _load(results, name)
            if d and "seconds" in d:
                stage_s[key] = d["seconds"]
        parts += ["\n**Last full run (jobs.yaml)**\n", run_table(run, stage_s)]
    if sc:
        parts += ["\n**Scaling (silver + gold stages)**\n", scaling_table(sc)]
    return "\n".join(parts)


def write(results: Path, readme: Path) -> None:
    text = readme.read_text()
    body = f"{START}\n{render(results)}\n{END}"
    readme.write_text(re.sub(re.escape(START) + ".*?" + re.escape(END), lambda _: body, text, flags=re.S))
