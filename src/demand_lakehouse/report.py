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
    lines = ["| Rule | Rows failing | Rate | Run fails above |", "|---|---|---|---|"]
    for name, r in dq["rules"].items():
        lines.append(f"| {name} | {r['failed']:,} | {100 * r['rate']:.2f}% | {100 * r['max_rate']:.1f}% |")
    lines.append(
        f"\n{dq['rows']:,} bronze rows; {dq['silver_rows']:,} reached silver and {dq['quarantined_rows']:,} went to quarantine."
    )
    return "\n".join(lines)


def drift_lines(bz: dict) -> str:
    seen: dict[str, set] = {}
    for m in bz["months"]:
        for d in m["drift"]:
            k = f"`{d['column']}` {d['change']} " + (
                f"{d.get('from')} -> {d['to']}" if d["change"] == "cast" else f"(filled with null {d['to']})"
            )
            seen.setdefault(k, set()).add(m["month"])
    if not seen:
        return "No schema drift across the 12 files."
    return "\n".join(f"- {k}: {len(v)} of {len(bz['months'])} files" for k, v in sorted(seen.items()))


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
