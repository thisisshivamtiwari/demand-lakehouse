"""A small job runner for jobs.yaml: dependency order, retries, and a run log in results/run.json."""

import json
import time
from collections.abc import Callable
from pathlib import Path

import yaml


def order(tasks: list[dict]) -> list[str]:
    """Topological order of task keys; raises on unknown dependencies or cycles."""
    deps = {t["task_key"]: [d["task_key"] for d in t.get("depends_on", [])] for t in tasks}
    for key, ds in deps.items():
        missing = [d for d in ds if d not in deps]
        if missing:
            raise ValueError(f"{key} depends on unknown task(s) {missing}")
    done: list[str] = []
    while len(done) < len(deps):
        ready = [k for k, ds in deps.items() if k not in done and all(d in done for d in ds)]
        if not ready:
            raise ValueError(f"cycle among {sorted(set(deps) - set(done))}")
        done.extend(ready)
    return done


def run(job_file: Path, actions: dict[str, Callable[[], dict]], only: list[str] | None = None, log: Path | None = None) -> dict:
    """Run every task (or only `only`) in dependency order. A failed task skips everything downstream of it."""
    job = yaml.safe_load(job_file.read_text())
    tasks = {t["task_key"]: t for t in job["tasks"]}
    record = {"job": job["name"], "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "tasks": []}
    failed: set[str] = set()
    for key in order(job["tasks"]):
        if only and key not in only:
            continue
        upstream = [d["task_key"] for d in tasks[key].get("depends_on", [])]
        if any(u in failed for u in upstream):
            failed.add(key)
            record["tasks"].append({"task_key": key, "status": "skipped", "reason": "upstream failed"})
            continue
        attempts, t0 = 0, time.perf_counter()
        while True:
            attempts += 1
            try:
                actions[key]()
                status, error = "ok", None
                break
            except Exception as e:  # the run log records any task failure
                status, error = "failed", f"{type(e).__name__}: {e}"[:500]
                if attempts > tasks[key].get("max_retries", 0):
                    failed.add(key)
                    break
        entry = {"task_key": key, "status": status, "attempts": attempts, "seconds": round(time.perf_counter() - t0, 1)}
        if error:
            entry["error"] = error
        record["tasks"].append(entry)
    record["status"] = "failed" if failed else "ok"
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(json.dumps(record, indent=1) + "\n")
    return record
