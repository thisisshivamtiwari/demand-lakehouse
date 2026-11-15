import pytest

from demand_lakehouse import runner

JOB = """
name: t
tasks:
  - task_key: a
    max_retries: 2
  - task_key: b
    depends_on: [{task_key: a}]
  - task_key: c
    depends_on: [{task_key: b}]
  - task_key: d
"""


def test_order_respects_dependencies():
    tasks = [
        {"task_key": "c", "depends_on": [{"task_key": "b"}]},
        {"task_key": "b", "depends_on": [{"task_key": "a"}]},
        {"task_key": "a"},
    ]
    assert runner.order(tasks) == ["a", "b", "c"]


def test_order_rejects_cycles_and_unknown_tasks():
    with pytest.raises(ValueError, match="cycle"):
        runner.order([{"task_key": "a", "depends_on": [{"task_key": "b"}]}, {"task_key": "b", "depends_on": [{"task_key": "a"}]}])
    with pytest.raises(ValueError, match="unknown"):
        runner.order([{"task_key": "a", "depends_on": [{"task_key": "x"}]}])


def test_retries_then_skips_downstream(tmp_path):
    job = tmp_path / "jobs.yaml"
    job.write_text(JOB)
    calls = {"a": 0}

    def flaky():
        calls["a"] += 1
        if calls["a"] < 2:
            raise RuntimeError("transient")
        return {}

    def broken():
        raise RuntimeError("bad data")

    rec = runner.run(job, {"a": flaky, "b": broken, "c": dict, "d": dict}, log=tmp_path / "run.json")
    by = {t["task_key"]: t for t in rec["tasks"]}
    assert by["a"]["status"] == "ok" and by["a"]["attempts"] == 2
    assert by["b"]["status"] == "failed" and "bad data" in by["b"]["error"]
    assert by["c"]["status"] == "skipped"
    assert by["d"]["status"] == "ok"
    assert rec["status"] == "failed" and (tmp_path / "run.json").exists()
