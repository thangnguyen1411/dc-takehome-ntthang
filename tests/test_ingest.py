from pathlib import Path

import pytest

from curation_agent.ingest import Ingestor

DATA = Path(__file__).resolve().parents[1] / "data" / "tasks.csv"


def test_load_tasks_reads_all_rows():
    tasks = Ingestor().load(DATA)
    assert len(tasks) == 10
    assert tasks[0].task_id == "task_001"
    assert tasks[2].ground_truth_signal == "contradicted"


def test_load_tasks_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        Ingestor().load("does/not/exist.csv")


def test_load_tasks_missing_columns_raises(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("task_id\tquestion\nt1\tq?\n", encoding="utf-8")
    with pytest.raises(ValueError):
        Ingestor().load(bad)
