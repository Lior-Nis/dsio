"""The consumer measurement tool counts what the warehouse must delete."""

from __future__ import annotations

from pathlib import Path

from tools import consumer_metrics


def _consumer(root: Path) -> Path:
    project = root / "reference_projects" / "kaggle" / "demo"
    (project / "tasks").mkdir(parents=True)
    (project / "flow.py").write_text("def flow():\n    return 1\n")
    (project / "data.py").write_text("ROWS = 1\nCOLUMNS = 2\nTARGET = 3\n")
    (project / "tasks" / "data.py").write_text("def ingest():\n    return 1\n")
    (project / "components.py").write_text(
        "from torch import nn\n"
        "from torch.utils.data import Dataset\n"
        "\n"
        "class Model(nn.Module):\n"
        "    pass\n"
        "\n"
        "class Rows(Dataset):\n"
        "    pass\n"
        "\n"
        "def validate(output):\n"
        "    return None\n"
        "\n"
        "def pad(items):\n"
        "    return items\n"
        "\n"
        "def helper():\n"
        "    return None\n"
    )
    (project / "tasks" / "training.py").write_text(
        "from demo.components import pad, validate\n"
        "build(validator=validate)\n"
        "DataModule(collate_fn=pad)\n"
    )
    nested = project / "sequence"
    nested.mkdir()
    (nested / "flow.py").write_text("def flow():\n    return 2\n")
    (nested / "components.py").write_text("class Other:\n    pass\n")
    return project


def test_lines_are_split_by_category_and_nested_consumers_stay_separate(tmp_path: Path) -> None:
    _consumer(tmp_path)
    lines = {
        row.consumer: row for row in consumer_metrics.count_lines(tmp_path / "reference_projects")
    }

    demo = lines["reference_projects/kaggle/demo"]
    assert (demo.ingestion, demo.model_side, demo.wiring, demo.flow) == (5, 17, 3, 2)
    assert demo.non_ingestion == 22
    assert demo.kaggle
    nested = lines["reference_projects/kaggle/demo/sequence"]
    assert (nested.model_side, nested.flow) == (2, 2)


def test_banned_definitions_find_modules_datasets_and_injected_callables(tmp_path: Path) -> None:
    _consumer(tmp_path)
    findings = consumer_metrics.banned_definitions(tmp_path / "reference_projects")

    found = {(finding.name, finding.kind) for finding in findings}
    assert found == {
        ("Model", "nn.Module/Dataset subclass"),
        ("Rows", "nn.Module/Dataset subclass"),
        ("validate", "passed as validator"),
        ("pad", "passed as collate_fn"),
    }
    assert all(finding.consumer == "reference_projects/kaggle/demo" for finding in findings)


def test_the_real_portfolio_is_measured() -> None:
    lines = consumer_metrics.count_lines()
    assert sum(1 for row in lines if row.kaggle) >= 9
    assert consumer_metrics.banned_definitions(), "the warehouse has consumer code left to absorb"
