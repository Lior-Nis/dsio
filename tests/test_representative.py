"""The representative-tier runner records and compares parity faithfully."""

from __future__ import annotations

from tools import representative


def test_compare_is_exact_by_default_and_reports_moved_and_missing_metrics() -> None:
    baseline = {"accuracy": 0.5, "rmse": 1.0, "dropped": 2.0}
    observed = {"accuracy": 0.5, "rmse": 1.0 + 1e-12, "added": 3.0}

    moved = representative.compare(baseline, observed, tolerance=0.0)

    assert moved == {
        "added": (None, 3.0),
        "dropped": (2.0, None),
        "rmse": (1.0, 1.0 + 1e-12),
    }
    assert "rmse" not in representative.compare(baseline, observed, tolerance=1e-9)


def test_single_and_multi_model_results_flatten_to_one_record() -> None:
    single = {"metrics": {"rmse": 2}, "train_run_id": "t", "evaluation_run_id": "e", "seed": 19}
    multi = {
        "data_run_id": "d",
        "models": {
            "fused": {"metrics": {"accuracy": 0.25}, "evaluation_run_id": "ef"},
            "tabular": {"metrics": {"accuracy": 0.5}, "evaluation_run_id": "et"},
        },
    }

    assert representative._metrics(single) == {"rmse": 2.0}
    assert representative._runs(single) == {"evaluation_run_id": "e", "train_run_id": "t"}
    assert representative._metrics(multi) == {"fused.accuracy": 0.25, "tabular.accuracy": 0.5}
    assert representative._runs(multi) == {
        "data_run_id": "d",
        "fused.evaluation_run_id": "ef",
        "tabular.evaluation_run_id": "et",
    }


def test_every_kaggle_consumer_has_a_representative_configuration() -> None:
    from tools import consumer_metrics

    kaggle = {
        row.consumer.removeprefix("reference_projects/kaggle/").replace("/", "_")
        for row in consumer_metrics.count_lines()
        if row.kaggle
    }
    assert kaggle == set(representative.CONSUMERS)
