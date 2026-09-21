"""Deterministic identity over safe normalized project configuration."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from collections.abc import Iterator, Mapping
from typing import Any

import pytest


def test_equivalent_supported_representations_have_one_identity() -> None:
    from dsio.tracking import execution_identity, normalize

    left = {
        "model": {"width": 32, "layers": (1, 2, 3)},
        "roles": {"train", "validation"},
    }
    right = {
        "roles": frozenset({"validation", "train"}),
        "model": {"layers": [1, 2, 3], "width": 32},
    }

    assert normalize(left) == normalize(right)
    assert execution_identity(left) == execution_identity(right)


def test_ordered_sequence_and_unordered_set_do_not_share_an_identity() -> None:
    from dsio.tracking import execution_identity

    assert execution_identity({"values": [1, 2]}) != execution_identity(
        {"values": {1, 2}}
    )


@pytest.mark.parametrize(
    ("field", "first", "second"),
    [
        ("seed", 7, 8),
        ("dataset", "sha256:a", "sha256:b"),
        ("split", "sha256:c", "sha256:d"),
        ("code", "commit-a", "commit-b"),
        ("environment", "lock-a", "lock-b"),
    ],
)
def test_every_declared_meaningful_input_changes_identity(
    field: str,
    first: object,
    second: object,
) -> None:
    from dsio.tracking import execution_identity

    baseline = {"seed": 7, "dataset": "sha256:a", "split": "sha256:c"}
    assert execution_identity({**baseline, field: first}) != execution_identity(
        {**baseline, field: second}
    )


def test_component_reference_changes_identity() -> None:
    from dsio.tracking import execution_identity

    config = {"seed": 7}
    assert execution_identity(
        config,
        components={"model": "project.models:SmallNet"},
    ) != execution_identity(
        config,
        components={"model": "project.models:LargeNet"},
    )


def test_component_references_must_be_explicit_strings() -> None:
    from dsio.contracts import NonCanonicalValueError
    from dsio.tracking import execution_identity

    with pytest.raises(NonCanonicalValueError, match="component reference"):
        execution_identity({"seed": 7}, components={"model": 42})  # type: ignore[dict-item]

    with pytest.raises(NonCanonicalValueError, match="component names.*int"):
        execution_identity(  # type: ignore[arg-type,dict-item]
            {"seed": 7},
            components={1: object()},
        )


def test_dsio_version_changes_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    from dsio.tracking import execution_identity

    provenance = importlib.import_module("dsio.tracking.provenance")
    monkeypatch.setattr(provenance, "version", lambda _: "1.0.0")
    first = execution_identity({"seed": 7})
    monkeypatch.setattr(provenance, "version", lambda _: "2.0.0")
    assert execution_identity({"seed": 7}) != first


def test_identity_is_independent_of_process_hash_scheduling() -> None:
    probe = """
from dsio.tracking import execution_identity
print(execution_identity({'roles': {'train', 'validation', 'test'}, 'seed': 7}))
"""
    identities = []
    for hash_seed in ("1", "987654"):
        identities.append(
            subprocess.check_output(
                [sys.executable, "-c", probe],
                env={**os.environ, "PYTHONHASHSEED": hash_seed},
                text=True,
            ).strip()
        )
    assert identities[0] == identities[1]


def test_secret_and_ephemeral_fields_are_removed_recursively_before_hashing() -> None:
    from dsio.tracking import execution_identity, normalize

    first = {
        "database": {"user": "reader", "password": "SECRET-ONE"},
        "task_run_id": "runtime-a",
        "nested": [{"token": "TOKEN-ONE", "value": 3}],
    }
    second = {
        "database": {"user": "reader", "password": "SECRET-TWO"},
        "task_run_id": "runtime-b",
        "nested": [{"token": "TOKEN-TWO", "value": 3}],
    }
    options = {"secrets": {"password", "token"}, "ephemeral": {"task_run_id"}}

    normalized = normalize(first, **options)
    rendered = repr(normalized)
    assert "password" not in rendered
    assert "token" not in rendered
    assert "SECRET-ONE" not in rendered
    assert "TOKEN-ONE" not in rendered
    assert "task_run_id" not in rendered
    assert execution_identity(first, **options) == execution_identity(second, **options)


def test_secret_value_is_not_inspected_or_exposed_by_normalization_error() -> None:
    from dsio.contracts import NonCanonicalValueError
    from dsio.tracking import execution_identity

    class Credential:
        def __repr__(self) -> str:
            return "SENTINEL-SECRET"

    assert execution_identity(
        {"seed": 7, "credential": Credential()},
        secrets={"credential"},
    )

    with pytest.raises(NonCanonicalValueError) as caught:
        execution_identity({"seed": 7, "unsupported": Credential()})

    message = str(caught.value)
    assert "Credential" in message
    assert "SENTINEL-SECRET" not in message


def test_cyclic_container_is_rejected_as_noncanonical_without_rendering_values() -> None:
    from dsio.contracts import NonCanonicalValueError
    from dsio.tracking import execution_identity

    cycle: list[object] = []
    cycle.append(cycle)

    with pytest.raises(NonCanonicalValueError, match="cyclic container"):
        execution_identity({"cycle": cycle})


def test_reserved_set_marker_cannot_be_supplied_as_project_configuration() -> None:
    from dsio.contracts import NonCanonicalValueError
    from dsio.tracking import execution_identity

    with pytest.raises(NonCanonicalValueError, match="reserved"):
        execution_identity({"$dsio.set": [1, 2]})


def test_lazy_mapping_secret_is_omitted_before_its_value_is_fetched() -> None:
    from dsio.tracking import normalize

    class LazyConfig(Mapping[str, object]):
        def __getitem__(self, key: str) -> object:
            if key == "password":
                raise RuntimeError("SENTINEL-SECRET-WAS-FETCHED")
            return 7

        def __iter__(self) -> Iterator[str]:
            return iter(("seed", "password"))

        def __len__(self) -> int:
            return 2

    assert normalize(LazyConfig(), secrets={"password"}) == {"seed": 7}


def test_bare_string_selector_names_one_field_instead_of_its_characters() -> None:
    from dsio.tracking import execution_identity, normalize

    first = {"seed": 7, "password": "SENTINEL-ONE"}
    second = {"seed": 7, "password": "SENTINEL-TWO"}
    assert normalize(first, secrets="password") == {"seed": 7}
    assert execution_identity(first, secrets="password") == execution_identity(
        second,
        secrets="password",
    )


@pytest.mark.parametrize("selector", [b"password", [b"password"], ["token", 7]])
def test_non_string_secret_selectors_fail_closed_without_rendering_values(
    selector: Any,
) -> None:
    from dsio.contracts import NonCanonicalValueError
    from dsio.tracking import normalize

    with pytest.raises(NonCanonicalValueError, match="field selectors.*(bytes|int)") as caught:
        normalize(
            {"password": "SENTINEL-PASSWORD", "token": "SENTINEL-TOKEN"},
            secrets=selector,
        )

    assert "SENTINEL" not in str(caught.value)


def test_one_shot_secret_selectors_are_not_consumed_before_filtering() -> None:
    from dsio.tracking import normalize

    selectors: Any = (name for name in ["password"])

    assert normalize(
        {"password": "SENTINEL-PASSWORD", "seed": 7},
        secrets=selectors,
    ) == {"seed": 7}
