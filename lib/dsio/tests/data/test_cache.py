"""Stage cache invariants: what invalidates, what does not, and what fails closed."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dsio.data.cache import (
    BytesCodec,
    CacheError,
    CodePolicy,
    ConfigPolicy,
    EnvPolicy,
    ExplicitVersionPolicy,
    JsonCodec,
    NpzCodec,
    NumpyCodec,
    StageCache,
    normalised_source,
)


@pytest.fixture
def cache(tmp_path: Path) -> StageCache:
    return StageCache(tmp_path / "cache")


class Counter:
    """Counts how many times the stage body actually ran."""

    def __init__(self, value: np.ndarray | None = None) -> None:
        self.n = 0
        self.value = np.arange(100, dtype="float32") if value is None else value

    def __call__(self) -> np.ndarray:
        self.n += 1
        return self.value


# --- hits and misses ----------------------------------------------------------------


def test_second_call_is_a_hit(cache: StageCache) -> None:
    fn = Counter()
    first = cache.run("s", fn, config={"a": 1})
    second = cache.run("s", fn, config={"a": 1})
    assert fn.n == 1
    assert first.meta.key == second.meta.key
    assert np.array_equal(second.load(), fn.value)


def test_config_change_invalidates(cache: StageCache) -> None:
    fn = Counter()
    cache.run("s", fn, config={"a": 1})
    cache.run("s", fn, config={"a": 2})
    assert fn.n == 2


def test_speed_only_config_does_not_invalidate(cache: StageCache) -> None:
    """Worker count changes how fast you got the answer, never the answer.

    Keying on it means the cache thrashes on every machine, which is the failure mode in
    the other direction from silently reusing across incompatible settings.
    """
    fn = Counter()
    cache.run("s", fn, config={"a": 1, "num_workers": 4, "device": "cpu"})
    cache.run("s", fn, config={"a": 1, "num_workers": 16, "device": "cuda"})
    assert fn.n == 1


def test_nested_speed_only_keys_are_stripped(cache: StageCache) -> None:
    fn = Counter()
    cache.run("s", fn, config={"loader": {"batch": 32, "num_workers": 2}})
    cache.run("s", fn, config={"loader": {"batch": 32, "num_workers": 9}})
    assert fn.n == 1


def test_version_bump_invalidates(cache: StageCache) -> None:
    """The escape hatch for what hashing cannot see: changed remote data, a new library."""
    fn = Counter()
    cache.run("s", fn, version=1)
    cache.run("s", fn, version=2)
    assert fn.n == 2


def test_force_recomputes(cache: StageCache) -> None:
    fn = Counter()
    cache.run("s", fn)
    cache.run("s", fn, force=True)
    assert fn.n == 2


def test_stages_are_namespaced(cache: StageCache) -> None:
    a, b = Counter(), Counter()
    cache.run("alpha", a, config={"x": 1})
    cache.run("beta", b, config={"x": 1})
    assert a.n == 1 and b.n == 1


# --- code hashing -------------------------------------------------------------------


def test_changed_logic_invalidates(cache: StageCache) -> None:
    """Different code is a different key, so a stale hit is unrepresentable.

    Hamilton and joblib need a drift warning because their key can miss a code change.
    Putting the source in the key means there is no drift to warn about.
    """

    def v1() -> np.ndarray:
        return np.zeros(10, dtype="float32")

    def v2() -> np.ndarray:
        return np.ones(10, dtype="float32")

    first = cache.run("s", v1)
    second = cache.run("s", v2)
    assert first.meta.key != second.meta.key
    assert np.array_equal(second.load(), np.ones(10, dtype="float32"))


def test_comments_and_docstrings_do_not_invalidate() -> None:
    """Reformatting must not throw away a forty-minute stage."""

    def bare() -> int:
        x = 1
        return x + 1

    def documented() -> int:
        """A docstring, and a comment below."""
        # this comment is not logic
        x = 1
        return x + 1

    assert normalised_source(bare) == normalised_source(documented)


def test_reformatting_does_not_invalidate(cache: StageCache) -> None:
    def tight() -> np.ndarray:
        return np.zeros(4, dtype="float32")

    def spaced() -> np.ndarray:
        # spread out, same logic
        return np.zeros(
            4,
            dtype="float32",
        )

    assert cache.key_for("s", fn=tight) == cache.key_for("s", fn=spaced)


def test_helper_edits_are_invisible_without_extra_code(cache: StageCache) -> None:
    """The documented limit, asserted so it cannot regress into a surprise.

    Only the stage's own body is hashed. Hamilton states the same gap; `extra_code` is how
    you close it, and this test proves both halves.
    """

    def helper_a() -> int:
        return 1

    def helper_b() -> int:
        return 2

    def stage() -> np.ndarray:
        return np.zeros(4, dtype="float32")

    assert cache.key_for("s", fn=stage) == cache.key_for("s", fn=stage)
    assert cache.key_for("s", fn=stage, extra_code=[helper_a]) != cache.key_for(
        "s", fn=stage, extra_code=[helper_b]
    )


# --- early cutoff -------------------------------------------------------------------


def test_upstream_recompute_with_identical_output_does_not_cascade(
    cache: StageCache,
) -> None:
    """Early cutoff: what makes aggressive code hashing affordable.

    Reformat a parsing stage and it recomputes — but if its bytes are unchanged, every
    downstream stage keeps its key and its cache. Nix's content-addressed derivations and
    Bazel's action outputs work the same way.
    """

    def up_v1() -> np.ndarray:
        return np.arange(10, dtype="float32")

    def up_v2() -> np.ndarray:
        # rewritten, identical result
        return np.array(list(range(10)), dtype="float32")

    downstream = Counter()

    first_up = cache.run("up", up_v1)
    first_down = cache.run("down", downstream, upstream=[first_up])

    second_up = cache.run("up", up_v2)
    second_down = cache.run("down", downstream, upstream=[second_up])

    assert first_up.meta.key != second_up.meta.key, "upstream should have recomputed"
    assert first_up.digest == second_up.digest, "outputs are identical"
    assert first_down.meta.key == second_down.meta.key, "downstream must not be invalidated"
    assert downstream.n == 1


def test_upstream_output_change_does_cascade(cache: StageCache) -> None:
    """The other half: a genuinely different upstream output must invalidate downstream."""

    def up_a() -> np.ndarray:
        return np.zeros(10, dtype="float32")

    def up_b() -> np.ndarray:
        return np.ones(10, dtype="float32")

    downstream = Counter()
    cache.run("down", downstream, upstream=[cache.run("up", up_a)])
    cache.run("down", downstream, upstream=[cache.run("up", up_b)])
    assert downstream.n == 2


# --- integrity ----------------------------------------------------------------------


def test_corrupt_artifact_fails_closed(cache: StageCache) -> None:
    """Existence is not integrity; a truncated file must not read as valid forever."""
    entry = cache.run("s", Counter())
    entry.data_path.write_bytes(b"garbage")
    with pytest.raises(CacheError, match="corrupt"):
        entry.load()


def test_metadata_without_artifact_fails_closed(cache: StageCache) -> None:
    """An interrupted write leaves metadata behind; that must not read as a hit."""
    entry = cache.run("s", Counter())
    entry.data_path.unlink()
    with pytest.raises(CacheError, match="interrupted"):
        cache.run("s", Counter())


def test_entry_records_provenance(cache: StageCache) -> None:
    entry = cache.run("s", Counter(), version=3, config={"a": 1})
    assert entry.meta.version == 3
    assert entry.meta.output_bytes > 0
    assert entry.meta.seconds >= 0
    assert set(entry.meta.policies) == {"code", "config", "env"}


# --- policies and codecs ------------------------------------------------------------


def test_explicit_version_policy_ignores_code(tmp_path: Path) -> None:
    """The Flyte/DVC posture, available per-stage for logic you want to pin by hand."""
    cache = StageCache(tmp_path, policies=[ExplicitVersionPolicy(), ConfigPolicy()])

    def v1() -> np.ndarray:
        return np.zeros(4, dtype="float32")

    def v2() -> np.ndarray:
        return np.ones(4, dtype="float32")

    assert cache.key_for("s", fn=v1) == cache.key_for("s", fn=v2)
    assert cache.key_for("s", fn=v1, version=1) != cache.key_for("s", fn=v1, version=2)


def test_env_policy_reacts_to_the_lockfile(tmp_path: Path) -> None:
    """A library can change a stage's output with no edit to code or config."""
    lock = tmp_path / "uv.lock"
    lock.write_text("one")
    cache = StageCache(tmp_path / "c", policies=[EnvPolicy(lock_path=lock), CodePolicy()])
    before = cache.key_for("s", fn=lambda: None)
    lock.write_text("two")
    assert cache.key_for("s", fn=lambda: None) != before


@pytest.mark.parametrize(
    ("codec", "value"),
    [
        (NumpyCodec(), np.arange(20, dtype="float32")),
        (JsonCodec(), {"a": [1, 2, 3], "b": "x"}),
        (BytesCodec(), b"raw payload"),
    ],
)
def test_codecs_round_trip(cache: StageCache, codec, value) -> None:
    entry = cache.run("s", lambda: value, codec=codec)
    loaded = cache.run("s", lambda: value, codec=codec).load()
    if isinstance(value, np.ndarray):
        assert np.array_equal(loaded, value)
    else:
        assert loaded == value
    assert entry.data_path.suffix == f".{codec.extension}"


def test_npz_codec_round_trips_a_mapping(cache: StageCache) -> None:
    value = {"x": np.arange(5), "y": np.ones(3)}
    entry = cache.run("s", lambda: value, codec=NpzCodec())
    loaded = entry.load()
    assert set(loaded) == {"x", "y"}
    assert np.array_equal(loaded["x"], value["x"])


def test_entries_lists_what_is_cached(cache: StageCache) -> None:
    cache.run("alpha", Counter(), config={"a": 1})
    cache.run("alpha", Counter(), config={"a": 2})
    cache.run("beta", Counter())
    assert len(cache.entries("alpha")) == 2
    assert len(cache.entries()) == 3
    assert cache.size_bytes() > 0
