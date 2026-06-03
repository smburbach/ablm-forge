"""Tests for the training-dataset spec parser (`parse_train_configs`)."""

from __future__ import annotations

import pytest

from ablm.data.config import parse_train_configs

# --------------------------------------------------------------------------- #
# parse_train_configs
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("raw", [None, "", "   ", {}])
def test_parse_train_empty_inputs(raw: object) -> None:
    """``None``/empty string/empty mapping all yield no training entries."""
    assert parse_train_configs(raw) == []


def test_parse_train_single_path_string() -> None:
    """A single path string expands to one full-weight entry named ``train``."""
    entries = parse_train_configs("/data/uniref50/")
    assert len(entries) == 1
    only = entries[0]
    assert only.name == "train"
    assert only.path == "/data/uniref50/"
    assert only.fraction == pytest.approx(1.0)


def test_parse_train_single_mapping_entry_is_full_weight() -> None:
    """A single mapping entry normalizes to fraction 1.0 even if unspecified."""
    entries = parse_train_configs({"uniref50": {"path": "/data/uniref50/"}})
    assert len(entries) == 1
    assert entries[0].name == "uniref50"
    assert entries[0].path == "/data/uniref50/"
    assert entries[0].fraction == pytest.approx(1.0)


def test_parse_train_fractions_normalize_to_one() -> None:
    """Specified fractions are renormalized so the total is 1.0."""
    entries = parse_train_configs(
        {
            "uniref50": {"path": "/data/uniref50/", "fraction": 0.8},
            "bfd": {"path": "/data/bfd/", "fraction": 0.4},
        }
    )
    fractions = {e.name: e.fraction for e in entries}
    assert sum(fractions.values()) == pytest.approx(1.0)
    # The 2:1 input ratio is preserved after normalization.
    assert fractions["uniref50"] == pytest.approx(2 / 3)
    assert fractions["bfd"] == pytest.approx(1 / 3)


def test_parse_train_omitted_fractions_split_remainder() -> None:
    """Entries lacking ``fraction`` split the remaining mass equally."""
    entries = parse_train_configs(
        {
            "a": {"path": "/a", "fraction": 0.6},
            "b": {"path": "/b"},
            "c": {"path": "/c"},
        }
    )
    fractions = {e.name: e.fraction for e in entries}
    assert sum(fractions.values()) == pytest.approx(1.0)
    assert fractions["a"] == pytest.approx(0.6)
    assert fractions["b"] == pytest.approx(0.2)
    assert fractions["c"] == pytest.approx(0.2)


def test_parse_train_all_omitted_is_uniform() -> None:
    """When no fractions are given, mass is split evenly across all entries."""
    entries = parse_train_configs({"a": {"path": "/a"}, "b": {"path": "/b"}})
    for e in entries:
        assert e.fraction == pytest.approx(0.5)


def test_parse_train_bare_string_value_shorthand() -> None:
    """A bare-string mapping value is treated as the dataset path."""
    entries = parse_train_configs({"a": "/a", "b": "/b"})
    assert {e.name: e.path for e in entries} == {"a": "/a", "b": "/b"}
    assert all(e.fraction == pytest.approx(0.5) for e in entries)


def test_parse_train_preserves_order() -> None:
    """Entry order follows the mapping's insertion order."""
    entries = parse_train_configs(
        {"first": {"path": "/1"}, "second": {"path": "/2"}, "third": {"path": "/3"}}
    )
    assert [e.name for e in entries] == ["first", "second", "third"]


def test_parse_train_missing_path_raises() -> None:
    """An entry mapping without ``path`` is an error."""
    with pytest.raises(ValueError, match="missing required 'path'"):
        parse_train_configs({"a": {"fraction": 0.5}})


def test_parse_train_negative_fraction_raises() -> None:
    """A negative fraction is rejected."""
    with pytest.raises(ValueError, match="must be >= 0"):
        parse_train_configs(
            {"a": {"path": "/a", "fraction": -0.1}, "b": {"path": "/b", "fraction": 0.5}}
        )


def test_parse_train_zero_total_raises() -> None:
    """Fractions that sum to zero cannot be normalized."""
    with pytest.raises(ValueError, match="must sum to > 0"):
        parse_train_configs({"a": {"path": "/a", "fraction": 0.0}})


@pytest.mark.parametrize("raw", [42, 3.14, ["/a", "/b"]])
def test_parse_train_invalid_type_raises(raw: object) -> None:
    """A train config that is neither a string nor a mapping is rejected."""
    with pytest.raises(ValueError, match="must be a path string or a"):
        parse_train_configs(raw)
