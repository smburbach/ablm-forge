"""Data-config parsing helpers.

Parses the ``data.train`` dataset specification declared in :mod:`ablm.config`
into :class:`~ablm.config.TrainDatasetEntry` objects with normalized fractions.
Eval datasets are read directly as ``{name: {path, type}}`` dicts by
``ablm.train.build_eval_datasets`` (HF's Trainer owns eval cadence), so no eval
parsing lives here.
"""

from __future__ import annotations

from typing import Any

from ablm.config import TrainDatasetEntry


def parse_train_configs(raw: Any) -> list[TrainDatasetEntry]:
    """Normalize a raw ``data.train`` config value into structured dataset entries.

    Accepts:

    * ``None`` / empty string / empty mapping → empty list (no training data).
    * A single path string → one entry (``name="train"``, ``fraction=1.0``).
    * A mapping ``{name: {path: str, fraction?: float}}`` → one entry per key. A
      bare-string value (``{name: path}``) is accepted as shorthand for
      ``{path: <string>}`` with an omitted fraction.

    Fractions are normalized to sum to ``1.0``. Entries that omit ``fraction``
    split the remaining mass (``1 - sum(specified)``) equally among themselves.

    Args:
        raw: The ``data.train`` value from config — a string, mapping, or ``None``.

    Returns:
        List of :class:`~ablm.config.TrainDatasetEntry` with normalized fractions.

    Raises:
        ValueError: If a specified fraction is negative, the resolved fractions do
            not sum to a positive value, an entry is missing ``path``, or the raw
            value is neither a string nor a mapping.
    """
    if raw is None:
        return []

    if isinstance(raw, str):
        path = raw.strip()
        if not path:
            return []
        return [TrainDatasetEntry(name="train", path=path, fraction=1.0)]

    if isinstance(raw, dict):
        return _parse_train_mapping(raw)

    raise ValueError(
        f"data.train must be a path string or a {{name: {{path, fraction}}}} "
        f"mapping, got {type(raw).__name__}"
    )


def _parse_train_mapping(raw: dict[Any, Any]) -> list[TrainDatasetEntry]:
    """Parse the ``{name: {path, fraction?}}`` form into normalized entries."""
    if not raw:
        return []

    names: list[str] = []
    paths: list[str] = []
    # None marks an omitted fraction (resolved by splitting the remainder).
    fractions: list[float | None] = []
    for name, value in raw.items():
        if value is None:
            continue
        if isinstance(value, str):
            path: Any = value
            frac: Any = None
        elif isinstance(value, dict):
            path = value.get("path")
            frac = value.get("fraction")
        else:
            raise ValueError(
                f"data.train.{name}: expected a path string or a "
                f"{{path, fraction}} mapping, got {type(value).__name__}"
            )

        if path is None or (isinstance(path, str) and not path.strip()):
            raise ValueError(f"data.train.{name} is missing required 'path'")

        names.append(str(name))
        paths.append(str(path).strip())
        fractions.append(float(frac) if frac is not None else None)

    if not names:
        return []

    for name, frac in zip(names, fractions, strict=True):
        if frac is not None and frac < 0:
            raise ValueError(f"data.train.{name}.fraction must be >= 0, got {frac}")

    specified_sum = sum(f for f in fractions if f is not None)
    n_omitted = sum(1 for f in fractions if f is None)
    if n_omitted:
        # Omitted entries share whatever mass the specified ones leave behind.
        share = max(0.0, 1.0 - specified_sum) / n_omitted
        resolved = [share if f is None else f for f in fractions]
    else:
        resolved = [f for f in fractions if f is not None]

    total = sum(resolved)
    if total <= 0:
        raise ValueError(f"data.train fractions must sum to > 0, got {total}")

    return [
        TrainDatasetEntry(name=name, path=path, fraction=frac / total)
        for name, path, frac in zip(names, paths, resolved, strict=True)
    ]
