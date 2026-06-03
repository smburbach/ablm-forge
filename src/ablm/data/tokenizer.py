"""Tokenizer access layer.

A thin, cached accessor over :class:`ablm.model.AblmTokenizerFast` (the single
source of truth for the vocabulary). Defines no vocabulary of its own.
"""

from __future__ import annotations

from functools import cache

from ablm.model import AblmTokenizerFast

__all__ = ["get_tokenizer"]


@cache
def get_tokenizer() -> AblmTokenizerFast:
    """Return the shared canonical tokenizer.

    Cached as a module-level singleton. The instance is treated as read-only;
    callers must not mutate it.
    """
    return AblmTokenizerFast()
