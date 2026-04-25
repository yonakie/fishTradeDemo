"""Pytest fixtures shared across the suite."""

from __future__ import annotations

import pytest

from fishtrade.models import IndicatorScore


def make_indicator(
    name: str = "PE_RATIO",
    score: int = 1,
    raw_value: float | str | None = 12.5,
    is_degraded: bool = False,
    degrade_reason: str | None = None,
) -> IndicatorScore:
    """Helper that builds a valid IndicatorScore with sensible defaults."""
    return IndicatorScore(
        name=name,
        display_name_zh=name,
        display_name_en=name,
        raw_value=raw_value,
        score=score,  # type: ignore[arg-type]
        reasoning=f"reasoning for {name}",
        is_degraded=is_degraded,
        degrade_reason=degrade_reason,
    )


def make_indicator_list(scores: list[int]) -> list[IndicatorScore]:
    """Build exactly len(scores) IndicatorScore items, indexing names i0..i9."""
    return [make_indicator(name=f"IND_{i}", score=s) for i, s in enumerate(scores)]


@pytest.fixture
def indicator_factory():
    return make_indicator


@pytest.fixture
def indicator_list_factory():
    return make_indicator_list
