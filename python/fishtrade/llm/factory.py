"""Resolve role -> Ark model id (each role has its own endpoint, with fallback)."""

from __future__ import annotations

from ..config.settings import settings


def resolve_model_id(role: str = "default") -> str:
    """Map ``role`` to a concrete ``ARK_MODEL_ID_*``.

    Roles: ``research`` / ``debate`` / ``judge`` / ``default``. Any role
    without a configured override falls back to ``ARK_MODEL_ID``.
    """
    return settings.resolve_model_id(role)
