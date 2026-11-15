"""End-to-end simulate -> register -> reconstruct -> evaluate (Phase 3)."""

from __future__ import annotations

from .types import Capture, Registration

__all__ = ["run_pipeline"]


def run_pipeline(capture: Capture, registration: Registration, method: str, scale: int = 2) -> dict:
    """Reconstruction, shifts, metrics vs truth and baselines, timings, effective gain."""
    raise NotImplementedError("Phase 3")
