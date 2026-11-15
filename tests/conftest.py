"""Shared, seeded fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from mlx_sr.scene import make_scene
from mlx_sr.types import SensorConfig


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


@pytest.fixture(scope="session")
def chart():
    return make_scene("chart")


@pytest.fixture(scope="session")
def textured():
    return make_scene("textured", contrast_k=2.0, seed=7)


@pytest.fixture(scope="session")
def flat():
    return make_scene("flat")


@pytest.fixture
def clean_bab() -> SensorConfig:
    """BAB sensor with every non-ideality switched off."""
    return SensorConfig.bab(netd_1hz_k=0.0, fpn_sigma_k=0.0, n_dead=0, psf_sigma_px=0.0, adc_lsb_k=0.0)
