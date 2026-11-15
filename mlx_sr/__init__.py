"""Simulation prototype: MLX90640 multi-frame super-resolution from small motion."""

from .types import *  # noqa: F401,F403
from .types import __all__ as _types_all
from .scene import make_scene
from .motion import angles_to_shift, build_motion, simulate_gyro, trace_from_shifts
from .sensor import merge_subpages, simulate_capture, truth_at_scale

__all__ = list(_types_all) + [
    "make_scene", "build_motion", "simulate_gyro", "trace_from_shifts", "angles_to_shift",
    "simulate_capture", "truth_at_scale", "merge_subpages",
]
__version__ = "0.1.0"
