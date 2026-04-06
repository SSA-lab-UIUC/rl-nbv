"""State transition module for orbital/target-based travel time calculations."""

from .travel_time import (
    TargetOrbitConfig,
    OrbitalConfig,
    angular_distance,
    get_travel_time,
    advance_time,
    compute_all_travel_times,
)

__all__ = [
    "TargetOrbitConfig",
    "OrbitalConfig",
    "angular_distance",
    "get_travel_time",
    "advance_time",
    "compute_all_travel_times",
]
