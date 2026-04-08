"""State transition module for orbital/target-based travel time calculations."""

from .coverage import update_coverage_map, CoverageUpdateResult
from .orchestrator import orchestrate_step, StepResult
from .reward import calculate_reward, RewardResult
from .state_builder import build_state, State
from .sun_position import calculate_sun_position
from .travel_time import (
    TargetOrbitConfig,
    OrbitalConfig,
    angular_distance,
    get_travel_time,
    advance_time,
    compute_all_travel_times,
)
from .visibility import get_lit_visible_points

__all__ = [
    "TargetOrbitConfig",
    "OrbitalConfig",
    "angular_distance",
    "get_travel_time",
    "advance_time",
    "compute_all_travel_times",
    "calculate_sun_position",
    "get_lit_visible_points",
    "update_coverage_map",
    "CoverageUpdateResult",
    "calculate_reward",
    "RewardResult",
    "build_state",
    "State",
    "orchestrate_step",
    "StepResult",
]
