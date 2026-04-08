"""Step orchestrator for state transition.

Implements ordered transition orchestration in step(action) by composing
helper functions for travel time, sun position, visibility, coverage, reward,
and next state building.

This module provides the core step logic that can be integrated into the
environment's step() entry point.
"""

from __future__ import annotations

from typing import NamedTuple
from collections.abc import Mapping

import numpy as np

from .travel_time import get_travel_time, advance_time, TargetOrbitConfig
from .sun_position import calculate_sun_position
from .visibility import get_lit_visible_points
from .coverage import update_coverage_map, CoverageUpdateResult
from .reward import calculate_reward, RewardResult
from .state_builder import build_state, State


class StepResult(NamedTuple):
    """Result of step orchestration.
    
    Attributes:
        next_state: Updated state dictionary with all transition outputs.
        reward: Scalar reward value.
        reward_breakdown: Dict with reward components for debugging.
        coverage_update: Coverage update result with newly covered count/ratio.
        travel_time: Travel time used in this step (float, seconds).
    """
    
    next_state: State
    reward: float
    reward_breakdown: dict[str, float]
    coverage_update: CoverageUpdateResult
    travel_time: float


def _as_int_scalar(name: str, value: object) -> int:
    """Normalize input to integer scalar."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be int, not bool")
    try:
        return int(value)
    except (TypeError, ValueError) as e:
        raise TypeError(f"{name} must be convertible to int: {e}")


def orchestrate_step(
    action: int,
    current_state: State,
    orbit_config: TargetOrbitConfig,
    geometry_cache: Mapping[str, object],
    orbital_params: Mapping[str, object],
    reward_config: Mapping[str, object] | None = None,
    total_points: int | None = None,
) -> StepResult:
    """Orchestrate state transition step with ordered helper function calls.
    
    This function implements the complete step logic:
    1. Get travel time from current view to target view
    2. Advance mission time by travel time
    3. Calculate new sun position based on new time
    4. Get lit-visible points using sun position and geometry
    5. Update coverage map with new lit-visible points
    6. Calculate reward based on coverage gain and visit status
    7. Build next state with updated fields
    
    Args:
        action: Selected view/action index (int).
        current_state: Current state dict with fields:
            - _current_view (int): current view index
            - _current_time (float): current absolute timestamp
            - sun_position (np.ndarray): current sun direction, shape (3,)
            - coverage_map (np.ndarray): current coverage mask, shape (N,)
            - views (np.ndarray): visited view flags, shape (V,)
            - travel_times (np.ndarray): travel times from current view, shape (V,)
        orbit_config: TargetOrbitConfig with mission parameters.
        geometry_cache: Geometry data and optional caches. Required keys:
            - canonical_points or point_count
            - surface_normals
            - view_positions or sensor_position
            Optional keys for visibility/illumination caching and parameters.
        orbital_params: Sun dynamics parameters. Required keys:
            - angular_velocity_rad_per_s or period_s
            Optional keys: initial_phase_rad, time_offset_s, action_phase_offsets_rad.
        reward_config: Optional reward configuration dict.
            - revisit_penalty (float, default -5.0)
            - coverage_coeff (float, default 1.0)
            - shaping_terms (dict, optional)
        total_points: Optional total canonical point count (for coverage validation).
    
    Returns:
        StepResult with:
        - next_state: Updated state dict
        - reward: Scalar reward value
        - reward_breakdown: Dict with reward components
        - coverage_update: CoverageUpdateResult with coverage details
        - travel_time: Travel time used
    
    Raises:
        TypeError: If inputs have wrong type.
        ValueError: If inputs are out of valid range.
        IndexError: If action is out of range.
    
    Notes:
        - All helper functions are side-effect free (no state mutation).
        - Integration tests should verify episode progression and termination.
        - Termination conditions (max_steps, coverage threshold) are handled
          by the calling environment, not by this orchestrator.
    """
    
    # Validate action
    action = _as_int_scalar("action", action)
    
    # Extract current state fields
    current_view = current_state["_current_view"]
    current_time = current_state["_current_time"]
    current_sun_position = current_state["sun_position"]
    prev_coverage_map = current_state["coverage_map"]
    prev_views = current_state["views"]
    travel_times_from_current = current_state["travel_times"]
    
    # Validate action is in range
    n_views = prev_views.shape[0]
    if action < 0 or action >= n_views:
        raise IndexError(
            f"action {action} out of range for views with {n_views} views"
        )
    
    # ============================================================================
    # STEP 1: GET TRAVEL TIME
    # ============================================================================
    # Use precomputed travel times from current state if available,
    # otherwise calculate from viewpoints in geometry cache.
    travel_time = float(travel_times_from_current[action])
    
    # ============================================================================
    # STEP 2: ADVANCE TIME
    # ============================================================================
    new_time = advance_time(
        current_time=current_time,
        travel_time=travel_time,
        total_mission_time=orbit_config.total_time,
        wrap_around=False,
    )
    
    # ============================================================================
    # STEP 3: CALCULATE SUN POSITION
    # ============================================================================
    new_sun_position = calculate_sun_position(
        action=action,
        new_time=new_time,
        prev_sun_position=current_sun_position,
        orbital_params=orbital_params,
    )
    
    # ============================================================================
    # STEP 4: GET LIT-VISIBLE POINTS
    # ============================================================================
    visible_lit_points = get_lit_visible_points(
        action=action,
        new_sun_position=new_sun_position,
        geometry_cache=geometry_cache,
    )
    
    # ============================================================================
    # STEP 5: UPDATE COVERAGE MAP
    # ============================================================================
    coverage_update = update_coverage_map(
        prev_coverage_map=prev_coverage_map,
        visible_lit_points=visible_lit_points,
        total_points=total_points,
    )
    
    # ============================================================================
    # STEP 6: CALCULATE REWARD
    # ============================================================================
    reward_result = calculate_reward(
        action=action,
        views=prev_views,
        newly_covered_ratio=coverage_update.newly_covered_ratio,
        travel_time=travel_time,
        reward_config=reward_config,
    )
    
    # ============================================================================
    # STEP 7: BUILD NEXT STATE
    # ============================================================================
    # Get travel times from the new current view (action)
    # For now, use the full travel_times matrix lookup if available in geometry_cache,
    # otherwise reuse the existing travel_times (suboptimal but functional).
    # In full integration, this should be precomputed per-view or cached.
    if "travel_times_matrix" in geometry_cache:
        # Use precomputed travel times from new view
        travel_times_matrix = np.asarray(geometry_cache["travel_times_matrix"])
        if travel_times_matrix.ndim != 2:
            raise ValueError(
                "geometry_cache['travel_times_matrix'] must have shape (V, V)"
            )
        if action >= travel_times_matrix.shape[0]:
            raise IndexError(
                f"action {action} out of range for travel_times_matrix with {travel_times_matrix.shape[0]} views"
            )
        new_travel_times = travel_times_matrix[action]
    else:
        # Fallback: reuse current travel_times (not ideal, but maintains compatibility)
        # In production, the environment should provide travel_times_matrix in geometry_cache
        new_travel_times = travel_times_from_current
    
    next_state = build_state(
        action=action,
        new_time=new_time,
        new_sun_position=new_sun_position,
        new_coverage_map=coverage_update.coverage_map,
        prev_views=prev_views,
        travel_times=new_travel_times,
    )
    
    return StepResult(
        next_state=next_state,
        reward=reward_result.reward,
        reward_breakdown=reward_result.breakdown,
        coverage_update=coverage_update,
        travel_time=travel_time,
    )
