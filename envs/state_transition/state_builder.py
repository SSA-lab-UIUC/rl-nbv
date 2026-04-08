"""Next state builder for state transition.

Implements a pure builder that creates next_state from updated transition outputs.
State must carry absolute timestamp and current sun_position for time-dependent dynamics.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np


class State(TypedDict, total=False):
    """State dictionary type definition.
    
    Required fields:
        _current_view: Current selected view index (int).
        _current_time: Absolute timestamp (float, seconds).
        sun_position: Sun direction vector, shape (3,).
        coverage_map: Cumulative coverage boolean mask, shape (N,).
        views: Boolean mask of visited views, shape (V,).
        travel_times: Travel times from current view, shape (V,).
    
    Optional fields can be added for environment-specific state extensions.
    """
    
    _current_view: int
    _current_time: float
    sun_position: np.ndarray
    coverage_map: np.ndarray
    views: np.ndarray
    travel_times: np.ndarray


def _as_int_scalar(name: str, value: object) -> int:
    """Normalize input to integer scalar.
    
    Args:
        name: Parameter name for error messages.
        value: Input value.
    
    Returns:
        Integer value.
    
    Raises:
        TypeError: If not convertible to integer.
    """
    if isinstance(value, bool):
        raise TypeError(f"{name} must be int, not bool")
    try:
        scalar = int(value)
    except (TypeError, ValueError) as e:
        raise TypeError(f"{name} must be convertible to int: {e}")
    return scalar


def _as_float_scalar(name: str, value: object) -> float:
    """Normalize input to float scalar.
    
    Args:
        name: Parameter name for error messages.
        value: Input value.
    
    Returns:
        Float value.
    
    Raises:
        TypeError: If not convertible to float.
        ValueError: If not finite.
    """
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric, not bool")
    try:
        scalar = float(value)
    except (TypeError, ValueError) as e:
        raise TypeError(f"{name} must be convertible to float: {e}")
    
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite, got {scalar}")
    
    return scalar


def _as_vector3(name: str, value: object) -> np.ndarray:
    """Normalize input to 3D vector.
    
    Args:
        name: Parameter name for error messages.
        value: Input value (array-like with shape (3,)).
    
    Returns:
        Numpy array with shape (3,) and dtype float.
    
    Raises:
        ValueError: If shape is wrong or contains non-finite values.
    """
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain finite values")
    return vector


def _as_bool_array(name: str, value: object) -> np.ndarray:
    """Normalize input to boolean array.
    
    Args:
        name: Parameter name for error messages.
        value: Input value (array-like).
    
    Returns:
        Boolean numpy array with dtype bool.
    
    Raises:
        ValueError: If input cannot be normalized.
    """
    arr = np.asarray(value, dtype=bool)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D array, got shape {arr.shape}")
    return arr


def _as_float_array(name: str, value: object) -> np.ndarray:
    """Normalize input to float array.
    
    Args:
        name: Parameter name for error messages.
        value: Input value (array-like).
    
    Returns:
        Float numpy array with dtype float.
    
    Raises:
        ValueError: If input cannot be normalized or contains non-finite values.
    """
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D array, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite values")
    return arr


def build_state(
    action: int,
    new_time: float,
    new_sun_position: object,
    new_coverage_map: object,
    prev_views: object,
    travel_times: object,
) -> State:
    """Build next state from updated transition outputs.
    
    Creates an immutable-friendly state dictionary with all required fields.
    Marks the chosen action as visited and updates all time-dependent state.
    
    Args:
        action: Selected view/action index (int).
        new_time: New absolute timestamp (float, seconds).
        new_sun_position: New sun direction vector, shape (3,).
        new_coverage_map: Updated coverage map, shape (N,), boolean mask.
        prev_views: Previous visit flags, shape (V,), boolean mask.
        travel_times: Travel times from new current view to all views, shape (V,).
    
    Returns:
        State dict with fields:
        - _current_view: action (int)
        - _current_time: new_time (float)
        - sun_position: new_sun_position (np.ndarray, shape (3,))
        - coverage_map: new_coverage_map (np.ndarray, shape (N,), dtype bool)
        - views: updated views with action marked as visited (np.ndarray, shape (V,), dtype bool)
        - travel_times: travel_times (np.ndarray, shape (V,), dtype float)
    
    Raises:
        TypeError: If inputs have wrong type.
        ValueError: If inputs are out of valid range or inconsistent shapes.
        IndexError: If action is out of range for views array.
    
    Notes:
        - This function does NOT mutate prev_views; it creates a new views array.
        - travel_times should be precomputed relative to the new current view and
          new absolute timestamp (if time-dependent).
        - State fields use underscore prefix (_current_view, _current_time) to
          distinguish internal state from observation features.
    """
    
    # Validate and normalize inputs
    action = _as_int_scalar("action", action)
    new_time = _as_float_scalar("new_time", new_time)
    new_sun_position = _as_vector3("new_sun_position", new_sun_position)
    new_coverage_map = _as_bool_array("new_coverage_map", new_coverage_map)
    prev_views_arr = _as_bool_array("prev_views", prev_views)
    travel_times_arr = _as_float_array("travel_times", travel_times)
    
    # Validate action is in range
    n_views = prev_views_arr.shape[0]
    if action < 0 or action >= n_views:
        raise IndexError(
            f"action {action} out of range for views array with {n_views} views"
        )
    
    # Validate travel_times shape matches views
    if travel_times_arr.shape[0] != n_views:
        raise ValueError(
            f"travel_times shape {travel_times_arr.shape} does not match views shape {prev_views_arr.shape}"
        )
    
    # Validate time is non-negative
    if new_time < 0.0:
        raise ValueError(f"new_time must be >= 0, got {new_time}")
    
    # Create new views array with action marked as visited
    # IMPORTANT: Create a copy to avoid mutating prev_views
    new_views = prev_views_arr.copy()
    new_views[action] = True
    
    # Build state dictionary
    state: State = {
        "_current_view": action,
        "_current_time": new_time,
        "sun_position": new_sun_position.copy(),  # Copy to ensure immutability
        "coverage_map": new_coverage_map.copy(),  # Copy to ensure immutability
        "views": new_views,
        "travel_times": travel_times_arr.copy(),  # Copy to ensure immutability
    }
    
    return state
