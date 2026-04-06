"""
Travel time calculation for movement between viewpoints on a unit sphere.

Uses constant angular velocity model for a camera/observer rotating around a target object.
Points are assumed to be normalized to unit sphere surface in dimensionless units.
All parameters are in abstract units (not kilometers or meters).
"""

import numpy as np
from typing import Union, Tuple, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class TargetOrbitConfig:
    """
    Orbital mission parameters for rotating around a target object.
    
    All parameters work in dimensionless units (no specific unit conversion).
    Useful for unit sphere coordinates and abstract orbital mechanics simulations.
    """

    def __init__(
        self,
        orbit_radius: float = 1.0,
        grav_param: float = 1.0,
        num_orbits: float = 2.0,
    ):
        """
        Initialize target orbit configuration.

        Args:
            orbit_radius: Orbital radius from target center (dimensionless units).
                         The camera rotates at this radius around the target.
                         Default: 1.0 (unit sphere).
            grav_param: Gravitational parameter (dimensionless units). 
                       This controls orbital dynamics and travel time scaling.
                       Default: 1.0.
            num_orbits: Number of complete orbits for mission horizon.
                       Default: 2.0.
        """
        self.orbit_radius = orbit_radius
        self.grav_param = grav_param
        self.num_orbits = num_orbits

        # Mean motion (angular velocity of circular orbit)
        # n = sqrt(grav_param / orbit_radius^3)
        self.mean_motion = np.sqrt(grav_param / (orbit_radius**3))

        # Single orbital period (dimensionless time units)
        # P = 2π / n
        self.orbital_period = 2.0 * np.pi / self.mean_motion

        # Total mission time (dimensionless time units)
        # T_total = num_orbits * P
        self.total_time = num_orbits * self.orbital_period

        # Angular velocity for unit sphere traversal (dimensionless)
        # omega = 2π / T_total
        self.angular_velocity = 2.0 * np.pi / self.total_time

        logger.info(
            f"TargetOrbitConfig: r_orbit={self.orbit_radius:.4f}, "
            f"μ={self.grav_param:.4f}, "
            f"P_orbit={self.orbital_period:.4f}, "
            f"T_total={self.total_time:.4f}, "
            f"ω={self.angular_velocity:.6f}"
        )


# Backward compatibility alias
OrbitalConfig = TargetOrbitConfig


def angular_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Calculate angular distance (great circle distance on unit sphere).

    Args:
        p1: Point 1, shape (3,), must be normalized or unit vector.
        p2: Point 2, shape (3,), must be normalized or unit vector.

    Returns:
        Angular distance in radians [0, π].
    """
    p1_norm = p1 / np.linalg.norm(p1)
    p2_norm = p2 / np.linalg.norm(p2)

    # Clamp dot product to [-1, 1] to avoid numerical errors in arccos
    dot_product = np.clip(np.dot(p1_norm, p2_norm), -1.0, 1.0)
    angle = np.arccos(dot_product)

    return angle


def get_travel_time(
    current_view_point: np.ndarray,
    target_view_point: np.ndarray,
    orbit_config: TargetOrbitConfig,
    travel_times_cache: Optional[np.ndarray] = None,
) -> float:
    """
    Calculate travel time between two viewpoints on unit sphere.

    Assumes constant angular velocity motion around the target object.
    Supports both precomputed lookup and dynamic calculation.

    Args:
        current_view_point: Current position, shape (3,), in dimensionless units.
        target_view_point: Target position, shape (3,), in dimensionless units.
        orbit_config: Target orbit configuration (mission parameters).
        travel_times_cache: Optional precomputed travel times matrix, shape (n_views, n_views).
                           If provided and indices available, uses lookup instead of calculation.

    Returns:
        Travel time in dimensionless time units.

    Raises:
        ValueError: If inputs are invalid or points are not on unit sphere.
    """
    # Validate inputs
    if current_view_point.shape != (3,):
        raise ValueError(f"current_view_point must have shape (3,), got {current_view_point.shape}")
    if target_view_point.shape != (3,):
        raise ValueError(f"target_view_point must have shape (3,), got {target_view_point.shape}")

    # Check if points are on unit sphere (with tolerance)
    curr_norm = np.linalg.norm(current_view_point)
    targ_norm = np.linalg.norm(target_view_point)
    if not (0.95 < curr_norm < 1.05):
        logger.warning(f"current_view_point norm = {curr_norm:.4f}, expected ≈1.0")
    if not (0.95 < targ_norm < 1.05):
        logger.warning(f"target_view_point norm = {targ_norm:.4f}, expected ≈1.0")

    # Calculate angular distance
    angle = angular_distance(current_view_point, target_view_point)

    # Travel time = angular distance / angular velocity
    travel_time = angle / orbit_config.angular_velocity

    return travel_time


def advance_time(
    current_time: float,
    travel_time: float,
    total_mission_time: Optional[float] = None,
    wrap_around: bool = False,
) -> float:
    """
    Calculate new absolute time after traveling for travel_time.

    Args:
        current_time: Current absolute timestamp (seconds).
        travel_time: Time to advance (seconds).
        total_mission_time: Total mission horizon. If provided, constrains result to [0, total_mission_time].
                           If None, time can exceed total_mission_time.
        wrap_around: If True and result exceeds total_mission_time, wraps time around.
                    If False, clamps to total_mission_time.

    Returns:
        New absolute time.

    Raises:
        ValueError: If inputs are invalid.
    """
    if current_time < 0:
        raise ValueError(f"current_time must be >= 0, got {current_time}")
    if travel_time < 0:
        raise ValueError(f"travel_time must be >= 0, got {travel_time}")

    new_time = current_time + travel_time

    # Honor mission time constraint if provided
    if total_mission_time is not None:
        if wrap_around and new_time > total_mission_time:
            new_time = new_time % total_mission_time
        elif new_time > total_mission_time:
            new_time = total_mission_time

    return new_time


def compute_all_travel_times(
    viewpoints: np.ndarray,
    orbit_config: TargetOrbitConfig,
) -> np.ndarray:
    """
    Precompute travel times between all pairs of viewpoints.

    Args:
        viewpoints: Array of viewpoints, shape (n_views, 3), in dimensionless units.
        orbit_config: Target orbit configuration (mission parameters).

    Returns:
        Travel times matrix, shape (n_views, n_views), where element [i, j]
        is travel time (in dimensionless time units) from viewpoint i to viewpoint j.
    """
    n_views = viewpoints.shape[0]
    travel_times = np.zeros((n_views, n_views), dtype=np.float32)

    for i in range(n_views):
        for j in range(n_views):
            travel_times[i, j] = get_travel_time(
                viewpoints[i], viewpoints[j], orbit_config
            )

    return travel_times


if __name__ == "__main__":
    # Example: Load viewpoints and compute travel times
    logging.basicConfig(level=logging.INFO)

    # Load example viewpoints
    viewpoints_path = "output/acrimsat_final/viewpoints_33.txt"
    viewpoints = np.loadtxt(viewpoints_path)

    print(f"Loaded {viewpoints.shape[0]} viewpoints")
    print(f"Viewpoint shape: {viewpoints.shape}")
    print(f"\nFirst 3 viewpoints:\n{viewpoints[:3]}")

    # Initialize orbit config (default: unit sphere, mu=1, 2 orbits)
    config = TargetOrbitConfig()

    print(f"\nTarget Orbit Parameters:")
    print(f"  Orbit radius: {config.orbit_radius:.4f}")
    print(f"  Gravitational parameter (μ): {config.grav_param:.6f}")
    print(f"  Orbital period: {config.orbital_period:.4f}")
    print(f"  Total mission time: {config.total_time:.4f}")
    print(f"  Angular velocity: {config.angular_velocity:.6f}")

    # Example: travel time between first two viewpoints
    p1 = viewpoints[0]
    p2 = viewpoints[1]
    dist = angular_distance(p1, p2)
    t_travel = get_travel_time(p1, p2, config)

    print(f"\nTravel from viewpoint 0 to viewpoint 1:")
    print(f"  Angular distance: {dist:.6f} rad ({np.degrees(dist):.2f}°)")
    print(f"  Travel time: {t_travel:.4f} (dimensionless)")

    # Compute full travel times matrix
    print(f"\nComputing travel times for all {viewpoints.shape[0]} viewpoints...")
    travel_times = compute_all_travel_times(viewpoints, config)

    print(f"Travel times matrix shape: {travel_times.shape}")
    print(f"Min travel time: {travel_times[travel_times > 0].min():.4f}")
    print(f"Max travel time: {travel_times.max():.4f}")
    print(f"Mean travel time: {travel_times[travel_times > 0].mean():.4f}")

    # Time advancement example
    current_time = 0.0
    travel_time = t_travel
    new_time = advance_time(current_time, travel_time, config.total_time)
    print(f"\nTime advancement:")
    print(f"  Current time: {current_time:.4f}")
    print(f"  Travel time: {travel_time:.4f}")
    print(f"  New time: {new_time:.4f}")
    print(f"  Total mission time: {config.total_time:.4f}")
    print(f"  Remaining time: {config.total_time - new_time:.4f}")
