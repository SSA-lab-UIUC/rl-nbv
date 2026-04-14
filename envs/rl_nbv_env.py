"""
RL-NBV Environment  –  Level 2  (Fuel-Aware)
=============================================

What is Level 2?
----------------
Level 1 selected viewpoints to maximise coverage while minimising travel time.
Level 2 goes further: it also tracks *fuel consumption* (Δv) and enforces a
hard fuel budget of ~50 m/s (dimensionless units here).

The key change is in the reward function:

    reward  =  coverage_reward  −  γ_time · travel_time
                                 −  γ_fuel · Δv_k

where Δv_k is the fuel cost (computed via Clohessy-Wiltshire equations) for
the manoeuvre from the current viewpoint to the next one.

The agent therefore learns:
  1. Which viewpoints give the most new coverage.
  2. How to plan routes that are both time-efficient AND fuel-efficient.
  3. How to complete a full inspection before the fuel budget runs out.

New features vs Level 1
------------------------
  * delta_v_matrix  : pre-computed (33×33) table of Δv costs.
  * fuel_budget     : total Δv budget for the episode (default 50 units).
  * cumulative_dv   : fuel spent so far in the current episode.
  * fuel_exhausted  : episode terminates early if budget is exceeded.
  * delta_v_weight  : scaling factor γ in the reward formula.
"""

from typing import Optional
import numpy as np
import math
import gym
from gym import spaces
import envs.shapenet_reader as shapenet_reader
import random
import torch
import sys
import os
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from distance.chamfer_distance import ChamferDistanceFunction
from envs.state_transition import (
    TargetOrbitConfig,
    get_travel_time,
    compute_all_travel_times,
)

# Import our new CW utilities
from envs.state_transition.cw_utils import CWDynamics, compute_delta_v_matrix

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# =============================================================================
# Helper functions (unchanged from Level 1)
# =============================================================================

def resample_pcd(pcd: np.ndarray, n: int, logger, name: str) -> np.ndarray:
    """
    Resize a point cloud to exactly n points by sampling or duplicating.
    Needed to keep observation tensors a fixed shape for the neural network.
    """
    if pcd.shape[0] == 0:
        logger.debug(f"Point cloud is empty for model: {name}")
        return np.zeros((n, 3))
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        # Duplicate random points if we don't have enough
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size=n - pcd.shape[0])])
    logger.debug(f"Resample {pcd.shape[0]} → {n} points  (model: {name})")
    return pcd[idx[:n]]


def normalize_pc(points: np.ndarray, logger, name: str) -> np.ndarray:
    """
    Centre a point cloud at the origin and scale so the furthest point is at
    distance 1.  This makes features scale-invariant for the neural network.
    """
    if points.shape[0] == 0:
        logger.debug(f"normalize received empty cloud for model: {name}")
        return points
    centroid = np.mean(points, axis=0)
    points -= centroid
    furthest = np.max(np.sqrt(np.sum(points ** 2, axis=-1)))
    if furthest == 0:
        logger.debug(f"normalize skipped (zero extent) for model: {name}")
        return points
    points /= furthest
    return points


# =============================================================================
# Level 2 Environment
# =============================================================================

class PointCloudNBVEnvLevel2(gym.Env):
    """
    Fuel-aware Next-Best-View environment  (RL-NBV Level 2).

    The agent selects viewpoints from a fixed set of 33 positions on a unit
    sphere around the target object.  At every step it receives a reward that
    balances three competing objectives:

      +  New coverage (positive)
      -  Travel time cost (negative)
      -  Fuel (Δv) cost (negative)

    The episode ends when any of these conditions is met:
      * Coverage ≥ terminated_coverage threshold
      * Steps > max_step
      * Cumulative Δv > fuel_budget  (new in Level 2)

    Parameters
    ----------
    data_path : str
        Root directory of ShapeNet point cloud data.
    view_num : int
        Number of candidate viewpoints (default 33).
    begin_view : int
        Starting viewpoint index.  -1 means random.
    observation_space_dim : int
        Number of points in each observation cloud.  -1 = no resampling (debug).
    terminated_coverage : float
        Coverage fraction at which the episode ends (default 0.97).
    max_step : int
        Maximum number of viewpoint selections per episode.
    fuel_budget : float
        Maximum total Δv allowed per episode.  Episode terminates if exceeded.
        Default: 50.0 (dimensionless units).
    delta_v_weight : float
        γ — scaling factor for the Δv penalty in the reward.
        reward = coverage_reward - time_cost_weight·travel_time - delta_v_weight·Δv
        Default: 1.0  (equal weighting with other terms).
    time_cost_weight : float
        Scaling factor for the travel-time penalty.
        Default: 1.0.
    """

    def __init__(
        self,
        data_path:               str,
        view_num:                int   = 33,
        begin_view:              int   = -1,
        observation_space_dim:   int   = -1,
        terminated_coverage:     float = 0.97,
        max_step:                int   = 11,
        env_id:                  Optional[int] = None,
        logger                         = logging.getLogger(__name__),
        is_normalize:            bool  = True,
        is_ratio_reward:         bool  = False,
        is_reward_with_cur_coverage: bool = False,
        cur_coverage_ratio:      float = 1.0,
        # ── Level 2 specific ──────────────────────────────────────────────────
        fuel_budget:             float = 50.0,
        delta_v_weight:          float = 1.0,
        time_cost_weight:        float = 1.0,
    ):
        # ── Bookkeeping flags ─────────────────────────────────────────────────
        self.COVERAGE_THRESHOLD          = 0.00005
        self.is_ratio_reward             = is_ratio_reward
        self.is_reward_with_cur_coverage = is_reward_with_cur_coverage
        self.cur_coverage_ratio          = cur_coverage_ratio
        self.terminated_coverage         = terminated_coverage

        # ── Reward weights ────────────────────────────────────────────────────
        self.time_cost_weight = time_cost_weight
        self.delta_v_weight   = delta_v_weight  # γ from the Level-2 spec

        # ── Fuel budget ───────────────────────────────────────────────────────
        self.fuel_budget    = fuel_budget
        self.cumulative_dv  = 0.0   # tracks how much fuel we have spent

        # ── Action / observation spaces ───────────────────────────────────────
        self.action_space      = spaces.Discrete(view_num)
        self.view_num          = view_num
        self.observation_space_dim = observation_space_dim
        self.is_normalize      = is_normalize
        self.max_step          = max_step
        self.begin_view        = begin_view

        # ── Device ────────────────────────────────────────────────────────────
        self.DEVICE = (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.logger = logger
        self.logger.info("PointCloudNBVEnvLevel2 initialising …")

        # ── Data reader ───────────────────────────────────────────────────────
        real_data_path = data_path
        if env_id is not None:
            real_data_path = os.path.join(data_path, str(env_id))
        self.shapenet_reader = shapenet_reader.ShapenetReader(
            real_data_path, view_num, self.logger, True
        )

        # ── Initial viewpoint ─────────────────────────────────────────────────
        self.view_state = np.zeros(view_num, dtype=np.int32)
        if self.begin_view == -1:
            self.current_view = random.randint(0, self.view_num - 1)
        else:
            self.current_view = self.begin_view
        self.action_history = [self.current_view]

        # ── Point cloud initialisation ─────────────────────────────────────────
        self.current_points_cloud   = self.shapenet_reader.get_point_cloud_by_view_id(
            self.current_view
        )
        self.ground_truth_points_cloud      = self.shapenet_reader.ground_truth
        self.ground_truth_points_cloud_size = self.ground_truth_points_cloud.shape[0]
        self.ground_truth_tensor = torch.tensor(
            self.shapenet_reader.ground_truth[np.newaxis, :, :].astype(np.float32)
        ).to(self.DEVICE)

        self.view_state[self.current_view] = 1

        # ── Observation space definition ──────────────────────────────────────
        self._define_observation_space()

        self.current_coverage = self._calculate_current_coverage()
        self.coverage_add     = self.current_coverage
        self.step_cnt         = 1
        self.model_name       = self.shapenet_reader.get_model_info()

        # ── Orbital configuration ─────────────────────────────────────────────
        # orbit_radius=1.0  : camera moves on a unit sphere
        # grav_param=1.0    : dimensionless gravitational parameter
        # num_orbits=2.0    : mission horizon = two full orbits
        self.orbit_config = TargetOrbitConfig(
            orbit_radius=1.0,
            grav_param=1.0,
            num_orbits=2.0,
        )

        # ── Load viewpoints from disk ─────────────────────────────────────────
        viewpoints_path = os.path.join(
            os.path.dirname(__file__), "..", "output", "acrimsat_final", "viewpoints_33.txt"
        )
        if os.path.exists(viewpoints_path):
            self.viewpoints = np.loadtxt(viewpoints_path)   # shape (33, 3)
            self.logger.info(f"Loaded {self.viewpoints.shape[0]} viewpoints.")
        else:
            self.logger.warning(f"Viewpoints file not found: {viewpoints_path}")
            self.viewpoints = None

        # ── Pre-compute travel-time table  (33×33) ────────────────────────────
        if self.viewpoints is not None:
            self.travel_times = compute_all_travel_times(
                self.viewpoints, self.orbit_config
            )
            self.logger.info(
                f"Travel-time matrix shape: {self.travel_times.shape}"
            )
        else:
            self.travel_times = None

        # ── Pre-compute Δv table  (33×33)  ──  NEW IN LEVEL 2 ─────────────────
        # Each entry [i, j] is the fuel cost to go from viewpoint i to viewpoint j
        # in the allocated travel time.  This is the key Level-2 addition.
        if self.viewpoints is not None and self.travel_times is not None:
            self.logger.info("Pre-computing Δv matrix via CW equations …")
            self.delta_v_matrix = compute_delta_v_matrix(
                viewpoints   = self.viewpoints,
                travel_times = self.travel_times,
                orbit_radius = self.orbit_config.orbit_radius,
                mean_motion  = self.orbit_config.mean_motion,
            )
            self.logger.info(
                f"Δv matrix shape: {self.delta_v_matrix.shape}  "
                f"(max={self.delta_v_matrix[self.delta_v_matrix < np.inf].max():.4f})"
            )
        else:
            self.delta_v_matrix = None
            self.logger.warning("Δv matrix not computed (viewpoints unavailable).")

        # ── Mission clock ─────────────────────────────────────────────────────
        self.current_time = 0.0

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _define_observation_space(self):
        """Set up gym observation space with the correct tensor dimensions."""
        if self.observation_space_dim == -1:
            # Debug mode: no resampling
            self.observation_space = spaces.Dict({
                "current_point_cloud": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(512, 3), dtype=np.float64
                ),
                "view_state": spaces.Box(
                    low=0, high=1, shape=(self.view_num,), dtype=np.int32
                ),
            })
        else:
            lo = float("-1") if self.is_normalize else float("-inf")
            hi = float("1")  if self.is_normalize else float("inf")
            self.observation_space = spaces.Dict({
                "current_point_cloud": spaces.Box(
                    low=lo, high=hi,
                    shape=(3, self.observation_space_dim),
                    dtype=np.float64,
                ),
                "view_state": spaces.Box(
                    low=0, high=1, shape=(self.view_num,), dtype=np.int32
                ),
            })

    def _calculate_current_coverage(self) -> float:
        """
        Measure what fraction of the ground-truth model is covered by the
        current accumulated point cloud.

        Uses Chamfer distance: a ground-truth point is considered 'covered'
        if the nearest point in the observation cloud is within COVERAGE_THRESHOLD.
        """
        cur_tensor = torch.tensor(
            self.current_points_cloud[np.newaxis, :, :].astype(np.float32)
        ).to(self.DEVICE)

        dist1, dist2 = ChamferDistanceFunction.apply(cur_tensor, self.ground_truth_tensor)
        dist2 = dist2.cpu().numpy()

        covered = dist2 < self.COVERAGE_THRESHOLD
        coverage = np.sum(covered) / self.ground_truth_points_cloud_size

        # Remove newly covered points so they are not double-counted
        self.current_points_cloud_from_gt = self.ground_truth_points_cloud[covered[0, :]]
        self.ground_truth_points_cloud    = self.ground_truth_points_cloud[~covered[0, :]]
        self.ground_truth_tensor = torch.tensor(
            self.ground_truth_points_cloud[np.newaxis, :, :].astype(np.float32)
        ).to(self.DEVICE)

        return coverage

    def _get_observation_space(self) -> dict:
        """Return the current observation dict for the RL agent."""
        source_pc = self.current_points_cloud_from_gt
        if source_pc.shape[0] == 0:
            source_pc = self.current_points_cloud

        if self.observation_space_dim == -1:
            return {"current_point_cloud": source_pc.T, "view_state": self.view_state}

        cur_pc = resample_pcd(source_pc, self.observation_space_dim, self.logger, self.model_name)
        if self.is_normalize:
            cur_pc = normalize_pc(cur_pc, self.logger, self.model_name)
        return {"current_point_cloud": cur_pc.T, "view_state": self.view_state}

    def _get_terminated(self) -> bool:
        """
        Check all termination conditions.

        Level 2 adds a fuel-exhaustion check:
        if the agent has spent more Δv than the budget, the episode ends.
        """
        if self.step_cnt > self.max_step:
            self.logger.debug("[TERMINATE] Max steps reached.")
            return True
        if self.current_coverage >= self.terminated_coverage:
            self.logger.debug("[TERMINATE] Coverage goal reached.")
            return True
        if self.cumulative_dv > self.fuel_budget:
            self.logger.debug(
                f"[TERMINATE] Fuel budget exhausted: "
                f"{self.cumulative_dv:.4f} > {self.fuel_budget:.4f}"
            )
            return True
        return False

    def _get_info(self) -> dict:
        return {
            "cur_points_cloud":    self.ground_truth_points_cloud,
            "model_name":          self.model_name,
            "current_coverage":    self.current_coverage,
            # Level-2 extras
            "cumulative_dv":       self.cumulative_dv,
            "fuel_budget":         self.fuel_budget,
            "fuel_remaining":      max(0.0, self.fuel_budget - self.cumulative_dv),
        }

    # =========================================================================
    # Reward  –  the core Level-2 change
    # =========================================================================

    def _get_reward(
        self,
        cover_add:   float,
        action:      int,
        travel_time: float = 0.0,
        delta_v:     float = 0.0,
    ) -> float:
        """
        Calculate reward for the chosen viewpoint.

        Level-2 formula
        ---------------
        ::

            reward = coverage_reward
                     - time_cost_weight  · travel_time
                     - delta_v_weight    · delta_v

        This teaches the agent three things simultaneously:
          1. Choose viewpoints that add a lot of new coverage.
          2. Prefer viewpoints that are quick to reach.
          3. Prefer viewpoints that are cheap in fuel to reach.

        Parameters
        ----------
        cover_add   : float   New coverage fraction gained (0.0 – 1.0).
        action      : int     The chosen viewpoint index.
        travel_time : float   Time cost of the manoeuvre (dimensionless).
        delta_v     : float   Fuel cost of the manoeuvre (Δv, dimensionless).

        Returns
        -------
        float  –  scalar reward.
        """
        # ── Step 1 : coverage reward ──────────────────────────────────────────
        if self.is_reward_with_cur_coverage:
            if self.step_cnt < 4:
                coverage_reward = cover_add * 10
            else:
                if cover_add <= 0:
                    coverage_reward = cover_add * 10
                else:
                    # Reward increases as coverage nears completion
                    # (rarer uncovered points are more valuable)
                    remain          = 1.0 - (self.current_coverage - cover_add)
                    coverage_reward = (cover_add / remain) * 5 + cover_add * 5
        elif self.is_ratio_reward:
            if cover_add <= 0:
                coverage_reward = cover_add * 10
            else:
                remain          = 1.0 - (self.current_coverage - cover_add)
                coverage_reward = (cover_add / remain) * 10
        else:
            coverage_reward = cover_add * 10   # simple linear

        # ── Step 2 : travel-time penalty ─────────────────────────────────────
        time_penalty = self.time_cost_weight * travel_time

        # ── Step 3 : fuel (Δv) penalty  ──  NEW IN LEVEL 2 ───────────────────
        #
        # γ · Δv_k  penalises expensive manoeuvres.
        # The agent learns: "Is the coverage gain worth the fuel?"
        #
        # Example:
        #   coverage_reward = 0.5   (nice new view)
        #   time_penalty    = 0.1   (quick to reach)
        #   fuel_penalty    = 0.3   (but costs a lot of Δv)
        #   →  reward = 0.5 - 0.1 - 0.3 = 0.1   (marginal)
        #
        # Compare with a closer view:
        #   coverage_reward = 0.3   (less new information)
        #   time_penalty    = 0.05
        #   fuel_penalty    = 0.05
        #   →  reward = 0.3 - 0.05 - 0.05 = 0.2  (better overall!)
        #
        fuel_penalty = self.delta_v_weight * delta_v

        final_reward = coverage_reward - time_penalty - fuel_penalty

        self.logger.debug(
            f"[REWARD] action={action:2d}  "
            f"coverage={coverage_reward:6.4f}  "
            f"time_pen={time_penalty:6.4f}  "
            f"fuel_pen={fuel_penalty:6.4f}  "
            f"total={final_reward:6.4f}  "
            f"cumΔv={self.cumulative_dv:6.4f}/{self.fuel_budget:.1f}"
        )
        return final_reward

    # =========================================================================
    # Core gym interface
    # =========================================================================

    def step(self, action: int):
        """
        Execute one step: move to the selected viewpoint and return the outcome.

        Step-by-step walkthrough
        ------------------------
        1.  Look up travel time and Δv cost for  current_view → action.
        2.  Update the mission clock.
        3.  Update the cumulative fuel counter.
        4.  Check if the viewpoint has already been visited (penalise revisit).
        5.  Load the new point cloud.
        6.  Compute newly observed (non-overlapping) points.
        7.  Measure coverage increment against the ground truth.
        8.  Compute the combined reward.
        9.  Return (observation, reward, terminated, info).

        Parameters
        ----------
        action : int
            Index of the selected next viewpoint (0 … view_num-1).

        Returns
        -------
        observation : dict    Next state seen by the agent.
        reward      : float   Scalar reward signal.
        terminated  : bool    Whether the episode has ended.
        info        : dict    Diagnostics (coverage, fuel used, etc.).
        """

        # ── 1. Look up travel time and fuel cost ──────────────────────────────
        travel_time = 0.0
        delta_v     = 0.0

        if self.travel_times is not None:
            travel_time = float(self.travel_times[self.current_view, action])

        if self.delta_v_matrix is not None:
            raw_dv = float(self.delta_v_matrix[self.current_view, action])
            # Guard against np.inf (singular CW transfer — treat as very expensive)
            delta_v = raw_dv if np.isfinite(raw_dv) else self.fuel_budget

        self.logger.debug(
            f"[STEP] view {self.current_view:2d} → {action:2d}  "
            f"travel_time={travel_time:.4f}  Δv={delta_v:.4f}"
        )

        # ── 2. Advance mission clock ──────────────────────────────────────────
        self.current_time = min(
            self.current_time + travel_time,
            self.orbit_config.total_time
        )

        # ── 3. Burn fuel ──────────────────────────────────────────────────────
        self.cumulative_dv += delta_v

        # ── 4. Update step counter and action history ─────────────────────────
        self.action_history.append(action)
        self.step_cnt += 1

        # ── 4b. Penalise revisiting an already-visited viewpoint ──────────────
        if self.view_state[action] == 1:
            reward      = self._get_reward(-0.05, action, travel_time, delta_v)
            observation = self._get_observation_space()
            terminated  = self._get_terminated()
            info        = self._get_info()
            self.logger.debug(
                f"[REVISIT] action={action}  cover_add=0  "
                f"cur_coverage={self.current_coverage*100:.2f}%"
            )
            return observation, reward, terminated, info

        # ── 5. Load point cloud from the new viewpoint ────────────────────────
        new_points_cloud = self.shapenet_reader.get_point_cloud_by_view_id(action)
        self.view_state[action] = 1
        self.current_view       = action

        new_pc_tensor = torch.tensor(
            new_points_cloud[np.newaxis, :, :].astype(np.float32)
        ).to(self.DEVICE)
        cur_pc_tensor = torch.tensor(
            self.current_points_cloud[np.newaxis, :, :].astype(np.float32)
        ).to(self.DEVICE)

        # ── 6. Find points that are genuinely new (not already seen) ──────────
        dist1, _ = ChamferDistanceFunction.apply(new_pc_tensor, cur_pc_tensor)
        dist1 = dist1.cpu().numpy()
        new_mask = ~(dist1 < self.COVERAGE_THRESHOLD)[0, :]
        increase_points = new_points_cloud[new_mask]

        if increase_points.shape[0] == 0:
            # All points from this view are already covered — no gain
            reward      = self._get_reward(0, action, travel_time, delta_v)
            observation = self._get_observation_space()
            terminated  = self._get_terminated()
            info        = self._get_info()
            self.logger.debug(f"[NO_NEW] action={action}  no new points")
            return observation, reward, terminated, info

        self.current_points_cloud = np.append(
            self.current_points_cloud, increase_points, axis=0
        )

        # ── 7. Compute coverage increment vs ground truth ─────────────────────
        inc_tensor = torch.tensor(
            increase_points[np.newaxis, :, :].astype(np.float32)
        ).to(self.DEVICE)

        _, dist2 = ChamferDistanceFunction.apply(inc_tensor, self.ground_truth_tensor)
        dist2 = dist2.cpu().numpy()

        cover_mask = dist2 < self.COVERAGE_THRESHOLD
        cover_add  = float(np.sum(cover_mask)) / self.ground_truth_points_cloud_size
        self.current_coverage += cover_add
        self.coverage_add      = cover_add

        # Remove newly covered ground-truth points (no double-counting)
        self.current_points_cloud_from_gt = np.append(
            self.current_points_cloud_from_gt,
            self.ground_truth_points_cloud[cover_mask[0, :]],
            axis=0,
        )
        self.ground_truth_points_cloud = self.ground_truth_points_cloud[~cover_mask[0, :]]
        self.ground_truth_tensor = torch.tensor(
            self.ground_truth_points_cloud[np.newaxis, :, :].astype(np.float32)
        ).to(self.DEVICE)

        # ── 8. Compute Level-2 reward ─────────────────────────────────────────
        reward = self._get_reward(cover_add, action, travel_time, delta_v)

        # ── 9. Build and return output ────────────────────────────────────────
        observation = self._get_observation_space()
        terminated  = self._get_terminated()
        info        = self._get_info()

        info.update({
            "travel_time":          travel_time,
            "delta_v":              delta_v,
            "mission_time":         self.current_time,
            "mission_time_horizon": self.orbit_config.total_time,
        })

        self.logger.debug(
            f"[SUCCESS] action={action:2d}  cover_add={cover_add*100:.2f}%  "
            f"cur_cover={self.current_coverage*100:.2f}%  "
            f"Δv={delta_v:.4f}  cumΔv={self.cumulative_dv:.4f}  "
            f"step={self.step_cnt}  terminated={terminated}"
        )
        return observation, reward, terminated, info

    def reset(self, init_step: int = -1):
        """
        Reset the environment to a new episode.

        Resets all state including the mission clock and the cumulative fuel
        counter.  A new ShapeNet model is loaded from the dataset.

        Parameters
        ----------
        init_step : int
            If >= 0, forces the starting viewpoint to this index.

        Returns
        -------
        observation : dict   Initial observation for the new episode.
        """
        self.shapenet_reader.get_next_model()
        self.view_state = np.zeros(self.view_num, dtype=np.int32)
        self.action_history.clear()

        # Choose starting viewpoint
        if init_step >= 0:
            self.current_view = init_step
        elif self.begin_view == -1:
            self.current_view = random.randint(0, self.view_num - 1)
        else:
            self.current_view = self.begin_view
        self.action_history.append(self.current_view)

        # Point clouds
        self.current_points_cloud = self.shapenet_reader.get_point_cloud_by_view_id(
            self.current_view
        )
        self.view_state[self.current_view] = 1

        self.ground_truth_points_cloud      = self.shapenet_reader.ground_truth
        self.ground_truth_points_cloud_size = self.ground_truth_points_cloud.shape[0]
        self.ground_truth_tensor = torch.tensor(
            self.shapenet_reader.ground_truth[np.newaxis, :, :].astype(np.float32)
        ).to(self.DEVICE)

        self.current_coverage = self._calculate_current_coverage()
        self.coverage_add     = self.current_coverage
        self.step_cnt         = 1
        self.model_name       = self.shapenet_reader.get_model_info()

        # Reset mission clock and fuel counter  ← Level-2 specific
        self.current_time  = 0.0
        self.cumulative_dv = 0.0

        self.logger.debug(
            f"[reset] starting view={self.current_view}  "
            f"fuel_budget={self.fuel_budget:.1f}"
        )
        return self._get_observation_space()

    # ── Standard gym stubs ────────────────────────────────────────────────────
    def close(self):
        pass

    def render(self):
        pass

    # =========================================================================
    # Greedy-policy helper (unchanged from Level 1, fuel-aware version)
    # =========================================================================

    def try_step(self, action: int) -> float:
        """
        Simulate an action *without* committing to it.

        Returns the estimated coverage gain if the agent were to visit this
        viewpoint next.  Used by a greedy baseline policy.

        Does NOT update: view_state, current_view, cumulative_dv, current_time.
        """
        if self.view_state[action] == 1:
            return 0.0

        new_pc = self.shapenet_reader.get_point_cloud_by_view_id(action)
        new_pc_tensor = torch.tensor(
            new_pc[np.newaxis, :, :].astype(np.float32)
        ).to(self.DEVICE)
        cur_pc_tensor = torch.tensor(
            self.current_points_cloud[np.newaxis, :, :].astype(np.float32)
        ).to(self.DEVICE)

        dist1, _ = ChamferDistanceFunction.apply(new_pc_tensor, cur_pc_tensor)
        dist1 = dist1.cpu().numpy()
        new_mask       = ~(dist1 < self.COVERAGE_THRESHOLD)[0, :]
        increase_points = new_pc[new_mask]

        if increase_points.shape[0] == 0:
            return 0.0

        inc_tensor = torch.tensor(
            increase_points[np.newaxis, :, :].astype(np.float32)
        ).to(self.DEVICE)
        _, dist2 = ChamferDistanceFunction.apply(inc_tensor, self.ground_truth_tensor)
        dist2 = dist2.cpu().numpy()

        cover_add = float(np.sum(dist2 < self.COVERAGE_THRESHOLD))
        cover_add /= self.ground_truth_points_cloud_size
        return cover_add