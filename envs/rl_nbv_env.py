import numpy as np
import math
import gym
from gym import spaces
import envs.shapenet_reader as shapenet_reader
import random
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from distance.chamfer_distance import ChamferDistanceFunction
from envs.state_transition import (
    TargetOrbitConfig,
    build_state,
    calculate_sun_position,
    compute_all_travel_times,
    get_lit_visible_points,
    orchestrate_step,
    update_coverage_map,
)
from envs.state_transition.cw_utils import compute_delta_v_matrix
import logging

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def resample_pcd(pcd, n, logger, name):
    """Drop or duplicate points so that pcd has exactly n points"""
    if pcd.shape[0] == 0:
        logger.debug("observation source point cloud is empty, model: {}".format(name))
        return np.zeros((n, 3))
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate(
            [idx, np.random.randint(pcd.shape[0], size=n - pcd.shape[0])]
        )
    logger.debug("resample_pcd from {} to {}, model: {}".format(pcd.shape[0], n, name))
    return pcd[idx[:n]]


def normalize_pc(points, logger, name):
    if points.shape[0] == 0:
        logger.debug("normalize received empty points, model: {}".format(name))
        return points
    centroid = np.mean(points, axis=0)
    points -= centroid
    furthest_distance = np.max(np.sqrt(np.sum(abs(points) ** 2, axis=-1)))
    if furthest_distance == 0:
        logger.debug(
            "normalize skipped due to zero furthest distance, model: {}".format(name)
        )
        return points
    points /= furthest_distance
    logger.debug(
        "normalize furthest distance: {:.6f}, model: {}".format(furthest_distance, name)
    )
    return points


class PointCloudNextBestViewEnv(gym.Env):
    def __init__(
        self,
        data_path,
        viewpoints_path,
        view_num=33,
        begin_view=-1,
        observation_space_dim=-1,
        terminated_coverage=0.97,
        max_step=11,
        env_id=None,
        logger=logging.getLogger(__name__),
        is_normalize=True,
        is_ratio_reward=False,
        is_reward_with_cur_coverage=False,
        cur_coverage_ratio=1.0,
        time_cost_weight=1.0,
        fuel_budget=50.0,
        delta_v_weight=1.0,
        sun_position_config=None,
    ):
        """
        Initialize Point Cloud Next Best View Environment.

        Args:
            time_cost_weight: Weight of travel time penalty in reward calculation.
                            reward = coverage_gain - time_cost_weight * travel_time
                            Default 1.0: equal weight to coverage and time cost
                            Higher value: penalize time more heavily
                            Lower value: focus more on coverage
        """
        self.COVERAGE_THRESHOLD = 0.00005
        self.is_ratio_reward = is_ratio_reward
        self.is_reward_with_cur_coverage = is_reward_with_cur_coverage
        self.cur_coverage_ratio = cur_coverage_ratio
        self.time_cost_weight = time_cost_weight
        self.delta_v_weight = delta_v_weight
        self.fuel_budget = fuel_budget
        self.cumulative_dv = 0.0
        self.sun_position_config = sun_position_config or {}
        self.terminated_coverage = terminated_coverage
        self.action_space = spaces.Discrete(view_num)
        self.DEVICE = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.logger = logger
        self.logger.info("PointCloudNextBestViewEnv is ok")
        real_data_path = data_path
        if env_id is not None:
            real_data_path = os.path.join(data_path, str(env_id))
        self.shapenet_reader = shapenet_reader.ShapenetReader(
            real_data_path, view_num, self.logger, True
        )
        self.view_state = np.zeros(view_num, dtype=np.int32)
        self.view_num = view_num
        self.begin_view = begin_view
        self.max_step = max_step
        if self.begin_view == -1:
            self.current_view = random.randint(0, self.view_num - 1)
            self.logger.info("random init view: {}".format(self.current_view))
        else:
            self.current_view = self.begin_view
        self.action_history = [self.current_view]
        self.current_points_cloud = self.shapenet_reader.get_point_cloud_by_view_id(
            self.current_view
        )
        self.ground_truth_points_cloud = self.shapenet_reader.ground_truth
        self.ground_truth_points_cloud_size = self.ground_truth_points_cloud.shape[0]
        self.ground_truth_tensor = self.shapenet_reader.ground_truth[
            np.newaxis, :, :
        ].astype(np.float32)
        self.ground_truth_tensor = torch.tensor(self.ground_truth_tensor).to(
            self.DEVICE
        )
        self.view_state[self.current_view] = 1
        self.observation_space_dim = observation_space_dim
        self.is_normalize = is_normalize
        if observation_space_dim == -1:
            # for debug
            self.observation_space = spaces.Dict(
                {
                    "current_point_cloud": spaces.Box(
                        low=float("-inf"),
                        high=float("inf"),
                        shape=(512, 3),
                        dtype=np.float64,
                    ),
                    "view_state": spaces.Box(
                        low=0, high=1, shape=(view_num,), dtype=np.int32
                    ),
                }
            )
        else:
            if self.is_normalize:
                self.observation_space = spaces.Dict(
                    {
                        "current_point_cloud": spaces.Box(
                            low=float("-1"),
                            high=float("1"),
                            shape=(3, observation_space_dim),
                            dtype=np.float64,
                        ),
                        "view_state": spaces.Box(
                            low=0, high=1, shape=(view_num,), dtype=np.int32
                        ),
                    }
                )
            else:
                self.observation_space = spaces.Dict(
                    {
                        "current_point_cloud": spaces.Box(
                            low=float("-inf"),
                            high=float("inf"),
                            shape=(3, observation_space_dim),
                            dtype=np.float64,
                        ),
                        "view_state": spaces.Box(
                            low=0, high=1, shape=(view_num,), dtype=np.int32
                        ),
                    }
                )
        self.current_coverage = 0.0
        self.coverage_add = 0.0
        self.step_cnt = 1
        self.model_name = self.shapenet_reader.get_model_info()
        self.reward_config = {
            "revisit_penalty": -5.0,
            "coverage_coeff": 1.0,
        }

        # ============================================================================
        # TRAVEL TIME INITIALIZATION
        # ============================================================================
        # Load viewpoints from the required path
        resolved_viewpoints_path = os.path.expanduser(viewpoints_path)
        if not os.path.isabs(resolved_viewpoints_path):
            resolved_viewpoints_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", resolved_viewpoints_path)
            )

        if not os.path.exists(resolved_viewpoints_path):
            raise FileNotFoundError(
                "viewpoints_path does not exist: {}".format(resolved_viewpoints_path)
            )

        self.viewpoints = np.loadtxt(resolved_viewpoints_path)  # Shape: (N, 3)
        self.logger.info(
            f"Loaded {self.viewpoints.shape[0]} viewpoints from {resolved_viewpoints_path}"
        )

        # Initialize orbital configuration for travel time calculations
        # orbit_radius: 1.0 (unit sphere)
        # grav_param: 1.0 (dimensionless, controls orbital dynamics)
        # num_orbits: 2.0 (mission horizon = 2 complete orbits)
        self.orbit_config = TargetOrbitConfig(
            orbit_radius=1.0, grav_param=1.0, num_orbits=2.0
        )

        # Precompute travel times between all pairs of viewpoints
        # Shape: (view_num, view_num) where [i,j] = travel time from view i to view j
        if self.viewpoints is not None:
            self.travel_times = compute_all_travel_times(
                self.viewpoints, self.orbit_config
            )
            self.logger.info(
                f"Precomputed travel times matrix shape: {self.travel_times.shape}"
            )
        else:
            self.travel_times = None
            self.logger.warning("Travel times not computed (viewpoints unavailable)")

        if self.viewpoints is not None and self.travel_times is not None:
            self.delta_v_matrix = compute_delta_v_matrix(
                viewpoints=self.viewpoints,
                travel_times=self.travel_times,
                orbit_radius=self.orbit_config.orbit_radius,
                mean_motion=self.orbit_config.mean_motion,
            )
        else:
            self.delta_v_matrix = None

        # Current mission time (starts at 0.0, increments as agent moves)
        self.current_time = 0.0

        # ============================================================================
        # SUN POSITION INITIALIZATION
        # ============================================================================
        default_sun_position = np.array([1.0, 0.0, 0.0], dtype=float)
        initial_sun_position = self.sun_position_config.get(
            "initial_direction", default_sun_position
        )
        self.initial_sun_position = np.asarray(initial_sun_position, dtype=float)
        if self.initial_sun_position.shape != (3,):
            raise ValueError(
                "sun_position_config['initial_direction'] must have shape (3,), got {}".format(
                    self.initial_sun_position.shape
                )
            )

        self.sun_orbital_params = self.sun_position_config.get("orbital_params", {})
        if not self.sun_orbital_params:
            self.sun_orbital_params = {"angular_velocity_rad_per_s": 0.0}

        self.current_sun_position = self.initial_sun_position.astype(float)
        self.logger.debug(
            "[SUN] Initialized. current_sun_position={} orbital_params={}".format(
                self.current_sun_position.tolist(), self.sun_orbital_params
            )
        )

        # State-transition runtime structures.
        self.geometry_cache = {}
        self._transition_state = None
        self._canonical_points = np.zeros((0, 3), dtype=np.float32)
        self._model_transition_cache = {}
        self._initialize_state_transition_for_current_model(self.current_view)

    def _estimate_surface_normals(self, points):
        if points.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float32)
        centroid = np.mean(points, axis=0, keepdims=True)
        vectors = points - centroid
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normals = vectors / np.maximum(norms, 1e-12)
        degenerate = norms[:, 0] <= 1e-12
        if np.any(degenerate):
            normals[degenerate] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return normals.astype(np.float32)

    def _build_visible_points_by_view(self, canonical_points):
        point_count = canonical_points.shape[0]
        visible_points = np.zeros((self.view_num, point_count), dtype=bool)
        if point_count == 0:
            return visible_points

        canonical_tensor = canonical_points[np.newaxis, :, :].astype(np.float32)
        canonical_tensor = torch.tensor(canonical_tensor).to(self.DEVICE)

        for view_id in range(self.view_num):
            view_points = self.shapenet_reader.get_point_cloud_by_view_id(view_id)
            if view_points is None or view_points.shape[0] == 0:
                continue
            view_tensor = view_points[np.newaxis, :, :].astype(np.float32)
            view_tensor = torch.tensor(view_tensor).to(self.DEVICE)
            _, dist_to_canonical = ChamferDistanceFunction.apply(
                view_tensor, canonical_tensor
            )
            visible_points[view_id] = (
                dist_to_canonical.detach().cpu().numpy()[0] < self.COVERAGE_THRESHOLD
            )

        return visible_points

    def _sync_legacy_coverage_buffers(self):
        if self._transition_state is None:
            return

        coverage_map = np.asarray(self._transition_state["coverage_map"], dtype=bool)
        self.current_points_cloud_from_gt = self._canonical_points[coverage_map]
        self.ground_truth_points_cloud = self._canonical_points[~coverage_map]
        self.ground_truth_tensor = self.ground_truth_points_cloud[
            np.newaxis, :, :
        ].astype(np.float32)
        self.ground_truth_tensor = torch.tensor(self.ground_truth_tensor).to(
            self.DEVICE
        )

    def _initialize_state_transition_for_current_model(self, initial_view):
        model_name = self.shapenet_reader.get_model_info()
        cached_geometry = self._model_transition_cache.get(model_name)

        if cached_geometry is None:
            self._canonical_points = np.asarray(
                self.shapenet_reader.ground_truth, dtype=np.float32
            )
            surface_normals = self._estimate_surface_normals(self._canonical_points)
            visible_points_by_view = self._build_visible_points_by_view(
                self._canonical_points
            )
            cached_geometry = {
                "canonical_points": self._canonical_points,
                "surface_normals": surface_normals,
                "visible_points_by_view": visible_points_by_view,
            }
            self._model_transition_cache[model_name] = cached_geometry
        else:
            self._canonical_points = cached_geometry["canonical_points"]

        self.ground_truth_points_cloud_size = self._canonical_points.shape[0]
        self.geometry_cache = {
            "canonical_points": cached_geometry["canonical_points"],
            "surface_normals": cached_geometry["surface_normals"],
            "view_positions": self.viewpoints,
            "travel_times_matrix": self.travel_times,
            "visible_points_by_view": cached_geometry["visible_points_by_view"],
        }

        initial_visible_lit = get_lit_visible_points(
            action=int(initial_view),
            new_sun_position=self.current_sun_position,
            geometry_cache=self.geometry_cache,
        )
        initial_coverage = update_coverage_map(
            prev_coverage_map=np.zeros(self.ground_truth_points_cloud_size, dtype=bool),
            visible_lit_points=initial_visible_lit,
            total_points=self.ground_truth_points_cloud_size,
        )

        if self.travel_times is not None:
            current_travel_times = self.travel_times[int(initial_view)]
        else:
            current_travel_times = np.zeros(self.view_num, dtype=np.float32)

        self._transition_state = build_state(
            action=int(initial_view),
            new_time=self.current_time,
            new_sun_position=self.current_sun_position,
            new_coverage_map=initial_coverage.coverage_map,
            prev_views=np.zeros(self.view_num, dtype=bool),
            travel_times=current_travel_times,
        )

        self.view_state = self._transition_state["views"].astype(np.int32)
        self.current_coverage = float(np.sum(initial_coverage.coverage_map)) / float(
            max(self.ground_truth_points_cloud_size, 1)
        )
        self.coverage_add = initial_coverage.newly_covered_ratio
        self._sync_legacy_coverage_buffers()

    def step(self, action):
        action = int(action)
        previous_state = self._transition_state
        was_visited = bool(previous_state["views"][action])
        prev_view = int(previous_state["_current_view"])

        delta_v = 0.0
        if self.delta_v_matrix is not None:
            raw_dv = float(self.delta_v_matrix[prev_view, action])
            delta_v = raw_dv if np.isfinite(raw_dv) else self.fuel_budget

        step_result = orchestrate_step(
            action=action,
            current_state=previous_state,
            orbit_config=self.orbit_config,
            geometry_cache=self.geometry_cache,
            orbital_params=self.sun_orbital_params,
            reward_config=self.reward_config,
            total_points=self.ground_truth_points_cloud_size,
        )

        self._transition_state = step_result.next_state
        self.current_view = int(self._transition_state["_current_view"])
        self.current_time = float(self._transition_state["_current_time"])
        self.current_sun_position = self._transition_state["sun_position"].copy()
        self.view_state = self._transition_state["views"].astype(np.int32)
        self.coverage_add = step_result.coverage_update.newly_covered_ratio
        self.current_coverage = float(
            np.sum(self._transition_state["coverage_map"])
        ) / float(max(self.ground_truth_points_cloud_size, 1))

        if not was_visited:
            selected_view_points = self.shapenet_reader.get_point_cloud_by_view_id(
                action
            )
            if selected_view_points is not None and selected_view_points.shape[0] > 0:
                self.current_points_cloud = np.append(
                    self.current_points_cloud, selected_view_points, axis=0
                )

        self._sync_legacy_coverage_buffers()
        self.action_history.append(action)
        self.step_cnt += 1
        self.cumulative_dv += delta_v

        observation = self._get_observation_space()
        terminated = self._get_terminated()
        info = self._get_info()

        reward = step_result.reward - (self.delta_v_weight * delta_v)

        info["travel_time"] = step_result.travel_time
        info["delta_v"] = delta_v
        info["mission_time"] = self.current_time
        info["mission_time_horizon"] = self.orbit_config.total_time
        info["reward_breakdown"] = step_result.reward_breakdown
        info["newly_covered_count"] = step_result.coverage_update.newly_covered_count

        if self.coverage_add == 1:
            self.logger.error("cover_add is 1")
            self._get_debug_info()

        log_label = "REVISIT" if was_visited else "SUCCESS"
        self.logger.debug(
            "[step] {} | action: {:2d}, travel_time: {:.6f}, time: {:.6f}/{:.6f}, cover_add: {:.2f}, "
            "cur_cover: {:.2f}, step_cnt: {:2d}, terminated: {}".format(
                log_label,
                action,
                step_result.travel_time,
                self.current_time,
                self.orbit_config.total_time,
                self.coverage_add * 100,
                self.current_coverage * 100,
                self.step_cnt,
                terminated,
            )
        )
        return observation, reward, terminated, info

    # for greedy policy test
    def try_step(self, action):
        # ============================================================================
        # TRY_STEP: Simulate action without updating state (for planning/evaluation)
        # ============================================================================
        # This method tests the value of an action without committing to it.
        # Used for greedy policy evaluation and planning.
        action = int(action)
        if self._transition_state is None:
            return 0
        if bool(self._transition_state["views"][action]):
            return 0

        result = orchestrate_step(
            action=action,
            current_state=self._transition_state,
            orbit_config=self.orbit_config,
            geometry_cache=self.geometry_cache,
            orbital_params=self.sun_orbital_params,
            reward_config=self.reward_config,
            total_points=self.ground_truth_points_cloud_size,
        )
        return result.coverage_update.newly_covered_ratio

    def reset(self, init_step=-1):
        self.shapenet_reader.get_next_model()
        self.action_history.clear()
        if self.begin_view == -1:
            self.current_view = random.randint(0, self.view_num - 1)
        else:
            self.current_view = self.begin_view
        if init_step != -1:
            self.current_view = init_step
        self.action_history.append(self.current_view)
        self.current_points_cloud = self.shapenet_reader.get_point_cloud_by_view_id(
            self.current_view
        )
        self.view_state = np.zeros(self.view_num, dtype=np.int32)
        self.step_cnt = 1
        self.model_name = self.shapenet_reader.get_model_info()

        # ============================================================================
        # RESET TRAVEL TIME AND MISSION TIME
        # ============================================================================
        # Reset mission time to beginning of episode
        self.current_time = 0.0
        self.logger.debug(
            f"[reset] Mission time reset to 0.0. Horizon: {self.orbit_config.total_time:.6f} time units"
        )

        # Re-initialize and synchronize sun direction with reset mission time.
        self.current_sun_position = calculate_sun_position(
            action=self.current_view,
            new_time=self.current_time,
            prev_sun_position=self.initial_sun_position,
            orbital_params=self.sun_orbital_params,
        )
        self.logger.debug(
            "[SUN] reset: t={:.6f} dir=[{:.6f}, {:.6f}, {:.6f}]".format(
                self.current_time,
                self.current_sun_position[0],
                self.current_sun_position[1],
                self.current_sun_position[2],
            )
        )

        self._initialize_state_transition_for_current_model(self.current_view)
        self.cumulative_dv = 0.0

        observation = self._get_observation_space()
        info = self._get_info()
        self.logger.debug("[reset] pass, init step: {}".format(self.current_view))
        return observation

    def close(self):
        pass

    def render(sellf):
        pass

    def _caculate_current_coverage(self):
        if self._transition_state is None:
            return 0.0
        coverage = float(np.sum(self._transition_state["coverage_map"])) / float(
            max(self.ground_truth_points_cloud_size, 1)
        )
        self._sync_legacy_coverage_buffers()
        return coverage

    def _get_reward(self, cover_add, action, travel_time=0.0, delta_v=0.0):
        """
        Calculate reward combining coverage gain and travel time cost.

        Reward = coverage_reward - time_cost_weight * travel_time - delta_v_weight * delta_v

        This encourages agent to:
        1. Maximize new coverage observation
        2. Minimize travel time (explore efficiently)
        3. Balance between distant high-value targets and nearby ones

        Args:
            cover_add: Coverage gained (0.0 to 1.0)
            action: Selected viewpoint action
            travel_time: Time cost to reach this viewpoint (dimensionless units)

        Returns:
            Scalar reward value
        """
        # ========================================================================
        # STEP 1: CALCULATE BASE COVERAGE REWARD
        # ========================================================================
        if self.is_reward_with_cur_coverage == True:
            # Reward based on current coverage progress
            if self.step_cnt < 4:
                coverage_reward = cover_add * 10
            else:
                if cover_add <= 0:
                    coverage_reward = cover_add * 10
                else:
                    # Reward increases as coverage completes (scarcity-based)
                    remain = 1.0 - (self.current_coverage - cover_add)
                    coverage_reward = (cover_add / remain) * 5 + cover_add * 5
        elif self.is_ratio_reward:
            # Ratio-based reward: prioritize new coverage when near completion
            if cover_add <= 0:
                coverage_reward = cover_add * 10
            else:
                remain = 1.0 - (self.current_coverage - cover_add)
                coverage_reward = (cover_add / remain) * 10
        else:
            # Simple linear reward: reward = coverage * constant
            coverage_reward = cover_add * 10

        # ========================================================================
        # STEP 2: APPLY TRAVEL TIME PENALTY
        # ========================================================================
        # Subtract time cost from coverage gain
        # This makes agent think about efficiency:
        # "Is this viewpoint worth the travel time?"
        #
        # Example:
        # - High coverage gain (0.10) but far away (travel_time=0.5)
        #   reward = 1.0 - 1.0*0.5 = 0.5
        # - Low coverage gain (0.02) but very close (travel_time=0.05)
        #   reward = 0.2 - 1.0*0.05 = 0.15
        #
        # Agent learns to balance:
        # coverage_reward / travel_time = efficiency
        time_penalty = self.time_cost_weight * travel_time
        fuel_penalty = self.delta_v_weight * delta_v
        final_reward = coverage_reward - time_penalty - fuel_penalty

        self.logger.debug(
            f"[REWARD] action={action:2d}, coverage_reward={coverage_reward:7.4f}, "
            f"time_penalty={time_penalty:7.4f}, fuel_penalty={fuel_penalty:7.4f}, final={final_reward:7.4f}"
        )

        return final_reward

    def _get_observation_space(self):
        if self.observation_space_dim == -1:
            # do not downsample, just for debug
            source_pc = self.current_points_cloud_from_gt
            if source_pc.shape[0] == 0:
                source_pc = self.current_points_cloud
            cur_pc = source_pc.T
            return {"current_point_cloud": cur_pc, "view_state": self.view_state}
        else:
            source_pc = self.current_points_cloud_from_gt
            if source_pc.shape[0] == 0:
                source_pc = self.current_points_cloud
            cur_pc = resample_pcd(
                source_pc,
                self.observation_space_dim,
                self.logger,
                self.model_name,
            )
            if self.is_normalize:
                cur_pc = normalize_pc(cur_pc, self.logger, self.model_name)
            # for PC_NBV net
            cur_pc = cur_pc.T
            return {"current_point_cloud": cur_pc, "view_state": self.view_state}

    def _get_terminated(self):
        if self.step_cnt > self.max_step:
            return True
        if self.current_coverage >= self.terminated_coverage:
            return True
        if self.cumulative_dv > self.fuel_budget:
            return True
        return False

    def _get_info(self):
        return {
            "cur_points_cloud": self.ground_truth_points_cloud,
            "model_name": self.model_name,
            "current_coverage": self.current_coverage,
            "sun_position": self.current_sun_position.copy(),
            "mission_time": self.current_time,
            "cumulative_dv": self.cumulative_dv,
            "fuel_budget": self.fuel_budget,
            "fuel_remaining": max(0.0, self.fuel_budget - self.cumulative_dv),
        }

    def _get_debug_info(self):
        self.logger.info(
            "model name:{}, action history: {}".format(
                self.model_name, self.action_history
            )
        )


PointCloudNBVEnvLevel2 = PointCloudNextBestViewEnv
