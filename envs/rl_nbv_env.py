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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from distance.chamfer_distance import ChamferDistanceFunction
from envs.state_transition import TargetOrbitConfig, get_travel_time, compute_all_travel_times
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
        self.current_coverage = self._caculate_current_coverage()
        self.coverage_add = self.current_coverage
        self.step_cnt = 1
        self.model_name = self.shapenet_reader.get_model_info()
        
        # ============================================================================
        # TRAVEL TIME INITIALIZATION
        # ============================================================================
        # Load viewpoints from disk (33 viewpoints on unit sphere surface)
        viewpoints_path = os.path.join(
            os.path.dirname(__file__), "..", "output", "acrimsat_final", "viewpoints_33.txt"
        )
        if os.path.exists(viewpoints_path):
            self.viewpoints = np.loadtxt(viewpoints_path)  # Shape: (33, 3)
            self.logger.info(f"Loaded {self.viewpoints.shape[0]} viewpoints from {viewpoints_path}")
        else:
            self.logger.warning(f"Viewpoints file not found at {viewpoints_path}")
            self.viewpoints = None
        
        # Initialize orbital configuration for travel time calculations
        # orbit_radius: 1.0 (unit sphere)
        # grav_param: 1.0 (dimensionless, controls orbital dynamics)
        # num_orbits: 2.0 (mission horizon = 2 complete orbits)
        self.orbit_config = TargetOrbitConfig(
            orbit_radius=1.0,
            grav_param=1.0,
            num_orbits=2.0
        )
        
        # Precompute travel times between all pairs of viewpoints
        # Shape: (view_num, view_num) where [i,j] = travel time from view i to view j
        if self.viewpoints is not None:
            self.travel_times = compute_all_travel_times(self.viewpoints, self.orbit_config)
            self.logger.info(f"Precomputed travel times matrix shape: {self.travel_times.shape}")
        else:
            self.travel_times = None
            self.logger.warning("Travel times not computed (viewpoints unavailable)")
        
        # Current mission time (starts at 0.0, increments as agent moves)
        self.current_time = 0.0

    def step(self, action):
        # ============================================================================
        # STEP 1: RETRIEVE TRAVEL TIME TO TARGET VIEWPOINT
        # ============================================================================
        # Travel time represents the time cost of rotating the camera from the current
        # viewpoint to the target viewpoint on the unit sphere surface.
        travel_time = 0.0
        if self.travel_times is not None and self.viewpoints is not None:
            # Lookup travel time from precomputed matrix
            # travel_times[current_view][action] = time to go from current_view to action
            travel_time = self.travel_times[self.current_view][action]
            self.logger.debug(
                f"[TRAVEL_TIME] From view {self.current_view} to view {action}: {travel_time:.6f} time units"
            )
        else:
            travel_time = 0.0
            self.logger.warning("[TRAVEL_TIME] Travel times unavailable, using travel_time=0")
        
        # ============================================================================
        # STEP 2: UPDATE MISSION TIME
        # ============================================================================
        # Advance the mission clock by the travel time.
        # The mission has a time horizon (total_time) that limits exploration.
        previous_time = self.current_time
        self.current_time = self.current_time + travel_time
        
        # Clamp time to mission horizon
        if self.current_time > self.orbit_config.total_time:
            self.current_time = self.orbit_config.total_time
            self.logger.debug(
                f"[TIME] Time exceeded mission horizon. Clamped: {previous_time:.6f} -> {self.current_time:.6f}"
            )
        else:
            self.logger.debug(
                f"[TIME] Mission time advanced: {previous_time:.6f} + {travel_time:.6f} = {self.current_time:.6f} "
                f"/ {self.orbit_config.total_time:.6f}"
            )
        
        # ============================================================================
        # STEP 3 & 4: UPDATE CURRENT VIEW AND TRACK ACTION HISTORY
        # ============================================================================
        self.action_history.append(action)
        self.step_cnt += 1
        
        if self.view_state[action] == 1:
            # ====================================================================
            # ALREADY VISITED: Penalize revisiting the same viewpoint
            # ====================================================================
            # The agent receives a penalty for revisiting a viewpoint that has
            # already been observed. This encourages exploration of new viewpoints.
            # Also charge travel time cost - discourage waste of time on revisits
            reward = self._get_reward(-0.05, action, travel_time=travel_time)
            observation = self._get_observation_space()
            terminated = self._get_terminated()
            info = self._get_info()
            self.logger.debug(
                "[step] REVISIT | action: {:2d}, travel_time: {:.6f}, time: {:.6f}/{:.6f}, cover_add: {:.2f}, "
                "cur_cover: {:.2f}, step_cnt: {:2d}, terminated: {}".format(
                    action, travel_time, self.current_time, self.orbit_config.total_time, 
                    0, self.current_coverage * 100, self.step_cnt, terminated
                )
            )
            return observation, reward, terminated, info
        
        # ============================================================================
        # STEP 5: LOAD POINT CLOUD FROM SELECTED VIEWPOINT
        # ============================================================================
        # Load the point cloud (3D points) that would be visible from the target viewpoint.
        new_points_cloud = self.shapenet_reader.get_point_cloud_by_view_id(action)
        self.view_state[action] = 1
        
        # Update current view for next iteration
        self.current_view = action
        new_points_cloud_tensor = new_points_cloud[np.newaxis, :, :].astype(np.float32)
        new_points_cloud_tensor = torch.tensor(new_points_cloud_tensor).to(self.DEVICE)
        cur_points_cloud_tensor = self.current_points_cloud[np.newaxis, :, :].astype(
            np.float32
        )
        cur_points_cloud_tensor = torch.tensor(cur_points_cloud_tensor).to(self.DEVICE)
        
        # ============================================================================
        # STEP 6: COMPUTE NEWLY OBSERVED POINTS (UNSEEN POINTS)
        # ============================================================================
        # Use Chamfer distance to identify which points in the new view have already
        # been observed (overlay_flag = True means already seen, False = newly observed).
        # Only newly observed points contribute to coverage reward.
        dist1, dist2 = ChamferDistanceFunction.apply(
            new_points_cloud_tensor, cur_points_cloud_tensor
        )
        dist1 = dist1.cpu().numpy()
        overlay_flag = dist1 < self.COVERAGE_THRESHOLD

        increase_points_cloud = new_points_cloud[~overlay_flag[0, :]]
        if increase_points_cloud.shape[0] == 0:
            # ====================================================================
            # NO NEW COVERAGE: All visible points from this view are already seen
            # ====================================================================
            # Agent still pays time cost, but doesn't gain coverage
            # Encourages learning which viewpoints are redundant
            reward = self._get_reward(0, action, travel_time=travel_time)
            observation = self._get_observation_space()
            terminated = self._get_terminated()
            info = self._get_info()
            self.logger.debug(
                "[step] NO_NEW | action: {:2d}, travel_time: {:.6f}, time: {:.6f}/{:.6f}, cover_add: {:.2f}, "
                "cur_cover: {:.2f}, step_cnt: {:2d}, terminated: {}".format(
                    action, travel_time, self.current_time, self.orbit_config.total_time,
                    0, self.current_coverage * 100, self.step_cnt, terminated
                )
            )
            return observation, reward, terminated, info

        self.current_points_cloud = np.append(
            self.current_points_cloud, increase_points_cloud, axis=0
        )
        increase_points_tensor = increase_points_cloud[np.newaxis, :, :].astype(
            np.float32
        )
        increase_points_tensor = torch.tensor(increase_points_tensor).to(self.DEVICE)
        
        # ============================================================================
        # STEP 7: COMPUTE COVERAGE INCREMENT
        # ============================================================================
        # Measure how many new points from this viewpoint cover previously unobserved
        # regions of the 3D model (measured against the ground truth).
        dist1, dist2 = ChamferDistanceFunction.apply(
            increase_points_tensor, self.ground_truth_tensor
        )
        dist2 = dist2.cpu().numpy()
        cover_flag = dist2 < self.COVERAGE_THRESHOLD
        cover_add = np.sum(cover_flag == True)
        cover_add = cover_add / self.ground_truth_points_cloud_size
        self.current_coverage += cover_add
        self.coverage_add = cover_add

        # ============================================================================
        # STEP 8: UPDATE COVERAGE STATE
        # ============================================================================
        # Remove newly covered ground truth points so they aren't counted again
        # (points already observed are excluded from future coverage calculations).
        # Points that have already been covered will no longer be counted repeatedly
        self.current_points_cloud_from_gt = np.append(
            self.current_points_cloud_from_gt,
            self.ground_truth_points_cloud[cover_flag[0, :]],
            axis=0,
        )
        self.ground_truth_points_cloud = self.ground_truth_points_cloud[
            ~cover_flag[0, :]
        ]
        self.ground_truth_tensor = self.ground_truth_points_cloud[
            np.newaxis, :, :
        ].astype(np.float32)
        self.ground_truth_tensor = torch.tensor(self.ground_truth_tensor).to(
            self.DEVICE
        )

        # ============================================================================
        # STEP 9: CALCULATE REWARD
        # ============================================================================
        # Reward combines:
        # 1. Coverage gained (cover_add) - positive reward for new observations
        # 2. Travel cost (travel_time) - negative reward for time spent
        # 3. Balance: agent learns efficiency = coverage_gain / travel_time
        # 
        # Example:
        # - Cover 5% in 0.1 time units: reward = 0.5 - 1.0*0.1 = 0.4
        # - Cover 1% in 0.5 time units: reward = 0.1 - 1.0*0.5 = -0.4
        # Agent learns second option is worse (wasted time)
        reward = self._get_reward(cover_add, action, travel_time=travel_time)
        
        # ============================================================================
        # STEP 10: BUILD NEXT STATE OBSERVATION
        # ============================================================================
        observation = self._get_observation_space()
        terminated = self._get_terminated()
        info = self._get_info()
        
        # Add travel time info to the info dictionary for monitoring
        info['travel_time'] = travel_time
        info['mission_time'] = self.current_time
        info['mission_time_horizon'] = self.orbit_config.total_time

        if cover_add == 1:
            self.logger.error("cover_add is 1")
            self._get_debug_info()
        self.logger.debug(
            "[step] SUCCESS  | action: {:2d}, travel_time: {:.6f}, time: {:.6f}/{:.6f}, cover_add: {:.2f}, "
            "cur_cover: {:.2f}, step_cnt: {:2d}, terminated: {}".format(
                action,
                travel_time,
                self.current_time,
                self.orbit_config.total_time,
                cover_add * 100,
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
        if self.view_state[action] == 1:
            return 0
        new_points_cloud = self.shapenet_reader.get_point_cloud_by_view_id(action)
        new_points_cloud_tensor = new_points_cloud[np.newaxis, :, :].astype(np.float32)
        new_points_cloud_tensor = torch.tensor(new_points_cloud_tensor).to(self.DEVICE)
        cur_points_cloud_tensor = self.current_points_cloud[np.newaxis, :, :].astype(
            np.float32
        )
        cur_points_cloud_tensor = torch.tensor(cur_points_cloud_tensor).to(self.DEVICE)
        dist1, dist2 = ChamferDistanceFunction.apply(
            new_points_cloud_tensor, cur_points_cloud_tensor
        )
        dist1 = dist1.cpu().numpy()
        overlay_flag = dist1 < self.COVERAGE_THRESHOLD

        increase_points_cloud = new_points_cloud[~overlay_flag[0, :]]
        if increase_points_cloud.shape[0] == 0:
            return 0

        increase_points_tensor = increase_points_cloud[np.newaxis, :, :].astype(
            np.float32
        )
        increase_points_tensor = torch.tensor(increase_points_tensor).to(self.DEVICE)
        dist1, dist2 = ChamferDistanceFunction.apply(
            increase_points_tensor, self.ground_truth_tensor
        )
        dist2 = dist2.cpu().numpy()
        cover_flag = dist2 < self.COVERAGE_THRESHOLD
        # cover_flag = cover_flag[0, :]
        cover_add = np.sum(cover_flag == True)
        cover_add = cover_add / self.ground_truth_points_cloud_size
        return cover_add

    def reset(self, init_step=-1):
        self.shapenet_reader.get_next_model()
        self.view_state = np.zeros(self.view_num, dtype=np.int32)
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
        self.view_state[self.current_view] = 1
        self.ground_truth_points_cloud = self.shapenet_reader.ground_truth
        self.ground_truth_points_cloud_size = self.ground_truth_points_cloud.shape[0]
        self.ground_truth_tensor = self.shapenet_reader.ground_truth[
            np.newaxis, :, :
        ].astype(np.float32)
        self.ground_truth_tensor = torch.tensor(self.ground_truth_tensor).to(
            self.DEVICE
        )
        self.current_coverage = self._caculate_current_coverage()
        self.coverage_add = self.current_coverage
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
        
        observation = self._get_observation_space()
        info = self._get_info()
        self.logger.debug("[reset] pass, init step: {}".format(self.current_view))
        return observation

    def close(self):
        pass

    def render(sellf):
        pass

    def _caculate_current_coverage(self):
        cur_points_cloud_tensor = self.current_points_cloud[np.newaxis, :, :].astype(
            np.float32
        )
        cur_points_cloud_tensor = torch.tensor(cur_points_cloud_tensor).to(self.DEVICE)
        dist1, dist2 = ChamferDistanceFunction.apply(
            cur_points_cloud_tensor, self.ground_truth_tensor
        )
        dist2 = dist2.cpu().numpy()
        cover_flag = dist2 < self.COVERAGE_THRESHOLD
        # cover_flag = cover_flag[0, :]
        coverage = np.sum(cover_flag == True)
        coverage = coverage / self.ground_truth_points_cloud_size

        # Points that have already been covered will no longer be counted repeatedly
        self.current_points_cloud_from_gt = self.ground_truth_points_cloud[
            cover_flag[0, :]
        ]
        self.ground_truth_points_cloud = self.ground_truth_points_cloud[
            ~cover_flag[0, :]
        ]
        self.ground_truth_tensor = self.ground_truth_points_cloud[
            np.newaxis, :, :
        ].astype(np.float32)
        self.ground_truth_tensor = torch.tensor(self.ground_truth_tensor).to(
            self.DEVICE
        )
        return coverage

    def _get_reward(self, cover_add, action, travel_time=0.0):
        """
        Calculate reward combining coverage gain and travel time cost.
        
        Reward = coverage_reward - time_cost_weight * travel_time
        
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
        final_reward = coverage_reward - time_penalty
        
        self.logger.debug(
            f"[REWARD] action={action:2d}, coverage_reward={coverage_reward:7.4f}, "
            f"time_penalty={time_penalty:7.4f}, final={final_reward:7.4f}"
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
        return False

    def _get_info(self):
        return {
            "cur_points_cloud": self.ground_truth_points_cloud,
            "model_name": self.model_name,
            "current_coverage": self.current_coverage,
        }

    def _get_debug_info(self):
        self.logger.info(
            "model name:{}, action history: {}".format(
                self.model_name, self.action_history
            )
        )