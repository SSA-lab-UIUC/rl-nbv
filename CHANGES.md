# Continuous Control Redesign - Changes Documentation

This document details all changes made to transition the RL-NBV system from discrete DQN-based view selection to continuous PPO-based trajectory optimization.

## Overview

**Goal**: Enable continuous control mode alongside existing discrete mode through configuration-based switching.

**Approach**: In-place modifications to existing files using configuration flags, maintaining full backward compatibility.

---

## File-by-File Changes

### 1. envs/rl_nbv_env.py

**Changes Made**:
- Added `continuous_mode` parameter to `__init__()` method
- Added conditional action space setup (Discrete vs Box(2,) with spherical coordinates)
- Added conditional observation space setup (with/without view_state, with/without scalars)
- Implemented `_step_continuous()` method for continuous action handling using spherical coordinates
- Implemented `_step_discrete()` method (renamed from existing `step()`)
- Added helper method `random_position_on_sphere()` (module-level function) using spherical coordinates
- Added helper method `_get_points_from_position()` for nearest-view approximation
- Added helper method `_update_coverage()` for coverage calculation using persistent coverage map
- Added `_get_observation_space_continuous()` method
- Added `_get_observation_space_discrete()` method (renamed from existing logic)
- Added `_get_info_continuous()` method
- Added `_get_info_discrete()` method (renamed from existing logic)
- Updated `step()` to dispatch based on mode
- Updated `reset()` for continuous initialization
- **Bug Fixes**:
  - Fixed `_update_coverage()` to use persistent coverage map instead of only new points
  - Fixed `_update_coverage()` to use `ChamferDistanceFunction.apply()` instead of class instantiation
  - Fixed `_update_coverage()` to use cached `self._canonical_tensor` instead of rebuilding every step
  - Fixed `_step_continuous()` to use `_get_reward()` for consistent reward calculation
  - Fixed `_step_continuous()` to use `_get_terminated()` for consistent termination checks
  - Fixed `_step_continuous()` to cap `current_points_cloud` size to prevent unbounded growth
  - Fixed CWDynamics re-instantiation by initializing once in `__init__`
  - Fixed `get_travel_time` import by moving to top of file instead of per-step import
  - Fixed `__init__()` to initialize `_coverage_map` for continuous mode
  - Fixed `reset()` to reset `_coverage_map` after model size is updated
  - Fixed `reset()` to load initial point cloud after `_canonical_points` is updated
  - Fixed `reset()` to reset `coverage_add` in continuous mode
  - Fixed `reset()` to reset `current_points_cloud_from_gt` in continuous mode
  - Fixed `_initialize_state_transition_for_current_model()` to cache `_canonical_tensor` for continuous mode
  - Fixed `_initialize_state_transition_for_current_model()` to return early in continuous mode
  - Fixed `_get_info_continuous()` to use `self._canonical_points` instead of stale `ground_truth_points_cloud`
  - Fixed `_get_terminated()` to check time limit in continuous mode
  - Added guard in `try_step()` to raise error in continuous mode

**Reason for Changes**:
- Environment is the core component that needs to handle both discrete and continuous action/observation spaces
- Continuous mode requires different action representation (3D direction vector vs discrete index)
- Continuous mode needs additional state information (camera position, fuel, time) in observations
- Nearest-view approximation provides a practical way to render from continuous positions without a full renderer
- Mode dispatching ensures backward compatibility

**Review Order**: Start here - this is the foundation that other files depend on.

---

### 2. models/pointnet2_cls_ssg.py

**Changes Made**:
- Added PointNetFeatureExtractionContinuous class for PPO continuous control mode.
- Processes point cloud and concatenates 6 scalar features (camera position, coverage, fuel remaining, time remaining) to produce a 128-dimensional feature vector.
- Added tanh activation to bound PointNet features to [-1, 1] for scale consistency with scalar features.

**Reason for Changes**:
- PPO requires a different feature extractor than DQN
- Continuous mode observations include scalar state information instead of view_state
- Feature vector feeds into PPO's actor and critic networks
- Tanh activation prevents PointNet features from dominating scalar features due to scale mismatch

**Review Order**: After environment - feature extractor needs to match observation space.

---

### 3. train.py

**Changes Made**:
- Added `ppo` config parsing in `config_to_args()`
- Added `algorithm` selection in `config_to_args()`
- Added `continuous_mode` parameter parsing
- Added PPO hyperparameters to args
- Updated `make_env()` to pass `continuous_mode` parameter
- Added conditional policy_kwargs based on algorithm
- Added conditional model creation (PPO vs DQN)
- Updated total_steps logic to use PPO steps when algorithm is PPO
- Added continuous_mode to verify_env and test_env creation

**Reason for Changes**:
- Training pipeline needs to support both DQN and PPO algorithms
- Configuration-based algorithm selection enables easy switching
- Environment needs continuous_mode flag for proper initialization
- Different hyperparameters for PPO vs DQN

**Review Order**: Third - orchestrates the training process using modified environment and feature extractor.

---

### 4. config.yaml

**Changes Made**:
- Added `continuous_mode: false` to environment section
- Added `algorithm: DQN` to training section
- Added `ppo` configuration section with hyperparameters
- Increased `max_step` from 11 to 30
- Added comments explaining mode selection

**Reason for Changes**:
- Configuration file is the single source of truth for mode selection
- PPO requires different hyperparameters than DQN
- Continuous episodes typically run longer, hence increased max_step
- Clear documentation helps users understand configuration options

**Review Order**: Fourth - configuration drives all other changes.

---

### 5. custom_callback.py

**Changes Made**:
- Added mode detection in `_init_callback()`
- Renamed existing `_caculate_average_coverage()` logic to `_evaluate_discrete()`
- Added `_evaluate_continuous()` method
- Updated `_caculate_average_coverage()` to dispatch based on mode

**Reason for Changes**:
- Callback needs to evaluate both discrete and continuous modes correctly
- Continuous mode doesn't use init_step (no predefined views)
- Different evaluation logic for continuous episodes

**Review Order**: Fifth - callback depends on environment mode.

---

### 6. CODEBASE_ANALYSIS.md

**Changes Made**:
- Added "Continuous Control Mode" section
- Documented discrete mode characteristics
- Documented continuous mode characteristics
- Added mode selection instructions
- Added key differences comparison

**Reason for Changes**:
- Documentation should reflect new capabilities
- Users need clear guidance on how to use continuous mode
- Comparison helps users understand trade-offs

**Review Order**: Sixth - documentation of all changes.

---

### 7. envs/state_transition/cw_utils.py

**Changes Made**:
- None

**Reason for No Changes**:
- Existing `CWDynamics.compute_delta_v()` already handles continuous positions
- Environment uses this directly in `_step_continuous()`
- No new functionality needed

**Review Order**: Last - verify no changes were needed.

### 8. random_coverage.py

**Changes Made**:
- Added `continuous_mode` parameter parsing in `config_to_args()`
- Updated `choose_action()` to handle continuous mode (no view_state logic)
- Added `continuous_mode` parameter to environment initialization

**Reason for Changes**:
- Random coverage script needs to support both modes
- Continuous mode doesn't have view_state, so unvisited view logic doesn't apply
- Environment needs continuous_mode flag for proper initialization

**Review Order**: After callback - utility script for baseline evaluation.

### 9. test.py

**Changes Made**:
- Added `continuous_mode` and `algorithm` parameter parsing in `config_to_args()`
- Added conditional model loading (PPO vs DQN) based on algorithm
- Added `continuous_mode` parameter to environment initialization

**Reason for Changes**:
- Test script needs to load and evaluate both DQN and PPO models
- Different feature extractors required for each algorithm
- Environment needs continuous_mode flag for proper initialization

**Review Order**: After random_coverage - main evaluation script.

---

## Step-by-Step Review Guide

### Phase 1: Foundation (Environment)
1. **Start with `envs/rl_nbv_env.py`**
   - Review the `continuous_mode` parameter addition
   - Check conditional action/observation space setup
   - Verify `_step_continuous()` implementation
   - Ensure helper methods are correct
   - Confirm mode dispatching in `step()` and `reset()`

### Phase 2: Feature Extraction
2. **Review `models/pointnet2_cls_ssg.py`**
   - Check `PointNetFeatureExtractionContinuous` class
   - Verify scalar dimensions match (6 total)
   - Ensure SA layers are identical to discrete version
   - Confirm output dimension is 128

### Phase 3: Training Pipeline
3. **Review `train.py`**
   - Check algorithm selection logic
   - Verify PPO config parsing
   - Ensure `make_env()` passes `continuous_mode`
   - Confirm conditional model creation
   - Check total_steps logic

### Phase 4: Configuration
4. **Review `config.yaml`**
   - Verify `continuous_mode` flag exists
   - Check `algorithm` selection
   - Confirm PPO hyperparameters are reasonable
   - Note the increased `max_step`

### Phase 5: Evaluation
5. **Review `custom_callback.py`**
   - Check mode detection logic
   - Verify `_evaluate_continuous()` implementation
   - Ensure dispatching works correctly

### Phase 6: Documentation
6. **Review `CODEBASE_ANALYSIS.md`**
   - Verify continuous mode section is accurate
   - Check configuration examples are correct
   - Ensure comparison is clear

### Phase 7: Verification
7. **Review `envs/state_transition/cw_utils.py`**
   - Confirm no changes were needed
   - Verify existing CW dynamics work for continuous positions

### Phase 8: Utility Scripts
8. **Review `random_coverage.py`**
   - Check continuous_mode parameter parsing
   - Verify choose_action() handles continuous mode
   - Ensure environment initialization includes continuous_mode

9. **Review `test.py`**
   - Check continuous_mode and algorithm parameter parsing
   - Verify conditional model loading (PPO vs DQN)
   - Ensure environment initialization includes continuous_mode

---

## Testing Checklist

### Discrete Mode (Regression Test)
- [ ] Set `continuous_mode: false` and `algorithm: DQN` in config.yaml
- [ ] Run training and verify it works as before
- [ ] Check that action space is Discrete(33)
- [ ] Verify observation space includes view_state

### Continuous Mode (New Feature)
- [ ] Set `continuous_mode: true` and `algorithm: PPO` in config.yaml
- [ ] Run training and verify it starts without errors
- [ ] Check that action space is Box(3,)
- [ ] Verify observation space includes scalars
- [ ] Confirm model uses PointNetFeatureExtractionContinuous
- [ ] Test that continuous step executes correctly
- [ ] Verify nearest-view approximation works

### Configuration Switching
- [ ] Toggle between modes and verify both work
- [ ] Check that config changes are properly parsed
- [ ] Ensure no conflicts between modes

---

## Key Design Decisions

1. **Spherical Coordinates**: Action space uses (theta, phi) instead of 3D Cartesian vector for natural sphere representation
   - theta: polar angle [0, π] (0 to 180 degrees)
   - phi: azimuthal angle [0, 2π] (0 to 360 degrees)
   - Ensures all actions are valid points on sphere surface
   - No need for normalization or handling invalid interior points
2. **128-dim Features**: Kept same dimension as discrete for easy comparison
3. **Nearest-View Approximation**: Practical solution for rendering without full continuous renderer
4. **Configuration-Based**: Single flag controls mode, easy to switch
5. **No New Files**: All changes in-place, maintains codebase structure
6. **Backward Compatible**: Discrete mode unchanged when flag is false

---

## Potential Future Improvements

1. **True Continuous Rendering**: Replace nearest-view approximation with ray tracing
2. **Orientation Control**: Add orientation deltas to action space if needed
3. **Curriculum Learning**: Start with discrete, transition to continuous
4. **Multi-Objective**: Add additional rewards for trajectory smoothness
5. **Advanced Dynamics**: Implement more sophisticated orbital maneuvers

---

## Summary

This redesign enables continuous control for the RL-NBV system while maintaining full backward compatibility with the existing discrete mode. All changes are configuration-based, allowing users to switch between modes by simply changing flags in `config.yaml`. The implementation follows the principle of minimal changes, modifying only what's necessary to support the new functionality.

**Total Files Modified: 9**
1. envs/rl_nbv_env.py - Core environment with continuous mode
2. models/pointnet2_cls_ssg.py - Continuous feature extractor
3. train.py - Training pipeline with PPO support
4. config.yaml - Configuration for both modes
5. custom_callback.py - Evaluation for both modes
6. CODEBASE_ANALYSIS.md - Documentation
7. envs/state_transition/cw_utils.py - No changes needed
8. random_coverage.py - Baseline evaluation support
9. test.py - Model evaluation support
