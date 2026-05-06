# RL-NBV Codebase Analysis

## Overview
RL-NBV is a reinforcement learning system for Next Best View (NBV) planning in 3D reconstruction. The system learns optimal viewpoint selection strategies for efficiently exploring 3D objects using Deep Q-Networks (DQN) with PointNet++ feature extraction.

## Architecture Summary

### Core Components
- **Environment**: `PointCloudNextBestViewEnv` - Gym environment for NBV task
- **Neural Network**: PointNet++ feature extractor + DQN policy
- **Data Loader**: `ShapenetReader` - ShapeNet dataset handler
- **Training**: Stable Baselines3 DQN with custom callbacks

### Key Specifications
- **Action Space**: Discrete (33 predefined viewpoints)
- **State Space**: Point cloud observations + visited view state
- **Feature Dimensions**: 128 total (95 from FC layers + 33 from view state)
- **Network**: PointNet++ (3 set abstraction layers) → DQN
- **Reward**: Coverage gain - travel time cost - fuel cost

## Detailed Component Analysis

### 1. Data Pipeline (`shapenet_reader.py`)

**Purpose**: Load and manage ShapeNet 3D model data

**Data Structure**:
```
dataset/
├── model_01/
│   ├── 0.pcd, 1.pcd, ..., 32.pcd (33 viewpoint clouds)
│   └── model.pcd (complete ground truth)
└── model_02/ ...
```

**Key Features**:
- Caches all viewpoint point clouds in memory
- Supports model switching during training
- Provides ground truth for coverage calculation
- Handles PCD file format using Open3D

**Methods**:
- `set_model_id()`: Switch between 3D models
- `get_point_cloud_by_view_id()`: Get specific viewpoint observation
- `get_next_model()`: Cycle to next model for training variety

### 2. Environment (`rl_nbv_env.py`)

**Purpose**: Implement NBV task as Gym environment with orbital mechanics

**State Representation**:
```python
observation = {
    "current_point_cloud": shape(3, N),  # Accumulated observations
    "view_state": shape(33,)              # Binary visited views
}
```

**Action Space**: 33 discrete viewpoints (predefined orbital positions)

**Reward Function**:
```python
reward = coverage_gain - time_cost_weight * travel_time - delta_v_weight * fuel_cost
```

**Advanced Features**:
- **Orbital Mechanics**: Realistic travel time calculations between viewpoints
- **Sun Position**: Dynamic lighting affecting visibility
- **Fuel Constraints**: Delta-V budget for mission planning
- **Coverage Calculation**: Chamfer distance for point cloud matching

**Key Methods**:
- `step()`: Execute action, calculate reward, update state
- `reset()`: Initialize new episode with random model
- `try_step()`: Simulate action without state change (for planning)

### 3. Neural Network Architecture (`pointnet2_cls_ssg.py`)

**PointNet++ Feature Extractor**:
```python
class PointNetFeatureExtraction(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        # SA1: 512 points, radius=0.2, 32 samples
        # SA2: 128 points, radius=0.4, 64 samples  
        # SA3: Global features (all points)
        # FC: 1024→512→256→95
        # Concat: 95 + 33(view_state) = 128 features
```

**Architecture Flow**:
1. **Input**: (B, 3, N) point cloud + (B, 33) view state
2. **Set Abstraction 1**: Local features (512 points, radius=0.2)
3. **Set Abstraction 2**: Mid-level features (128 points, radius=0.4)
4. **Set Abstraction 3**: Global features (all points)
5. **Fully Connected**: 1024→512→256→95 dimensions
6. **Concatenation**: 95 features + 33 view state = **128 total**
7. **Output**: Feature vector for DQN Q-value calculation

**DQN Integration**:
- Uses Stable Baselines3 `MultiInputPolicy`
- Q-network: 128 features → 33 Q-values (one per viewpoint)
- Target network for stable training
- ε-greedy exploration during training

### 4. Training Pipeline (`train.py`)

**Training Configuration** (`config.yaml`):
```yaml
training:
  dqn:
    learning_rate: 0.001
    batch_size: 128
    buffer_size: 100000
    total_steps: 500000
    gamma: 0.1  # Low discount factor for myopic planning
  pretrained:
    is_transform: 1  # Transfer learning from classification
    model_path: ./models/pretrained/pointnet2_ssg_wo_normals/checkpoints/best_model.pth
```

**Key Features**:
- **Transfer Learning**: Initializes PointNet from pretrained classification model
- **Vectorized Environment**: 8 parallel workers for efficient training
- **Custom Callbacks**: Periodic evaluation and best model saving
- **Replay Buffer**: Experience replay for stable learning
- **GPU Optimization**: CUDA memory management and profiling

**Training Loop**:
1. Initialize environments (train/verify/test)
2. Create DQN with PointNet feature extractor
3. Load pretrained weights (optional)
4. Train with custom callbacks
5. Evaluate coverage progress
6. Save best model checkpoints

### 5. State Transition System (`envs/state_transition/`)

**Purpose**: Modular orbital mechanics and visibility calculations

**Modules**:
- `travel_time.py`: Orbital dynamics, Lambert solver
- `visibility.py`: Line-of-sight and lighting calculations
- `coverage.py`: Point cloud coverage tracking
- `reward.py`: Reward computation components
- `orchestrator.py`: Step execution coordination
- `sun_position.py`: Dynamic sun position modeling

**Key Features**:
- **Orbital Config**: Target orbit parameters (radius, gravity, period)
- **Travel Time Matrix**: Precomputed times between all viewpoint pairs
- **Delta-V Matrix**: Fuel cost calculations for maneuvers
- **Visibility Maps**: Points visible from each viewpoint under current lighting

### 6. Utilities and Tools

**Distance Metrics** (`distance/`):
- `chamfer_distance.py`: Point cloud similarity (PyTorch implementation)
- `emd_module.py`: Earth Mover's Distance for alternative matching

**Optimization** (`optim/`):
- `adamw.py`: AdamW optimizer implementation

**Callbacks** (`custom_callback.py`):
- Periodic evaluation on verification/test sets
- Best model saving based on coverage performance
- Training progress monitoring

**Tools**:
- Dataset splitting utilities
- Coverage evaluation scripts
- Ground truth generation
- Rendering pipelines for data preprocessing

## Data Flow Summary

```
Raw ShapeNet Data → ShapenetReader → Environment Observation → 
PointNet++ Feature Extraction (128-dim) → DQN Q-values → 
Action Selection → Environment Step → New Observation → (repeat)

Goal: Learn policy maximizing coverage while minimizing travel time and fuel costs
```

## Key Hyperparameters

### Environment
- **Viewpoints**: 33 predefined orbital positions
- **Max Steps**: 11 per episode
- **Coverage Threshold**: 97% for episode termination
- **Fuel Budget**: 50.0 delta-V units
- **Time Weight**: 1.0 (travel time penalty)
- **Fuel Weight**: 1.0 (delta-V penalty)

### Network
- **Point Cloud Dim**: 1024 points (resampled)
- **Feature Dim**: 128 (95 + 33)
- **Normalization**: Enabled (-1 to 1 range)
- **Batch Size**: 128
- **Learning Rate**: 0.001 (with linear scheduling)

### Training
- **Total Steps**: 500,000
- **Buffer Size**: 100,000 experiences
- **Exploration**: 50% of training, final ε=0.2
- **Evaluation**: Every 10,000 steps
- **Checkpoint**: Every 10,000 steps

## Performance Metrics

### Coverage Evaluation
- Average coverage per step across test set
- Optimal view count to reach 97% coverage
- Success rate (models reaching target coverage)

### Training Metrics
- Episode rewards
- Coverage progress curves
- Action selection patterns
- Fuel consumption efficiency

## Usage Examples

### Training
```bash
python train.py --config config.yaml
```

### Testing
```bash
python test.py --config config.yaml --model_path final_model
```

### Benchmarking
```bash
python benchmark.py --config config.yaml
```

## File Structure
```
RL-NBV/
├── train.py              # Main training script
├── test.py               # Model evaluation
├── benchmark.py          # Performance comparison
├── config.yaml           # Training configuration
├── envs/                 # Environment modules
│   ├── rl_nbv_env.py     # Main Gym environment
│   ├── shapenet_reader.py # Data loader
│   └── state_transition/ # Orbital mechanics
├── models/               # Neural networks
│   ├── pointnet2_cls_ssg.py # PointNet++ architecture
│   ├── pointnet2_utils.py  # PointNet utilities
│   └── pretrained/         # Pretrained models
├── distance/             # Distance metrics
├── optim/                # Optimizers
├── tools/                # Data preprocessing
└── artefacts/            # Training outputs
```

## Key Insights

1. **Modular Design**: Clear separation between environment, network, and training components
2. **Realistic Physics**: Orbital mechanics and fuel constraints for practical applications
3. **Transfer Learning**: Leverages pretrained PointNet for better feature extraction
4. **Efficient Training**: Vectorized environments and GPU optimization
5. **Comprehensive Evaluation**: Multiple metrics and benchmarking tools

## Current State

- **Complete training pipeline** with DQN + PointNet++
- **Pretrained model integration** for transfer learning
- **Orbital mechanics simulation** with realistic constraints
- **Comprehensive evaluation framework**
- **Modular, extensible architecture**

The codebase represents a complete, production-ready system for reinforcement learning research in next best view planning with practical applications in satellite observation and 3D reconstruction.

## Continuous Control Mode

The system now supports both discrete and continuous control modes via configuration.

### Discrete Mode (Default)
- **Algorithm**: DQN (Deep Q-Network)
- **Action Space**: 33 discrete viewpoints
- **Feature Extractor**: PointNetFeatureExtraction (128-dim: 95 PC + 33 view_state)
- **Use Case**: Baseline, predefined view selection
- **Configuration**: Set `continuous_mode: false` and `algorithm: DQN` in config.yaml

### Continuous Mode
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Action Space**: 2 continuous dimensions (spherical coordinates: theta [0,π], phi [0,2π])
- **Feature Extractor**: PointNetFeatureExtractionContinuous (128-dim: 122 PC + 6 scalars)
- **Observation**: camera_position(3) + coverage(1) + fuel_remaining(1) + time_remaining(1)
- **Use Case**: Flexible trajectory optimization, realistic mission planning
- **Configuration**: Set `continuous_mode: true` and `algorithm: PPO` in config.yaml

### Mode Selection
Toggle between modes in `config.yaml`:
```yaml
environment:
  continuous_mode: true  # Enable continuous control

training:
  algorithm: PPO  # Use PPO instead of DQN
```

### Key Differences
- **Discrete**: Agent selects from 33 predefined viewpoints, tracks visited views
- **Continuous**: Agent outputs 3D direction vector, no view tracking, smooth trajectories
- **Rendering**: Continuous mode uses nearest-viewpoint approximation (can be upgraded to true continuous rendering)
- **Training**: PPO requires more steps (1M vs 500K) but handles continuous actions naturally
