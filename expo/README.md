# 🤖 UR5 YCB Object Manipulation Automated Experiment System

This directory contains a comprehensive automated experiment system for UR5 robotic simulation environments using the RoboManipBaselines platform, specifically designed for **YCB object manipulation tasks** including pick, push, and place operations.

## 🚀 Overview

This system enables automated batch testing of multiple imitation learning policies across various UR5 simulation environments, with automatic success/failure recording and comprehensive result analysis focused on **YCB_sim objects** and basic manipulation primitives.

### 🎯 Key Features
- **YCB Object Support**: Full support for cracker_box, pudding_box, potted_meat_can
- **Multiple Task Types**: Pick, push, place, and complex manipulation tasks
- **Automated Success Recording**: Reward-based success detection (≥1.0 threshold)
- **Comprehensive Analysis**: Statistical analysis, visualizations, and performance reports
- **Batch Policy Testing**: Simultaneous evaluation of multiple imitation learning policies

### 📦 Supported YCB Objects
- **cracker_box**: 0.16×0.21×0.07m, 0.411kg
- **pudding_box**: 0.095×0.095×0.11m, 0.187kg  
- **potted_meat_can**: 0.102×0.084×0.054m, 0.370kg

### 🌍 Supported Environments
- **MujocoUR5ePick**: YCB object picking tasks
- **MujocoUR5eCable**: Cable manipulation tasks
- **MujocoUR5eCabinet**: Cabinet door manipulation (push-based)
- **MujocoUR5eDoor**: Door opening tasks (push-based)

### 🧠 Supported Policies
- **Act**: Action Chunking Transformer
- **Mlp**: Multi-layer Perceptron
- **Sarnn**: Spatial Attention RNN
- **DiffusionPolicy**: Diffusion-based behavior cloning

## 📁 Complete Directory Structure
```
expo/
├── README.md                           # This documentation
├── run_experiment.py                   # Main experiment runner
├── data_collector.py                   # YCB data collection utilities
├── result_analyzer.py                  # Comprehensive result analysis
├── config/
│   ├── experiment_config.yaml          # Main experiment configuration
│   ├── train/                          # Training configurations
│   │   ├── act_mujocoUR5ePick.txt
│   │   └── mlp_mujocoUR5ePick.txt
│   └── rollout/                        # Rollout configurations
│       ├── mujocoUR5ePick.txt
│       └── mujocoUR5eCable.txt
├── scripts/
│   └── setup_datasets.sh               # Dataset preparation script
├── datasets/                           # YCB demonstration data (created)
└── results/                           # Experiment results (auto-generated)
    ├── experiment_log.yaml
    ├── experiment_summary.yaml
    ├── experiment_report.md
    └── plots/
```

## 🚀 Quick Start

### 1. Setup
```bash
# Navigate to expo directory
cd expo/

# Setup dataset directories
chmod +x scripts/setup_datasets.sh
./scripts/setup_datasets.sh
```

### 2. Data Collection (Optional)
```bash
# Collect YCB object demonstrations
python data_collector.py --all_tasks --episodes 50

# Collect for specific environment
python data_collector.py --env MujocoUR5ePick --episodes 30 --task_type pick
```

### 3. Run Experiments
```bash
# Run all YCB manipulation experiments
python run_experiment.py

# Run specific policy on YCB picking
python run_experiment.py --policy Act --env MujocoUR5ePick

# Run specific task type
python run_experiment.py --task_type pick
```

### 4. Analyze Results
```bash
# Generate comprehensive analysis
python result_analyzer.py --plot --comprehensive_report

# YCB-specific analysis
python result_analyzer.py --ycb_analysis --save_plots

# Policy comparison
python result_analyzer.py --compare_policies
```

### 4. Analyze Results
```bash
# Generate comprehensive analysis
python result_analyzer.py --plot --comprehensive_report

# YCB-specific analysis
python result_analyzer.py --ycb_analysis --save_plots

# Policy comparison
python result_analyzer.py --compare_policies
```

## 📊 Output Results

### YAML Results (`result_Policy_Environment.yaml`)
```yaml
success: [true, false, true, ...]  # Success/failure per episode
reward: [1.0, 0.2, 1.0, ...]      # Reward per episode
duration: [15.2, 30.0, 12.5, ...] # Duration per episode
```

### Summary Report (`experiment_summary.yaml`)
```yaml
Act:
  MujocoUR5ePick:
    success_rate: 0.85
    avg_reward: 0.92
    avg_duration: 18.5
    task_type: "pick"
    total_episodes: 10
```

### Comprehensive Analysis Report (`comprehensive_analysis_report.md`)
- Detailed performance metrics by policy and environment
- YCB object-specific analysis
- Task type comparisons
- Statistical summaries and recommendations

### Visualizations (`results/plots/`)
- Success rate heatmaps
- Task performance comparisons
- YCB object-specific analysis charts
- Reward distribution plots
- Duration analysis graphs

## 🎯 Success Criteria

Each environment has built-in success criteria:

| Environment | Success Condition | YCB Objects | Task Type |
|-------------|------------------|-------------|-----------|
| MujocoUR5ePick | Object lifted and held (reward ≥ 1.0) | cracker_box, pudding_box, potted_meat_can | pick |
| MujocoUR5eCable | Cable properly positioned (reward ≥ 1.0) | - | manipulation |
| MujocoUR5eCabinet | Door opened ≥ target angle (reward ≥ 1.0) | - | push |
| MujocoUR5eDoor | Door opened ≥ 45° (reward ≥ 1.0) | - | push |

## 🔧 Configuration Details

### Main Configuration (`config/experiment_config.yaml`)
```yaml
policies: ["Act", "Mlp", "Sarnn", "DiffusionPolicy"]
environments:
  MujocoUR5ePick:
    dataset_location: "./datasets/MujocoUR5ePick"
    world_indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    success_threshold: 1.0
    task_type: "pick"
    ycb_objects: ["cracker_box", "pudding_box", "potted_meat_can"]
```

### Training Configuration Example (`config/train/act_mujocoUR5ePick.txt`)
```
--num_epochs 1000
--batch_size 32
--lr 1e-4
--chunk_size 100
--kl_weight 10
--hidden_dim 512
--dim_feedforward 3200
--camera_names front hand
--state_keys measured_joint_pos
--action_keys command_joint_pos
```

## 🔍 Advanced Features

### YCB Object Analysis
- Per-object success rate analysis
- Object-specific performance metrics
- Physical property correlation analysis

### Task-Specific Metrics
- **Pick Tasks**: Grasp success, lift height, hold stability
- **Push Tasks**: Contact force, displacement accuracy
- **Manipulation Tasks**: Multi-step operation success

### Automated Error Recovery
- Experiment retry on transient failures
- Detailed error logging and classification
- Graceful timeout handling

### Comprehensive Visualization
- Interactive performance dashboards
- Success rate heatmaps by policy×environment
- Task type comparison charts
- Reward distribution analysis

## 🐛 Troubleshooting

### Common Issues
1. **Missing Dataset**: Run `./scripts/setup_datasets.sh`
2. **Import Errors**: Ensure RoboManipBaselines is installed: `pip install -e .`
3. **CUDA Memory Error**: Reduce batch size in training configs
4. **Environment Not Found**: Check environment name spelling in config

### Debug Commands
```bash
# Test single experiment
python run_experiment.py --policy Act --env MujocoUR5ePick

# Collect sample data
python data_collector.py --env MujocoUR5ePick --episodes 5

# Verify configuration
python -c "import yaml; print(yaml.safe_load(open('config/experiment_config.yaml')))"
```

## 🤝 Contributing

To add new environments or policies:

1. **New Environment**: Add to `config/experiment_config.yaml` and create corresponding train/rollout configs
2. **New Policy**: Add to policies list and create training configurations
3. **New Task Type**: Extend task-specific analysis in `result_analyzer.py`

## 📝 Citation

If you use this experiment system, please cite RoboManipBaselines:

```bibtex
@software{RoboManipBaselines_GitHub2024,
  author = {Murooka, Masaki and Motoda, Tomohiro and Nakajo, Ryoichi},
  title = {{RoboManipBaselines}},
  url = {https://github.com/isri-aist/RoboManipBaselines},
  version = {1.0.0},
  year = {2024},
  month = dec,
}
```
