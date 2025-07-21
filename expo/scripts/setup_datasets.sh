#!/bin/bash
# Dataset Setup Script for UR5 YCB Object Manipulation Experiments

set -e

echo "🚀 Setting up datasets for UR5 YCB object manipulation experiments"

# Create datasets directory
mkdir -p datasets

# Function to create dataset directory structure
create_dataset_structure() {
    local env_name=$1
    local description=$2
    
    echo "📁 Creating dataset structure for $env_name"
    mkdir -p "datasets/$env_name"
    
    # Create metadata file
    cat > "datasets/$env_name/dataset_info.yaml" << EOF
environment: $env_name
description: $description
created_date: $(date -Iseconds)
task_type: manipulation
ycb_objects: []
world_indices: []
collection_status: pending
EOF
    
    echo "✅ Dataset structure created for $env_name"
}

# Create dataset structures for all environments
create_dataset_structure "MujocoUR5ePick" "YCB object picking tasks (cracker_box, pudding_box, potted_meat_can)"
create_dataset_structure "MujocoUR5eCable" "Cable manipulation and positioning tasks"
create_dataset_structure "MujocoUR5eCabinet" "Cabinet door opening/closing tasks"
create_dataset_structure "MujocoUR5eDoor" "Door opening manipulation tasks"

# Create results directory
mkdir -p results
echo "📊 Results directory created"

# Create plots directory
mkdir -p results/plots
echo "📈 Plots directory created"

echo ""
echo "🎯 Dataset setup completed!"
echo ""
echo "Next steps:"
echo "1. Collect demonstration data:"
echo "   python data_collector.py --all_tasks --episodes 50"
echo ""
echo "2. Run experiments:"
echo "   python run_experiment.py"
echo ""
echo "3. Analyze results:"
echo "   python result_analyzer.py --plot --comprehensive_report"