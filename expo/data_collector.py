#!/usr/bin/env python3
"""
YCB Object Data Collector for UR5 Simulation

This script collects demonstration data for YCB object manipulation tasks
including pick, push, and place operations using teleoperation.
"""

import os
import sys
import argparse
import subprocess
import yaml
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))


class YCBDataCollector:
    """Data collector for YCB object manipulation tasks"""
    
    def __init__(self, output_dir: str = "./datasets"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load YCB object configuration
        self.ycb_objects = {
            'cracker_box': {'size': [0.16, 0.21, 0.07], 'weight': 0.411},
            'pudding_box': {'size': [0.095, 0.095, 0.11], 'weight': 0.187},
            'potted_meat_can': {'size': [0.102, 0.084, 0.054], 'weight': 0.370}
        }
    
    def collect_demonstrations(self, env_name: str, episodes: int = 50, 
                             task_type: str = "pick", input_device: str = "keyboard") -> bool:
        """Collect demonstration data for specified environment and task"""
        
        print(f"🎮 Starting data collection for {env_name}")
        print(f"📊 Task Type: {task_type}")
        print(f"📦 Episodes: {episodes}")
        print(f"🎯 Input Device: {input_device}")
        
        # Create environment-specific output directory
        env_output_dir = self.output_dir / env_name
        env_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare teleoperation command
        teleop_script = Path(__file__).parent.parent / "bin" / "Teleop.py"
        
        if not teleop_script.exists():
            print(f"❌ Teleop script not found: {teleop_script}")
            return False
        
        # Build command arguments
        cmd = [
            "python", str(teleop_script),
            env_name,
            "--input_device", input_device,
            "--output_dir", str(env_output_dir)
        ]
        
        # Add world indices based on environment
        world_indices = self._get_world_indices(env_name)
        if world_indices:
            cmd.extend(["--world_idx_list"] + [str(i) for i in world_indices])
        
        # Add episode count
        if episodes > 0:
            cmd.extend(["--episode_count", str(episodes)])
        
        # Add task-specific parameters
        task_params = self._get_task_parameters(task_type)
        for param, value in task_params.items():
            cmd.extend([f"--{param}", str(value)])
        
        try:
            print(f"🚀 Executing: {' '.join(cmd[:4])}...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode == 0:
                print(f"✅ Data collection completed for {env_name}")
                self._save_collection_metadata(env_name, task_type, episodes, env_output_dir)
                return True
            else:
                print(f"❌ Data collection failed for {env_name}")
                print(f"Error: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Data collection timed out for {env_name}")
            return False
        except Exception as e:
            print(f"💥 Exception during data collection: {e}")
            return False
    
    def collect_all_tasks(self, episodes_per_task: int = 50) -> Dict[str, bool]:
        """Collect data for all YCB manipulation tasks"""
        
        # Define task configurations
        tasks = {
            'MujocoUR5ePick': {'task_type': 'pick', 'episodes': episodes_per_task},
            'MujocoUR5eCable': {'task_type': 'manipulation', 'episodes': episodes_per_task // 2},
            'MujocoUR5eCabinet': {'task_type': 'push', 'episodes': episodes_per_task // 2},
            'MujocoUR5eDoor': {'task_type': 'push', 'episodes': episodes_per_task // 3}
        }
        
        results = {}
        total_tasks = len(tasks)
        
        print(f"🎯 Starting comprehensive YCB data collection")
        print(f"📋 Total tasks: {total_tasks}")
        print(f"📊 Total episodes: {sum(task['episodes'] for task in tasks.values())}")
        
        for i, (env_name, config) in enumerate(tasks.items(), 1):
            print(f"\n{'='*60}")
            print(f"📈 Task {i}/{total_tasks}: {env_name}")
            print(f"{'='*60}")
            
            success = self.collect_demonstrations(
                env_name=env_name,
                episodes=config['episodes'],
                task_type=config['task_type']
            )
            
            results[env_name] = success
            
            if success:
                print(f"✅ {env_name}: Data collection successful")
            else:
                print(f"❌ {env_name}: Data collection failed")
        
        # Generate summary report
        self._generate_collection_report(results, tasks)
        
        return results
    
    def _get_world_indices(self, env_name: str) -> List[int]:
        """Get appropriate world indices for environment"""
        world_config = {
            'MujocoUR5ePick': list(range(10)),      # 10 different YCB object configurations
            'MujocoUR5eCable': list(range(5)),      # 5 cable configurations
            'MujocoUR5eCabinet': list(range(6)),    # 6 cabinet configurations
            'MujocoUR5eDoor': list(range(4))        # 4 door configurations
        }
        
        return world_config.get(env_name, list(range(5)))
    
    def _get_task_parameters(self, task_type: str) -> Dict[str, Any]:
        """Get task-specific parameters"""
        params = {
            'pick': {
                'max_duration': 30.0,
                'auto_exit': True
            },
            'push': {
                'max_duration': 35.0,
                'auto_exit': True
            },
            'place': {
                'max_duration': 40.0,
                'auto_exit': True
            },
            'manipulation': {
                'max_duration': 45.0,
                'auto_exit': True
            }
        }
        
        return params.get(task_type, {'max_duration': 30.0, 'auto_exit': True})
    
    def _save_collection_metadata(self, env_name: str, task_type: str, episodes: int, output_dir: Path):
        """Save metadata about the data collection session"""
        metadata = {
            'collection_date': datetime.now().isoformat(),
            'environment': env_name,
            'task_type': task_type,
            'episodes_collected': episodes,
            'ycb_objects': self.ycb_objects if 'Pick' in env_name else None,
            'world_indices': self._get_world_indices(env_name),
            'output_directory': str(output_dir),
            'collection_parameters': self._get_task_parameters(task_type)
        }
        
        metadata_file = output_dir / "collection_metadata.yaml"
        with open(metadata_file, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)
        
        print(f"📝 Metadata saved to: {metadata_file}")
    
    def _generate_collection_report(self, results: Dict[str, bool], tasks: Dict[str, Dict]):
        """Generate comprehensive collection report"""
        report_file = self.output_dir / "collection_report.md"
        
        with open(report_file, 'w') as f:
            f.write("# YCB Object Manipulation Data Collection Report\n\n")
            f.write(f"**Collection Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Collection Summary\n\n")
            f.write("| Environment | Task Type | Episodes | Status |\n")
            f.write("|-------------|-----------|----------|--------|\n")
            
            total_episodes = 0
            successful_tasks = 0
            
            for env_name, success in results.items():
                task_config = tasks.get(env_name, {})
                task_type = task_config.get('task_type', 'unknown')
                episodes = task_config.get('episodes', 0)
                status = "✅ Success" if success else "❌ Failed"
                
                f.write(f"| {env_name} | {task_type} | {episodes} | {status} |\n")
                
                total_episodes += episodes
                if success:
                    successful_tasks += 1
            
            f.write(f"\n**Total Episodes Collected:** {total_episodes}\n")
            f.write(f"**Successful Tasks:** {successful_tasks}/{len(results)}\n\n")
            
            f.write("## YCB Objects Used\n\n")
            for obj_name, specs in self.ycb_objects.items():
                f.write(f"- **{obj_name}**: {specs['size']} m, {specs['weight']} kg\n")
            
            f.write("\n## Task Descriptions\n\n")
            f.write("- **Pick Tasks**: Grasping and lifting YCB objects\n")
            f.write("- **Push Tasks**: Object manipulation through contact forces\n")
            f.write("- **Place Tasks**: Precise object placement\n")
            f.write("- **Manipulation Tasks**: Complex multi-step operations\n")
        
        print(f"📋 Collection report saved to: {report_file}")


def main():
    parser = argparse.ArgumentParser(description="Collect YCB object manipulation demonstration data")
    parser.add_argument("--env", type=str, help="Specific environment to collect data for")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes to collect")
    parser.add_argument("--task_type", type=str, choices=['pick', 'push', 'place', 'manipulation'],
                       default='pick', help="Type of manipulation task")
    parser.add_argument("--input_device", type=str, default="keyboard", 
                       choices=['keyboard', 'joystick', 'spacemouse'], help="Input device for teleoperation")
    parser.add_argument("--output_dir", type=str, default="./datasets", help="Output directory for datasets")
    parser.add_argument("--all_tasks", action="store_true", help="Collect data for all YCB manipulation tasks")
    
    args = parser.parse_args()
    
    collector = YCBDataCollector(args.output_dir)
    
    if args.all_tasks:
        # Collect data for all tasks
        results = collector.collect_all_tasks(args.episodes)
        
        # Print summary
        print(f"\n{'='*60}")
        print("📊 DATA COLLECTION SUMMARY")
        print(f"{'='*60}")
        successful = sum(1 for success in results.values() if success)
        print(f"Successful tasks: {successful}/{len(results)}")
        
    elif args.env:
        # Collect data for specific environment
        success = collector.collect_demonstrations(
            env_name=args.env,
            episodes=args.episodes,
            task_type=args.task_type,
            input_device=args.input_device
        )
        
        if success:
            print(f"✅ Data collection completed successfully for {args.env}")
        else:
            print(f"❌ Data collection failed for {args.env}")
    
    else:
        print("Please specify --env for specific environment or --all_tasks for comprehensive collection")


if __name__ == "__main__":
    main()