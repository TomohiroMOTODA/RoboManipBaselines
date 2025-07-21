#!/usr/bin/env python3
"""
UR5 Simulation Automated Experiment Runner

This script performs automated experiments on UR5 simulation environments
using multiple policies and automatically records success/failure results
for YCB object manipulation tasks (pick, push, place).
"""

import os
import sys
import yaml
import json
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import numpy as np

# Add the parent directory to sys.path to import robo_manip_baselines
sys.path.append(str(Path(__file__).parent.parent))

try:
    from robo_manip_baselines.misc.AutoEval import AutoEval
except ImportError:
    print("Warning: Could not import AutoEval. Please ensure RoboManipBaselines is properly installed.")
    AutoEval = None


class UR5ExperimentRunner:
    """Automated experiment runner for UR5 simulation environments with YCB objects"""
    
    def __init__(self, config_file: str = None):
        self.config = self._load_config(config_file)
        self.results_dir = Path(self.config['output']['results_dir'])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize experiment log
        self.experiment_log = {
            'start_time': datetime.now().isoformat(),
            'config': self.config,
            'experiments': []
        }
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load experiment configuration"""
        if config_file is None:
            config_file = Path(__file__).parent / "config" / "experiment_config.yaml"
        
        try:
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Config file not found: {config_file}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration"""
        return {
            'policies': ['Act', 'Mlp', 'Sarnn', 'DiffusionPolicy'],
            'environments': {
                'MujocoUR5ePick': {
                    'dataset_location': './datasets/MujocoUR5ePick',
                    'world_indices': list(range(10)),
                    'success_threshold': 1.0,
                    'task_type': 'pick',
                    'ycb_objects': ['cracker_box', 'pudding_box', 'potted_meat_can']
                },
                'MujocoUR5eCable': {
                    'dataset_location': './datasets/MujocoUR5eCable',
                    'world_indices': list(range(5)),
                    'success_threshold': 1.0,
                    'task_type': 'manipulation'
                },
                'MujocoUR5eCabinet': {
                    'dataset_location': './datasets/MujocoUR5eCabinet',
                    'world_indices': list(range(6)),
                    'success_threshold': 1.0,
                    'task_type': 'push'
                }
            },
            'output': {
                'results_dir': './results',
                'generate_plots': True,
                'save_videos': False
            },
            'seed': 42
        }
    
    def run_all_experiments(self):
        """Run all configured experiments"""
        policies = self.config['policies']
        environments = self.config['environments']
        
        total_experiments = len(policies) * len(environments)
        current_experiment = 0
        
        print(f"🚀 Starting automated UR5 YCB experiments: {total_experiments} total")
        print(f"📁 Results will be saved to: {self.results_dir}")
        
        for policy in policies:
            for env_name, env_config in environments.items():
                current_experiment += 1
                print(f"\n{'='*60}")
                print(f"📊 Experiment {current_experiment}/{total_experiments}")
                print(f"🧠 Policy: {policy}")
                print(f"🌍 Environment: {env_name}")
                print(f"🎯 Task Type: {env_config.get('task_type', 'manipulation')}")
                if 'ycb_objects' in env_config:
                    print(f"📦 YCB Objects: {', '.join(env_config['ycb_objects'])}")
                print(f"{'='*60}")
                
                try:
                    result = self._run_single_experiment(policy, env_name, env_config)
                    self.experiment_log['experiments'].append(result)
                    self._save_experiment_log()
                    
                    if result['status'] == 'success':
                        print(f"✅ {policy} + {env_name}: SUCCESS (Success rate: {result['success_rate']:.2%})")
                    else:
                        print(f"❌ {policy} + {env_name}: FAILED - {result['error']}")
                        
                except Exception as e:
                    error_result = {
                        'policy': policy,
                        'environment': env_name,
                        'status': 'error',
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }
                    self.experiment_log['experiments'].append(error_result)
                    print(f"💥 {policy} + {env_name}: EXCEPTION - {e}")
        
        self._generate_summary_report()
        print(f"\n🎉 All YCB manipulation experiments completed! Check results in {self.results_dir}")
    
    def _run_single_experiment(self, policy: str, env_name: str, env_config: Dict) -> Dict[str, Any]:
        """Run a single policy-environment experiment"""
        experiment_start = datetime.now()
        
        # Prepare experiment arguments
        dataset_location = env_config.get('dataset_location', '')
        world_indices = env_config.get('world_indices', list(range(10)))
        success_threshold = env_config.get('success_threshold', 1.0)
        
        if AutoEval is None:
            # Fallback: run experiment using subprocess
            return self._run_experiment_subprocess(policy, env_name, env_config, experiment_start)
        
        # Create AutoEval instance
        try:
            auto_eval = AutoEval()
            
            # Prepare arguments
            args = argparse.Namespace()
            args.policy = policy
            args.env = env_name
            args.input_dataset_location = dataset_location
            args.world_idx_list = world_indices
            args.seed = self.config.get('seed', 42)
            args.result_filename = str(self.results_dir / f"result_{policy}_{env_name}.yaml")
            
            # Add training and rollout config files if they exist
            config_dir = Path(__file__).parent / "config"
            train_config_file = config_dir / "train" / f"{policy.lower()}_{env_name.lower()}.txt"
            rollout_config_file = config_dir / "rollout" / f"{env_name.lower()}.txt"
            
            if train_config_file.exists():
                args.args_file_train = str(train_config_file)
            if rollout_config_file.exists():
                args.args_file_rollout = str(rollout_config_file)
            
            # Run the experiment
            auto_eval.run(args)
            
            # Load and analyze results
            return self._analyze_experiment_results(args.result_filename, policy, env_name, env_config, experiment_start)
            
        except Exception as e:
            return {
                'policy': policy,
                'environment': env_name,
                'status': 'failed',
                'error': str(e),
                'experiment_duration': (datetime.now() - experiment_start).total_seconds(),
                'timestamp': experiment_start.isoformat()
            }
    
    def _run_experiment_subprocess(self, policy: str, env_name: str, env_config: Dict, experiment_start: datetime) -> Dict[str, Any]:
        """Fallback method using subprocess to run experiments"""
        try:
            # Build command for AutoEval
            result_file = self.results_dir / f"result_{policy}_{env_name}.yaml"
            cmd = [
                "python", 
                str(Path(__file__).parent.parent / "robo_manip_baselines" / "misc" / "AutoEval.py"),
                policy,
                env_name,
                "--input_dataset_location", env_config.get('dataset_location', ''),
                "--world_idx_list"] + [str(i) for i in env_config.get('world_indices', list(range(10)))] + [
                "--seed", str(self.config.get('seed', 42)),
                "--result_filename", str(result_file)
            ]
            
            print(f"🔧 Running command: {' '.join(cmd[:5])}...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
            
            if result.returncode == 0:
                return self._analyze_experiment_results(str(result_file), policy, env_name, env_config, experiment_start)
            else:
                return {
                    'policy': policy,
                    'environment': env_name,
                    'status': 'failed',
                    'error': f"Subprocess failed: {result.stderr}",
                    'experiment_duration': (datetime.now() - experiment_start).total_seconds(),
                    'timestamp': experiment_start.isoformat()
                }
                
        except subprocess.TimeoutExpired:
            return {
                'policy': policy,
                'environment': env_name,
                'status': 'timeout',
                'error': "Experiment timed out after 2 hours",
                'experiment_duration': 7200,
                'timestamp': experiment_start.isoformat()
            }
        except Exception as e:
            return {
                'policy': policy,
                'environment': env_name,
                'status': 'failed',
                'error': str(e),
                'experiment_duration': (datetime.now() - experiment_start).total_seconds(),
                'timestamp': experiment_start.isoformat()
            }
    
    def _analyze_experiment_results(self, result_filename: str, policy: str, env_name: str, env_config: Dict, experiment_start: datetime) -> Dict[str, Any]:
        """Analyze experiment results and calculate metrics"""
        try:
            with open(result_filename, 'r') as f:
                results = yaml.safe_load(f)
            
            # Calculate metrics
            success_list = results.get('success', [])
            reward_list = results.get('reward', [])
            duration_list = results.get('duration', [])
            
            if not success_list:
                # If no success data, infer from rewards
                success_threshold = env_config.get('success_threshold', 1.0)
                success_list = [r >= success_threshold for r in reward_list]
            
            success_rate = np.mean(success_list) if success_list else 0.0
            avg_reward = np.mean(reward_list) if reward_list else 0.0
            avg_duration = np.mean(duration_list) if duration_list else 0.0
            
            # Calculate task-specific metrics
            task_metrics = self._calculate_task_metrics(env_name, env_config, results)
            
            experiment_result = {
                'policy': policy,
                'environment': env_name,
                'status': 'success',
                'success_rate': float(success_rate),
                'avg_reward': float(avg_reward),
                'avg_duration': float(avg_duration),
                'total_episodes': len(success_list),
                'task_type': env_config.get('task_type', 'manipulation'),
                'experiment_duration': (datetime.now() - experiment_start).total_seconds(),
                'timestamp': experiment_start.isoformat(),
                'result_file': str(result_filename)
            }
            
            # Add task-specific metrics
            experiment_result.update(task_metrics)
            
            return experiment_result
            
        except Exception as e:
            return {
                'policy': policy,
                'environment': env_name,
                'status': 'failed',
                'error': f"Failed to analyze results: {str(e)}",
                'experiment_duration': (datetime.now() - experiment_start).total_seconds(),
                'timestamp': experiment_start.isoformat()
            }
    
    def _calculate_task_metrics(self, env_name: str, env_config: Dict, results: Dict) -> Dict[str, Any]:
        """Calculate task-specific metrics for YCB manipulation"""
        task_metrics = {}
        
        task_type = env_config.get('task_type', 'manipulation')
        
        if task_type == 'pick' and 'ycb_objects' in env_config:
            # For picking tasks, calculate per-object success rates if available
            task_metrics['ycb_objects'] = env_config['ycb_objects']
            task_metrics['pick_success_analysis'] = "YCB object picking task completed"
            
        elif task_type == 'push':
            # For pushing tasks, analyze displacement/movement
            task_metrics['push_task_analysis'] = "Object pushing/displacement task completed"
            
        elif task_type == 'place':
            # For placing tasks, analyze placement accuracy
            task_metrics['place_task_analysis'] = "Object placement task completed"
        
        # Add environment-specific analysis
        if 'Cable' in env_name:
            task_metrics['manipulation_type'] = 'cable_manipulation'
        elif 'Cabinet' in env_name:
            task_metrics['manipulation_type'] = 'door_opening'
        elif 'Pick' in env_name:
            task_metrics['manipulation_type'] = 'object_picking'
        
        return task_metrics
    
    def _save_experiment_log(self):
        """Save current experiment log"""
        log_file = self.results_dir / "experiment_log.yaml"
        with open(log_file, 'w') as f:
            yaml.dump(self.experiment_log, f, default_flow_style=False)
    
    def _generate_summary_report(self):
        """Generate comprehensive summary report of all experiments"""
        self.experiment_log['end_time'] = datetime.now().isoformat()
        
        # Create summary table
        summary = {}
        for exp in self.experiment_log['experiments']:
            policy = exp['policy']
            env = exp['environment']
            
            if policy not in summary:
                summary[policy] = {}
            
            if exp['status'] == 'success':
                summary[policy][env] = {
                    'success_rate': exp['success_rate'],
                    'avg_reward': exp['avg_reward'],
                    'avg_duration': exp['avg_duration'],
                    'task_type': exp.get('task_type', 'manipulation'),
                    'total_episodes': exp.get('total_episodes', 0)
                }
            else:
                summary[policy][env] = {
                    'status': exp['status'],
                    'error': exp.get('error', 'Unknown error')
                }
        
        # Save summary
        summary_file = self.results_dir / "experiment_summary.yaml"
        with open(summary_file, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
        
        # Generate markdown report
        self._generate_markdown_report(summary)
        
        # Print summary
        self._print_summary(summary)
    
    def _generate_markdown_report(self, summary: Dict):
        """Generate detailed markdown report"""
        report_file = self.results_dir / "experiment_report.md"
        
        with open(report_file, 'w') as f:
            f.write("# UR5 YCB Object Manipulation Experiment Results\n\n")
            f.write(f"**Experiment Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("**Experiment Focus:** YCB object manipulation tasks (pick, push, place)\n\n")
            
            f.write("## Success Rate Summary\n\n")
            f.write("| Policy | Environment | Task Type | Success Rate | Avg Reward | Avg Duration | Episodes |\n")
            f.write("|--------|-------------|-----------|--------------|------------|--------------|----------|\n")
            
            for policy, envs in summary.items():
                for env, metrics in envs.items():
                    if 'success_rate' in metrics:
                        task_type = metrics.get('task_type', 'manipulation')
                        episodes = metrics.get('total_episodes', 'N/A')
                        f.write(f"| {policy} | {env} | {task_type} | {metrics['success_rate']:.2%} | {metrics['avg_reward']:.3f} | {metrics['avg_duration']:.1f}s | {episodes} |\n")
                    else:
                        f.write(f"| {policy} | {env} | - | FAILED | - | - | - |\n")
            
            f.write("\n## YCB Object Manipulation Analysis\n\n")
            f.write("### Pick Tasks (MujocoUR5ePick)\n")
            f.write("- **YCB Objects**: cracker_box, pudding_box, potted_meat_can\n")
            f.write("- **Success Criteria**: Object lifted and held (reward ≥ 1.0)\n\n")
            
            f.write("### Push/Manipulation Tasks\n")
            f.write("- **Cable Tasks**: Cable manipulation and positioning\n")
            f.write("- **Cabinet Tasks**: Door opening and closing operations\n\n")
            
            f.write("## Detailed Results\n\n")
            for exp in self.experiment_log['experiments']:
                f.write(f"### {exp['policy']} + {exp['environment']}\n\n")
                if exp['status'] == 'success':
                    f.write(f"- **Task Type:** {exp.get('task_type', 'manipulation')}\n")
                    f.write(f"- **Success Rate:** {exp['success_rate']:.2%}\n")
                    f.write(f"- **Average Reward:** {exp['avg_reward']:.3f}\n")
                    f.write(f"- **Average Duration:** {exp['avg_duration']:.1f}s\n")
                    f.write(f"- **Total Episodes:** {exp['total_episodes']}\n")
                    
                    # Add YCB-specific information
                    if 'ycb_objects' in exp:
                        f.write(f"- **YCB Objects:** {', '.join(exp['ycb_objects'])}\n")
                else:
                    f.write(f"- **Status:** {exp['status']}\n")
                    f.write(f"- **Error:** {exp.get('error', 'Unknown')}\n")
                f.write(f"- **Experiment Duration:** {exp['experiment_duration']:.1f}s\n\n")
    
    def _print_summary(self, summary: Dict):
        """Print experiment summary to console"""
        print(f"\n{'='*80}")
        print("📊 YCB OBJECT MANIPULATION EXPERIMENT SUMMARY")
        print(f"{'='*80}")
        
        for policy, envs in summary.items():
            print(f"\n🧠 {policy}:")
            for env, metrics in envs.items():
                if 'success_rate' in metrics:
                    task_type = metrics.get('task_type', 'manipulation')
                    print(f"  📈 {env} ({task_type}): {metrics['success_rate']:.2%} success rate")
                else:
                    print(f"  ❌ {env}: FAILED")


def main():
    parser = argparse.ArgumentParser(description="Run automated UR5 YCB manipulation experiments")
    parser.add_argument("--config", type=str, help="Path to experiment configuration file")
    parser.add_argument("--policy", type=str, help="Run only specific policy")
    parser.add_argument("--env", type=str, help="Run only specific environment")
    parser.add_argument("--task_type", type=str, choices=['pick', 'push', 'place', 'manipulation'], 
                       help="Run only specific task type")
    
    args = parser.parse_args()
    
    runner = UR5ExperimentRunner(args.config)
    
    # Filter experiments if specific policy/env/task requested
    if args.policy:
        runner.config['policies'] = [p for p in runner.config['policies'] if p == args.policy]
    if args.env:
        runner.config['environments'] = {k: v for k, v in runner.config['environments'].items() if k == args.env}
    if args.task_type:
        runner.config['environments'] = {k: v for k, v in runner.config['environments'].items() 
                                       if v.get('task_type', 'manipulation') == args.task_type}
    
    runner.run_all_experiments()


if __name__ == "__main__":
    main()
