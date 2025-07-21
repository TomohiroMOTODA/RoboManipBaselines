#!/usr/bin/env python3
"""
Result Analyzer for UR5 YCB Object Manipulation Experiments

This script analyzes experiment results, generates visualizations,
and provides comprehensive performance metrics for YCB manipulation tasks.
"""

import os
import sys
import yaml
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class YCBExperimentAnalyzer:
    """Analyzer for YCB object manipulation experiment results"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        if not self.results_dir.exists():
            raise FileNotFoundError(f"Results directory not found: {results_dir}")
        
        self.results_data = {}
        self.summary_data = {}
        self.ycb_objects = ['cracker_box', 'pudding_box', 'potted_meat_can']
        
    def load_results(self) -> bool:
        """Load all experiment results from the results directory"""
        try:
            # Load individual result files
            result_files = list(self.results_dir.glob("result_*.yaml"))
            print(f"📁 Found {len(result_files)} result files")
            
            for result_file in result_files:
                # Parse filename to extract policy and environment
                parts = result_file.stem.split('_', 2)
                if len(parts) >= 3:
                    policy = parts[1]
                    environment = parts[2]
                    
                    with open(result_file, 'r') as f:
                        data = yaml.safe_load(f)
                    
                    if policy not in self.results_data:
                        self.results_data[policy] = {}
                    
                    self.results_data[policy][environment] = data
            
            # Load summary if available
            summary_file = self.results_dir / "experiment_summary.yaml"
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    self.summary_data = yaml.safe_load(f)
            
            print(f"✅ Loaded results for {len(self.results_data)} policies")
            return len(self.results_data) > 0
            
        except Exception as e:
            print(f"❌ Failed to load results: {e}")
            return False
    
    def analyze_performance(self) -> pd.DataFrame:
        """Analyze performance metrics across all experiments"""
        analysis_data = []
        
        for policy, environments in self.results_data.items():
            for env, data in environments.items():
                # Calculate basic metrics
                success_list = data.get('success', [])
                reward_list = data.get('reward', [])
                duration_list = data.get('duration', [])
                
                if not success_list and reward_list:
                    # Infer success from rewards (threshold = 1.0)
                    success_list = [r >= 1.0 for r in reward_list]
                
                if success_list:
                    success_rate = np.mean(success_list)
                    success_std = np.std(success_list)
                else:
                    success_rate = 0.0
                    success_std = 0.0
                
                if reward_list:
                    avg_reward = np.mean(reward_list)
                    reward_std = np.std(reward_list)
                    max_reward = np.max(reward_list)
                    min_reward = np.min(reward_list)
                else:
                    avg_reward = reward_std = max_reward = min_reward = 0.0
                
                if duration_list:
                    avg_duration = np.mean(duration_list)
                    duration_std = np.std(duration_list)
                else:
                    avg_duration = duration_std = 0.0
                
                # Determine task type
                task_type = self._get_task_type(env)
                
                # Calculate task-specific metrics
                task_metrics = self._calculate_task_specific_metrics(env, data)
                
                analysis_row = {
                    'Policy': policy,
                    'Environment': env,
                    'Task_Type': task_type,
                    'Success_Rate': success_rate,
                    'Success_Std': success_std,
                    'Avg_Reward': avg_reward,
                    'Reward_Std': reward_std,
                    'Max_Reward': max_reward,
                    'Min_Reward': min_reward,
                    'Avg_Duration': avg_duration,
                    'Duration_Std': duration_std,
                    'Total_Episodes': len(success_list),
                    'YCB_Task': 'Pick' in env
                }
                
                # Add task-specific metrics
                analysis_row.update(task_metrics)
                
                analysis_data.append(analysis_row)
        
        df = pd.DataFrame(analysis_data)
        return df
    
    def _get_task_type(self, environment: str) -> str:
        """Determine task type from environment name"""
        if 'Pick' in environment:
            return 'pick'
        elif 'Cable' in environment:
            return 'manipulation'
        elif 'Cabinet' in environment or 'Door' in environment:
            return 'push'
        else:
            return 'unknown'
    
    def _calculate_task_specific_metrics(self, env: str, data: Dict) -> Dict[str, Any]:
        """Calculate task-specific performance metrics"""
        metrics = {}
        
        reward_list = data.get('reward', [])
        success_list = data.get('success', [])
        
        if not success_list and reward_list:
            success_list = [r >= 1.0 for r in reward_list]
        
        # YCB object picking specific metrics
        if 'Pick' in env and reward_list:
            # Analyze picking performance
            high_rewards = [r for r in reward_list if r >= 0.8]
            metrics['High_Performance_Rate'] = len(high_rewards) / len(reward_list)
            
            # Estimate per-object performance (if world indices correlate with objects)
            if len(reward_list) >= len(self.ycb_objects):
                obj_rewards = np.array_split(reward_list, len(self.ycb_objects))
                for i, obj_name in enumerate(self.ycb_objects):
                    if i < len(obj_rewards):
                        obj_success_rate = np.mean([r >= 1.0 for r in obj_rewards[i]])
                        metrics[f'{obj_name}_Success_Rate'] = obj_success_rate
        
        # Manipulation task metrics
        elif 'Cable' in env:
            # Cable manipulation specific analysis
            if reward_list:
                manipulation_efficiency = np.mean([r for r in reward_list if r > 0])
                metrics['Manipulation_Efficiency'] = manipulation_efficiency
        
        # Push task metrics
        elif 'Cabinet' in env or 'Door' in env:
            # Push/door opening specific analysis
            if reward_list:
                opening_success = np.mean([r >= 1.0 for r in reward_list])
                metrics['Opening_Success_Rate'] = opening_success
        
        return metrics
    
    def generate_visualizations(self, df: pd.DataFrame, save_plots: bool = True):
        """Generate comprehensive visualizations"""
        output_dir = self.results_dir / "plots"
        if save_plots:
            output_dir.mkdir(exist_ok=True)
        
        # 1. Success Rate Heatmap
        self._plot_success_heatmap(df, output_dir if save_plots else None)
        
        # 2. Performance Comparison by Task Type
        self._plot_task_performance(df, output_dir if save_plots else None)
        
        # 3. YCB Object Specific Analysis
        self._plot_ycb_analysis(df, output_dir if save_plots else None)
        
        # 4. Reward Distribution
        self._plot_reward_distribution(df, output_dir if save_plots else None)
        
        # 5. Duration Analysis
        self._plot_duration_analysis(df, output_dir if save_plots else None)
        
        if save_plots:
            print(f"📊 Plots saved to: {output_dir}")
    
    def _plot_success_heatmap(self, df: pd.DataFrame, output_dir: Path = None):
        """Create success rate heatmap"""
        plt.figure(figsize=(12, 8))
        
        # Pivot data for heatmap
        pivot_data = df.pivot(index='Policy', columns='Environment', values='Success_Rate')
        
        # Create heatmap
        sns.heatmap(pivot_data, annot=True, fmt='.2%', cmap='RdYlGn', 
                   cbar_kws={'label': 'Success Rate'})
        
        plt.title('Success Rate Heatmap: Policies vs Environments', fontsize=16, fontweight='bold')
        plt.xlabel('Environment', fontsize=12)
        plt.ylabel('Policy', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(output_dir / 'success_rate_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_task_performance(self, df: pd.DataFrame, output_dir: Path = None):
        """Plot performance comparison by task type"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Success rate by task type
        sns.boxplot(data=df, x='Task_Type', y='Success_Rate', hue='Policy', ax=axes[0,0])
        axes[0,0].set_title('Success Rate by Task Type')
        axes[0,0].set_ylabel('Success Rate')
        
        # Average reward by task type
        sns.boxplot(data=df, x='Task_Type', y='Avg_Reward', hue='Policy', ax=axes[0,1])
        axes[0,1].set_title('Average Reward by Task Type')
        axes[0,1].set_ylabel('Average Reward')
        
        # Duration by task type
        sns.boxplot(data=df, x='Task_Type', y='Avg_Duration', hue='Policy', ax=axes[1,0])
        axes[1,0].set_title('Average Duration by Task Type')
        axes[1,0].set_ylabel('Duration (seconds)')
        
        # Performance correlation
        sns.scatterplot(data=df, x='Avg_Reward', y='Success_Rate', 
                       hue='Task_Type', style='Policy', s=100, ax=axes[1,1])
        axes[1,1].set_title('Success Rate vs Average Reward')
        axes[1,1].set_xlabel('Average Reward')
        axes[1,1].set_ylabel('Success Rate')
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(output_dir / 'task_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_ycb_analysis(self, df: pd.DataFrame, output_dir: Path = None):
        """Plot YCB object specific analysis"""
        # Filter for YCB tasks only
        ycb_df = df[df['YCB_Task'] == True]
        
        if ycb_df.empty:
            print("⚠️ No YCB task data found for analysis")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # YCB picking success rates
        sns.barplot(data=ycb_df, x='Policy', y='Success_Rate', ax=axes[0,0])
        axes[0,0].set_title('YCB Object Picking Success Rates')
        axes[0,0].set_ylabel('Success Rate')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # High performance rate
        if 'High_Performance_Rate' in ycb_df.columns:
            sns.barplot(data=ycb_df, x='Policy', y='High_Performance_Rate', ax=axes[0,1])
            axes[0,1].set_title('High Performance Rate (Reward ≥ 0.8)')
            axes[0,1].set_ylabel('High Performance Rate')
            axes[0,1].tick_params(axis='x', rotation=45)
        
        # Per-object success rates if available
        obj_columns = [col for col in ycb_df.columns if any(obj in col for obj in self.ycb_objects)]
        if obj_columns:
            obj_data = []
            for _, row in ycb_df.iterrows():
                for col in obj_columns:
                    if col.endswith('_Success_Rate'):
                        obj_name = col.replace('_Success_Rate', '')
                        obj_data.append({
                            'Policy': row['Policy'],
                            'Object': obj_name,
                            'Success_Rate': row[col]
                        })
            
            if obj_data:
                obj_df = pd.DataFrame(obj_data)
                sns.barplot(data=obj_df, x='Object', y='Success_Rate', hue='Policy', ax=axes[1,0])
                axes[1,0].set_title('Success Rate by YCB Object')
                axes[1,0].set_ylabel('Success Rate')
                axes[1,0].tick_params(axis='x', rotation=45)
        
        # Reward distribution for YCB tasks
        sns.boxplot(data=ycb_df, x='Policy', y='Avg_Reward', ax=axes[1,1])
        axes[1,1].set_title('YCB Task Reward Distribution')
        axes[1,1].set_ylabel('Average Reward')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(output_dir / 'ycb_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_reward_distribution(self, df: pd.DataFrame, output_dir: Path = None):
        """Plot reward distribution analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Reward distribution by policy
        for policy in df['Policy'].unique():
            policy_data = df[df['Policy'] == policy]
            axes[0,0].hist(policy_data['Avg_Reward'], alpha=0.7, label=policy, bins=20)
        axes[0,0].set_title('Reward Distribution by Policy')
        axes[0,0].set_xlabel('Average Reward')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].legend()
        
        # Reward vs Success Rate
        sns.scatterplot(data=df, x='Avg_Reward', y='Success_Rate', 
                       hue='Policy', style='Task_Type', s=100, ax=axes[0,1])
        axes[0,1].set_title('Reward vs Success Rate')
        
        # Reward variability
        sns.barplot(data=df, x='Policy', y='Reward_Std', hue='Task_Type', ax=axes[1,0])
        axes[1,0].set_title('Reward Variability (Standard Deviation)')
        axes[1,0].set_ylabel('Reward Std Dev')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Max vs Min rewards
        width = 0.35
        policies = df['Policy'].unique()
        x = np.arange(len(policies))
        
        for i, policy in enumerate(policies):
            policy_data = df[df['Policy'] == policy]
            axes[1,1].bar(x[i] - width/2, policy_data['Max_Reward'].mean(), 
                         width, label=f'{policy} Max', alpha=0.8)
            axes[1,1].bar(x[i] + width/2, policy_data['Min_Reward'].mean(), 
                         width, label=f'{policy} Min', alpha=0.8)
        
        axes[1,1].set_title('Max vs Min Rewards by Policy')
        axes[1,1].set_xlabel('Policy')
        axes[1,1].set_ylabel('Reward')
        axes[1,1].set_xticks(x)
        axes[1,1].set_xticklabels(policies, rotation=45)
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(output_dir / 'reward_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_duration_analysis(self, df: pd.DataFrame, output_dir: Path = None):
        """Plot duration analysis"""
        plt.figure(figsize=(12, 8))
        
        # Duration comparison
        sns.boxplot(data=df, x='Environment', y='Avg_Duration', hue='Policy')
        plt.title('Task Duration Analysis by Environment and Policy')
        plt.xlabel('Environment')
        plt.ylabel('Average Duration (seconds)')
        plt.xticks(rotation=45, ha='right')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(output_dir / 'duration_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_comprehensive_report(self, df: pd.DataFrame):
        """Generate comprehensive analysis report"""
        report_file = self.results_dir / "comprehensive_analysis_report.md"
        
        with open(report_file, 'w') as f:
            f.write("# Comprehensive YCB Object Manipulation Analysis Report\n\n")
            f.write(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary statistics
            f.write("## Summary Statistics\n\n")
            f.write(f"- **Total Experiments:** {len(df)}\n")
            f.write(f"- **Policies Tested:** {', '.join(df['Policy'].unique())}\n")
            f.write(f"- **Environments:** {', '.join(df['Environment'].unique())}\n")
            f.write(f"- **Task Types:** {', '.join(df['Task_Type'].unique())}\n\n")
            
            # Best performing combinations
            f.write("## Best Performing Combinations\n\n")
            best_overall = df.loc[df['Success_Rate'].idxmax()]
            f.write(f"**Overall Best:** {best_overall['Policy']} on {best_overall['Environment']} ")
            f.write(f"({best_overall['Success_Rate']:.2%} success rate)\n\n")
            
            # Task-specific best performers
            for task_type in df['Task_Type'].unique():
                task_df = df[df['Task_Type'] == task_type]
                best_task = task_df.loc[task_df['Success_Rate'].idxmax()]
                f.write(f"**Best {task_type.title()} Task:** {best_task['Policy']} on {best_task['Environment']} ")
                f.write(f"({best_task['Success_Rate']:.2%} success rate)\n\n")
            
            # Detailed analysis
            f.write("## Detailed Performance Analysis\n\n")
            
            # Success rate analysis
            f.write("### Success Rate Analysis\n\n")
            success_stats = df.groupby('Policy')['Success_Rate'].agg(['mean', 'std', 'min', 'max'])
            f.write("| Policy | Mean Success Rate | Std Dev | Min | Max |\n")
            f.write("|--------|------------------|---------|-----|-----|\n")
            for policy, stats in success_stats.iterrows():
                f.write(f"| {policy} | {stats['mean']:.2%} | {stats['std']:.3f} | {stats['min']:.2%} | {stats['max']:.2%} |\n")
            f.write("\n")
            
            # YCB specific analysis
            ycb_df = df[df['YCB_Task'] == True]
            if not ycb_df.empty:
                f.write("### YCB Object Manipulation Analysis\n\n")
                f.write("YCB object picking tasks show the following performance:\n\n")
                for _, row in ycb_df.iterrows():
                    f.write(f"- **{row['Policy']}**: {row['Success_Rate']:.2%} success rate, ")
                    f.write(f"{row['Avg_Reward']:.3f} average reward\n")
                f.write("\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            
            # Policy recommendations
            policy_ranking = df.groupby('Policy')['Success_Rate'].mean().sort_values(ascending=False)
            f.write("### Policy Ranking (by average success rate):\n\n")
            for i, (policy, success_rate) in enumerate(policy_ranking.items(), 1):
                f.write(f"{i}. **{policy}**: {success_rate:.2%} average success rate\n")
            
            f.write("\n### Task-Specific Recommendations:\n\n")
            
            # Task-specific recommendations
            for task_type in df['Task_Type'].unique():
                task_df = df[df['Task_Type'] == task_type]
                best_policy = task_df.loc[task_df['Success_Rate'].idxmax(), 'Policy']
                f.write(f"- **{task_type.title()} Tasks**: Use {best_policy} policy\n")
            
        print(f"📋 Comprehensive analysis report saved to: {report_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze YCB object manipulation experiment results")
    parser.add_argument("--results_dir", type=str, default="./results", 
                       help="Directory containing experiment results")
    parser.add_argument("--plot", action="store_true", help="Generate visualization plots")
    parser.add_argument("--compare_policies", action="store_true", 
                       help="Generate policy comparison analysis")
    parser.add_argument("--ycb_analysis", action="store_true", 
                       help="Focus on YCB object specific analysis")
    parser.add_argument("--save_plots", action="store_true", help="Save plots to files")
    parser.add_argument("--comprehensive_report", action="store_true", 
                       help="Generate comprehensive analysis report")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = YCBExperimentAnalyzer(args.results_dir)
    
    # Load results
    if not analyzer.load_results():
        print("❌ Failed to load experiment results")
        return
    
    # Analyze performance
    print("🔍 Analyzing performance metrics...")
    df = analyzer.analyze_performance()
    
    print(f"📊 Analysis Summary:")
    print(f"  - {len(df)} experiment configurations analyzed")
    print(f"  - {len(df['Policy'].unique())} policies tested")
    print(f"  - {len(df['Environment'].unique())} environments tested")
    print(f"  - {len(df[df['YCB_Task'] == True])} YCB manipulation tasks")
    
    # Generate visualizations
    if args.plot or args.compare_policies:
        print("📈 Generating visualizations...")
        analyzer.generate_visualizations(df, args.save_plots)
    
    # YCB specific analysis
    if args.ycb_analysis:
        print("📦 Performing YCB-specific analysis...")
        ycb_df = df[df['YCB_Task'] == True]
        if not ycb_df.empty:
            analyzer._plot_ycb_analysis(ycb_df, 
                                      analyzer.results_dir / "plots" if args.save_plots else None)
        else:
            print("⚠️ No YCB task data found")
    
    # Comprehensive report
    if args.comprehensive_report:
        print("📋 Generating comprehensive report...")
        analyzer.generate_comprehensive_report(df)
    
    # Print top performers
    print(f"\n{'='*60}")
    print("🏆 TOP PERFORMERS")
    print(f"{'='*60}")
    
    best_overall = df.loc[df['Success_Rate'].idxmax()]
    print(f"🥇 Overall Best: {best_overall['Policy']} on {best_overall['Environment']}")
    print(f"   Success Rate: {best_overall['Success_Rate']:.2%}")
    print(f"   Average Reward: {best_overall['Avg_Reward']:.3f}")
    
    # Best by task type
    for task_type in df['Task_Type'].unique():
        task_df = df[df['Task_Type'] == task_type]
        best_task = task_df.loc[task_df['Success_Rate'].idxmax()]
        print(f"🎯 Best {task_type.title()}: {best_task['Policy']} ({best_task['Success_Rate']:.2%})")


if __name__ == "__main__":
    main()