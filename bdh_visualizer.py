#!/usr/bin/env python
# BDH Performance Visualizer
# This tool visualizes where BDH outperforms larger transformer models

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style for better visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

class ModelComparisonVisualizer:
    def __init__(self):
        self.models = {
            "BDH": {"color": "#ff9900", "marker": "o"},
            "GPT-2": {"color": "#3366cc", "marker": "s"},
            "BERT": {"color": "#dc3912", "marker": "^"},
            "T5": {"color": "#109618", "marker": "d"},
        }
        
        # Sample data - in a real scenario, this would come from actual benchmarks
        self.sample_data = self._generate_sample_data()
    
    def _generate_sample_data(self):
        """Generate sample performance data for different models across tasks"""
        tasks = [
            "Text Classification", 
            "Question Answering", 
            "Summarization", 
            "Entity Recognition",
            "Sentiment Analysis"
        ]
        
        # Metrics to compare
        metrics = {
            "accuracy": {
                "BDH": [0.82, 0.76, 0.71, 0.85, 0.88],
                "GPT-2": [0.88, 0.82, 0.79, 0.81, 0.86],
                "BERT": [0.86, 0.84, 0.75, 0.89, 0.87],
                "T5": [0.89, 0.85, 0.83, 0.87, 0.89]
            },
            "latency_ms": {
                "BDH": [15, 18, 25, 12, 14],
                "GPT-2": [45, 52, 60, 48, 42],
                "BERT": [38, 42, 45, 36, 40],
                "T5": [55, 62, 70, 58, 52]
            },
            "memory_mb": {
                "BDH": [120, 125, 130, 115, 118],
                "GPT-2": [450, 460, 470, 445, 455],
                "BERT": [350, 360, 365, 340, 355],
                "T5": [520, 530, 540, 515, 525]
            },
            "efficiency_score": {
                "BDH": [0.92, 0.88, 0.85, 0.94, 0.93],
                "GPT-2": [0.65, 0.62, 0.58, 0.64, 0.67],
                "BERT": [0.72, 0.70, 0.68, 0.74, 0.71],
                "T5": [0.60, 0.58, 0.55, 0.61, 0.62]
            }
        }
        
        return {"tasks": tasks, "metrics": metrics}
    
    def plot_radar_chart(self, save_path=None):
        """Create a radar chart comparing models across different tasks"""
        tasks = self.sample_data["tasks"]
        efficiency = self.sample_data["metrics"]["efficiency_score"]
        
        # Number of variables
        N = len(tasks)
        
        # Compute angle for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
        
        # Draw one axis per variable and add labels
        plt.xticks(angles[:-1], tasks, size=12)
        
        # Plot data
        for model_name, model_data in self.models.items():
            values = efficiency[model_name]
            values += values[:1]  # Close the loop
            ax.plot(angles, values, 'o-', linewidth=2, 
                    label=model_name, color=model_data["color"], 
                    marker=model_data["marker"], markersize=8)
            ax.fill(angles, values, alpha=0.1, color=model_data["color"])
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.title("Model Efficiency Comparison Across Tasks", size=15, y=1.1)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Radar chart saved to {save_path}")
        
        return fig
    
    def plot_efficiency_vs_accuracy(self, save_path=None):
        """Create a scatter plot of efficiency vs accuracy"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        tasks = self.sample_data["tasks"]
        accuracy = self.sample_data["metrics"]["accuracy"]
        efficiency = self.sample_data["metrics"]["efficiency_score"]
        
        for model_name, model_data in self.models.items():
            for i, task in enumerate(tasks):
                ax.scatter(accuracy[model_name][i], efficiency[model_name][i], 
                          color=model_data["color"], marker=model_data["marker"],
                          s=100, label=model_name if i == 0 else "")
                ax.annotate(task, (accuracy[model_name][i], efficiency[model_name][i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel("Accuracy", fontsize=12)
        ax.set_ylabel("Efficiency Score", fontsize=12)
        ax.set_title("Efficiency vs. Accuracy Across Tasks", fontsize=15)
        ax.legend()
        
        # Add a line to highlight the "sweet spot" of high efficiency and good accuracy
        ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0.8, color='gray', linestyle='--', alpha=0.5)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Efficiency vs accuracy plot saved to {save_path}")
        
        return fig
    
    def plot_resource_comparison(self, save_path=None):
        """Create a bar chart comparing resource usage"""
        latency = [np.mean(self.sample_data["metrics"]["latency_ms"][model]) 
                  for model in self.models.keys()]
        memory = [np.mean(self.sample_data["metrics"]["memory_mb"][model]) / 100  # Scale down for visualization
                 for model in self.models.keys()]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(self.models))
        width = 0.35
        
        ax.bar(x - width/2, latency, width, label='Latency (ms)', color='#5DA5DA')
        ax.bar(x + width/2, memory, width, label='Memory (100MB)', color='#FAA43A')
        
        ax.set_xticks(x)
        ax.set_xticklabels(self.models.keys())
        ax.legend()
        
        ax.set_ylabel('Resource Usage')
        ax.set_title('Resource Usage Comparison')
        
        # Add value labels on top of bars
        for i, v in enumerate(latency):
            ax.text(i - width/2, v + 0.5, f"{v:.1f}", ha='center', fontsize=9)
        
        for i, v in enumerate(memory):
            ax.text(i + width/2, v + 0.5, f"{v*100:.0f}MB", ha='center', fontsize=9)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Resource comparison plot saved to {save_path}")
        
        return fig
    
    def generate_all_visualizations(self, output_dir="./figs"):
        """Generate all visualizations and save them to the output directory"""
        import os
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        self.plot_radar_chart(os.path.join(output_dir, "bdh_radar_chart.png"))
        self.plot_efficiency_vs_accuracy(os.path.join(output_dir, "bdh_efficiency_accuracy.png"))
        self.plot_resource_comparison(os.path.join(output_dir, "bdh_resource_comparison.png"))
        
        print(f"All visualizations saved to {output_dir}")

if __name__ == "__main__":
    visualizer = ModelComparisonVisualizer()
    visualizer.generate_all_visualizations()
    
    # Display plots if running interactively
    plt.show()