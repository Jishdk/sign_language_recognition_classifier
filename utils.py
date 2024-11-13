# utils.py

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict

class DataUtils:
    """Utility functions for data handling and visualization."""
    
    @staticmethod
    def create_folders(base_path: str):
        """
        Create the results folder for the project.
        
        Args:
            base_path: Base directory path
        """
        # Define and create the results folder
        results_path = Path(base_path).joinpath('results')
        results_path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def save_results(results: Dict, filepath: str):
        """
        Save results to JSON file.
        
        Args:
            results: Dictionary of results to save
            filepath: Path to save file
        """
        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            return obj

        # Convert and save
        serializable_results = convert_to_serializable(results)
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        
        print(f"Results saved to: {filepath}")

class Visualizer:
    """Utility functions for visualization."""
    
    @staticmethod
    def plot_model_comparison(models_results: Dict, save_path: str = None):
        """
        Plot comparison of model performances including accuracy, recall, F1 score, and precision.
        
        Args:
            models_results: Dictionary containing results for each model
            save_path: Path to save the plot (optional)
        """
        # Metrics to include in comparison
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        model_names = list(models_results.keys())
        
        # Extract metric values for each model
        metric_values = {metric: [models_results[model][metric] for model in model_names] 
                        for metric in metrics}
        
        # Set up bar plot with grouped bars for each metric
        x = np.arange(len(metrics))
        width = 0.50
        plt.figure(figsize=(25, 15))
        plt.tight_layout()
        
        # Plot bars for each model
        for i, model_name in enumerate(model_names):
            plt.bar(x + i * width, [metric_values[metric][i] for metric in metrics], 
                    width, label=model_name)
        
        # Customize axis labels and title
        plt.xticks(x + width / 2, [metric.capitalize() for metric in metrics],
                fontsize=26, rotation=0)  
        plt.yticks(fontsize=24) 
        
        # Add labels and title
        plt.ylabel('Scores', fontsize=26, labelpad=15)  
        plt.xlabel('Metrics', fontsize=26, labelpad=15) 
        plt.title('Model Performance Comparison', fontsize=24, pad=20) 
        
        # Move the legend outside the plot
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=16)
        
        # Display values on top of each bar
        for i, model_name in enumerate(model_names):
            for j, metric in enumerate(metrics):
                plt.text(x[j] + i * width, metric_values[metric][i] + 0.01, 
                        f"{metric_values[metric][i]:.2f}",
                        ha='center', va='bottom',
                        fontsize=26)
        
        # Save
        if save_path:
            plt.savefig(save_path, 
                        bbox_inches='tight', 
                        dpi=300,  
                        pad_inches=0.5)
            print(f"Model comparison plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()

__all__ = ['DataUtils', 'Visualizer']