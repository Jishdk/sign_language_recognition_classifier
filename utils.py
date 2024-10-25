# utils.py

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List

class DataUtils:
    """Utility functions for data handling and visualization."""
    
    @staticmethod
    def create_folders(base_path: str):
        """
        Create necessary folders for the project.
        
        Args:
            base_path: Base directory path
        """
        # Define required folders
        folders = [
            'data',
            'results',
            'results',
            'logs'
        ]
        
        # Create each folder
        for folder in folders:
            Path(base_path).joinpath(folder).mkdir(parents=True, exist_ok=True)
            
        print("Created project folders:")
        for folder in folders:
            print(f"  - {folder}")

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
    def plot_learning_curves(history: Dict, save_path: str = None):
        """
        Plot training and validation curves.
        
        Args:
            history: Dictionary containing training history
            save_path: Path to save the plot (optional)
        """
        plt.figure(figsize=(12, 4))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Training')
        if 'val_loss' in history:
            plt.plot(history['val_loss'], label='Validation')
        plt.title('Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Training')
        if 'val_acc' in history:
            plt.plot(history['val_acc'], label='Validation')
        plt.title('Accuracy Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Learning curves saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_model_comparison(models_results: Dict, save_path: str = None):
        """
        Plot comparison of model performances.
        
        Args:
            models_results: Dictionary containing results for each model
            save_path: Path to save the plot (optional)
        """
        # Extract metrics
        models = list(models_results.keys())
        accuracies = [results['accuracy'] for results in models_results.values()]
        
        # Create bar plot
        plt.figure(figsize=(8, 6))
        bars = plt.bar(models, accuracies)
        plt.title('Model Performance Comparison')
        plt.ylabel('Accuracy')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2%}',
                    ha='center', va='bottom')
        
        plt.ylim(0, 1.1)  # Set y-axis limit to accommodate labels
        
        if save_path:
            plt.savefig(save_path)
            print(f"Model comparison plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()

__all__ = ['DataUtils', 'Visualizer']