# evaluation.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
from pathlib import Path

class Evaluator:
    """Class for evaluating sign language recognition models."""
    
    def __init__(self, model_name: str, save_dir: str = 'results'):
        """
        Initialize evaluator.
        
        Args:
            model_name: Name of the model being evaluated
            save_dir: Directory to save results
        """
        self.model_name = model_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
    
    def plot_confusion_matrix(self, conf_matrix: np.ndarray, labels: List[str]):
        """
        Plot and save confusion matrix.
        
        Args:
            conf_matrix: Confusion matrix array
            labels: List of class labels
        """
        plt.figure(figsize=(12, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.title(f'{self.model_name} - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.save_dir / f'{self.model_name.lower()}_confusion_matrix.png')
        plt.close()
    

class StressTester:
    """Class for stress testing sign language recognition models."""
    
    def __init__(self, model):
        """
        Initialize stress tester.
        
        Args:
            model: Model to test
        """
        self.model = model
    
    def test_feature_scaling(self, features: np.ndarray, labels: np.ndarray) -> Dict:
        """
        Test model with scaled features.
        
        Args:
            features: Input features
            labels: True labels
            
        Returns:
            Dictionary of test results
        """
        results = {}
        
        # Test different scaling factors
        scales = [0.5, 0.75, 1.25, 1.5]
        print("\nTesting feature scaling...")
        
        for scale in scales:
            # Scale features
            scaled_features = features * scale
            
            # Get predictions
            predictions = self.model.predict(scaled_features)
            
            # Calculate accuracy
            accuracy = np.mean(predictions == labels)
            results[f'scale_{scale}'] = accuracy
            print(f"Scale {scale}: {accuracy:.4f} accuracy")
        
        return results
    
    def test_noise_resistance(self, features: np.ndarray, labels: np.ndarray) -> Dict:
        """
        Test model with noisy features.
        
        Args:
            features: Input features
            labels: True labels
            
        Returns:
            Dictionary of test results
        """
        results = {}
        
        # Test different noise levels
        noise_levels = [0.01, 0.05, 0.1]
        print("\nTesting noise resistance...")
        
        for noise in noise_levels:
            # Add Gaussian noise
            noise_matrix = np.random.normal(0, noise, features.shape)
            noisy_features = features + noise_matrix
            
            # Get predictions
            predictions = self.model.predict(noisy_features)
            
            # Calculate accuracy
            accuracy = np.mean(predictions == labels)
            results[f'noise_{noise}'] = accuracy
            print(f"Noise {noise}: {accuracy:.4f} accuracy")
        
        return results
    
    def test_feature_dropout(self, features: np.ndarray, labels: np.ndarray) -> Dict:
        """
        Test model with random feature dropout.
        
        Args:
            features: Input features
            labels: True labels
            
        Returns:
            Dictionary of test results
        """
        results = {}
        
        # Test different dropout rates
        dropout_rates = [0.1, 0.2, 0.3]
        print("\nTesting feature dropout...")
        
        for rate in dropout_rates:
            # Create feature mask
            mask = np.random.rand(*features.shape) > rate
            dropped_features = features * mask
            
            # Get predictions
            predictions = self.model.predict(dropped_features)
            
            # Calculate accuracy
            accuracy = np.mean(predictions == labels)
            results[f'dropout_{rate}'] = accuracy
            print(f"Dropout {rate}: {accuracy:.4f} accuracy")
        
        return results
    
    def plot_results(self, results: Dict, title: str, save_path: str):
        """
        Plot stress test results.
        
        Args:
            results: Dictionary of test results
            title: Plot title
            save_path: Path to save plot
        """
        plt.figure(figsize=(15, 10))
        
        # Extract values
        conditions = list(results.keys())
        accuracies = list(results.values())
        
        # Create bar plot
        plt.bar(conditions, accuracies)
        plt.title(title)
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45, fontsize=24)
        plt.yticks(fontsize=24)
        
        # Add value labels
        for i, v in enumerate(accuracies):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=24)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    
__all__ = ['Evaluator', 'StressTester']