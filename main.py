# main.py

import numpy as np
import mediapipe as mp
from pathlib import Path

# Import our modules
from config import Config
from utils import DataUtils, Visualizer
from hand_detector import HandDetector
from data_processing import SignLanguageDataset
from models import RandomForest, NeuralNetwork
from evaluation import Evaluator, StressTester

def main():
    
    # Initialize Configuration
    print("\n1. Setting up project...")
    config = Config()
    DataUtils.create_folders(config.BASE_DIR)
    
    # Initialize Hand Detector
    print("\n2. Initializing hand detector...")
    hand_detector = HandDetector(**config.MEDIAPIPE_PARAMS)
    
    # Load and Process Dataset
    print("\n3. Loading and processing dataset...")
    dataset = SignLanguageDataset(str(config.DATA_DIR), hand_detector)
    
    # Get data splits
    data_splits = dataset.prepare_data(
        valid_size=config.VAL_SPLIT,
        test_size=config.TEST_SPLIT
    )
    
    # Print data shapes
    for split_name, (X, y) in data_splits.items():
        print(f"{split_name} split shape: {X.shape}")
    
    # Prepare Data and Train Models
    print("\n4. Training models...")
    
    # Shift labels to start from 0
    for split in ['train', 'valid', 'test']:
        X, y = data_splits[split]
        # Shift labels down by 1 (1-26 -> 0-25)
        data_splits[split] = (X, y - 1)
        
    # Get number of classes
    num_classes = len(np.unique(data_splits['train'][1]))

    # Define class labels based on number of classes
    class_labels = [chr(ord('A') + i) for i in range(num_classes)]
    
    # Random Forest
    print("\nTraining Random Forest...")
    rf_model = RandomForest(n_trees=config.RF_PARAMS['n_estimators'])
    rf_model.train(data_splits['train'][0], data_splits['train'][1])
    
    # Neural Network
    print("\nTraining Neural Network...")
    input_size = data_splits['train'][0].shape[1]
    
    nn_model = NeuralNetwork(
        input_size=input_size,
        hidden_size=config.NN_PARAMS['hidden_sizes'][0],
        num_classes=num_classes,
        learning_rate=config.NN_PARAMS['learning_rate']
    )
    
    nn_history = nn_model.train(
        data_splits['train'][0], data_splits['train'][1],
        data_splits['valid'][0], data_splits['valid'][1],
        batch_size=config.NN_PARAMS['batch_size'],
        num_epochs=config.NN_PARAMS['num_epochs']
    )
    
    # Evaluate Models
    print("\n5. Evaluating models...")

    # Initialize evaluators
    rf_evaluator = Evaluator("Random Forest", str(config.RESULTS_DIR))
    nn_evaluator = Evaluator("Neural Network", str(config.RESULTS_DIR))

    # Get predictions and metrics
    rf_metrics = rf_model.evaluate(data_splits['test'][0], data_splits['test'][1])
    nn_metrics = nn_model.evaluate(data_splits['test'][0], data_splits['test'][1])

    # Display results
    print(f"Random Forest Metrics:\nAccuracy: {rf_metrics['accuracy']:.4f}\n"
        f"Precision: {rf_metrics['precision']:.4f}\nRecall: {rf_metrics['recall']:.4f}\n"
        f"F1 Score: {rf_metrics['f1']:.4f}\n")
    print(f"Neural Network Metrics:\nAccuracy: {nn_metrics['accuracy']:.4f}\n"
        f"Precision: {nn_metrics['precision']:.4f}\nRecall: {nn_metrics['recall']:.4f}\n"
        f"F1 Score: {nn_metrics['f1']:.4f}")

    # Plot confusion matrices
    rf_evaluator.plot_confusion_matrix(rf_metrics['confusion_matrix'], class_labels)
    nn_evaluator.plot_confusion_matrix(nn_metrics['confusion_matrix'], class_labels)

    
    # Stress Testing
    print("\n6. Running stress tests...")
    
    # Test both models
    rf_stress = StressTester(rf_model)
    nn_stress = StressTester(nn_model)
    
    # Get test data
    X_test, y_test = data_splits['test']
    
    # Run tests
    test_types = ['feature_scaling', 'noise_resistance', 'feature_dropout']
    stress_results = {
        'random_forest': {},
        'neural_network': {}
    }
    
    for test_type in test_types:
        # Run test for both models
        rf_test = getattr(rf_stress, f'test_{test_type}')(X_test, y_test)
        nn_test = getattr(nn_stress, f'test_{test_type}')(X_test, y_test)
        
        # Save results
        stress_results['random_forest'][test_type] = rf_test
        stress_results['neural_network'][test_type] = nn_test
        
        # Plot results
        for model_name, results in [('rf', rf_test), ('nn', nn_test)]:
            save_path = config.RESULTS_DIR / f'{model_name}_{test_type}_results.png'
            rf_stress.plot_results(results, f'{test_type.replace("_", " ").title()} Results', str(save_path))
    
    # Save Results
    print("\n7. Saving results...")
    
    # Combine all results
    final_results = {
        'random_forest': {
            'metrics': rf_metrics,
            'stress_tests': stress_results['random_forest']
        },
        'neural_network': {
            'metrics': nn_metrics,
            'training_history': nn_history,
            'stress_tests': stress_results['neural_network']
        },
        'label_mapping': {
            'original': list(range(1, num_classes + 1)),
            'shifted': list(range(num_classes)),
            'letters': class_labels
        }
    }
    
    # Save results
    DataUtils.save_results(final_results, str(config.RESULTS_DIR / 'final_results.json'))
    
    # Create model comparison plot for Accuracy, Recall, Precision and F1 Score
    Visualizer.plot_model_comparison(
        {
            'Random Forest': rf_metrics,
            'Neural Network': nn_metrics
        },
        str(config.RESULTS_DIR / 'model_comparison.png')
    )

    
    print("\nProject completed successfully!")
    print(f"Results saved in: {config.RESULTS_DIR}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise