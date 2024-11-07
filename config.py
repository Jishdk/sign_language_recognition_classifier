from pathlib import Path

class Config:
    """Configuration settings for sign language recognition project."""
    
    def __init__(self):
        # Dynamically set BASE_DIR based on the location of config.py
        self.BASE_DIR = Path(__file__).resolve().parent 
        self.DATA_DIR = self.BASE_DIR / "data"
        self.RESULTS_DIR = self.BASE_DIR / "results"
        
        # Data split ratios
        self.TRAIN_SPLIT = 0.7
        self.VAL_SPLIT = 0.15
        self.TEST_SPLIT = 0.15
        
        # Random Forest parameters
        self.RF_PARAMS = {
            'n_estimators': 100,     # Number of trees
            'max_depth': None,       # Maximum depth of trees (None = unlimited)
            'random_state': 42       # For reproducibility
        }
        
        # Neural Network parameters
        self.NN_PARAMS = {
            'hidden_sizes': [256, 128, 64],  # Sizes of hidden layers
            'learning_rate': 0.001,          # Learning rate
            'batch_size': 32,                # Batch size for training
            'num_epochs': 100                # Number of training epochs
        }
        
        # MediaPipe hand detection parameters
        self.MEDIAPIPE_PARAMS = {
            'max_num_hands': 1,                    # Maximum number of hands to detect
            'min_detection_confidence': 0.7,       # Minimum confidence for detection
            'min_tracking_confidence': 0.5         # Minimum confidence for tracking
        }
        
        # Visualization settings
        self.VIS_PARAMS = {
            'fig_size': (12, 8),           # Default figure size
            'font_size': 12,               # Default font size
            'dpi': 100                     # Figure resolution
        }

    def display(self):
        """Display current configuration settings."""
        print("\nCurrent Configuration:")
        print("----------------------")
        
        print("\nDirectories:")
        print(f"Base Directory: {self.BASE_DIR}")
        print(f"Data Directory: {self.DATA_DIR}")
        print(f"Results Directory: {self.RESULTS_DIR}")
        print(f"Logs Directory: {self.LOGS_DIR}")
        
        print("\nData Splits:")
        print(f"Training: {self.TRAIN_SPLIT:.1%}")
        print(f"Validation: {self.VAL_SPLIT:.1%}")
        print(f"Testing: {self.TEST_SPLIT:.1%}")
        
        print("\nRandom Forest Parameters:")
        for key, value in self.RF_PARAMS.items():
            print(f"{key}: {value}")
        
        print("\nNeural Network Parameters:")
        for key, value in self.NN_PARAMS.items():
            print(f"{key}: {value}")
        
        print("\nMediaPipe Parameters:")
        for key, value in self.MEDIAPIPE_PARAMS.items():
            print(f"{key}: {value}")
