# sign_language_recognition_classifier
Group assignment 
# Sign Language Alphabet Recognition

A machine learning system for recognizing sign language alphabets using hand landmark detection.

## Project Overview

This project implements a sign language recognition system using hand landmark detection and machine learning. It compares the performance of two different classifiers (Random Forest and Neural Network) and includes comprehensive evaluation and stress testing.

## Features

* Hand landmark detection using MediaPipe
* Support for multiple machine learning models:
  - Random Forest Classifier
  - Neural Network
* Comprehensive evaluation system
* Stress testing for model robustness
* Detailed visualization tools

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd sign-language-recognition
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
sign_language_recognition/
├── data_processing.py    # Data loading and processing
├── hand_detector.py      # MediaPipe hand detection
├── models.py            # Model implementations
├── evaluation.py       # Evaluation and testing
├── utils.py           # Utility functions
├── config.py         # Configuration settings
└── main.py          # Main execution script
```

## Data Format

The project expects data in COCO format with the following structure:
```
data/
├── train/
│   ├── images/
│   └── _annotations.coco.json
├── valid/
│   ├── images/
│   └── _annotations.coco.json
└── test/
    ├── images/
    └── _annotations.coco.json
```

## Usage

1. Configure settings in `config.py`
2. Run the main script:
```bash
python main.py
```

## Output

The system generates:
* Model performance metrics
* Confusion matrices
* Training history plots
* Stress test results
* Model comparison visualizations

Results are saved in the configured results directory.

## Model Performance

The system evaluates models on:
* Accuracy
* Confusion matrix
* Stress test performance:
  - Scale variation
  - Noise resistance
  - Feature dropout

## Development

### Adding New Models

1. Inherit from `SignClassifier` in models.py
2. Implement required methods:
   - `train()`
   - `predict()`
   - `evaluate()`

### Adding Stress Tests

1. Add new test method in `StressTester` class
2. Update `run_all_tests()` to include new test
3. Add visualization for new test results

## Dependencies

* numpy
* opencv-python
* mediapipe
* torch
* scikit-learn
* matplotlib
* pandas

## Limitations

* Designed for single-hand gestures
* Requires good lighting conditions
* Limited to static gestures

## Future Improvements

* Support for dynamic gestures
* Real-time recognition
* Multiple hand support
* Additional model architectures
* More comprehensive stress tests


## Authors

Jishnu Harinandansingh

