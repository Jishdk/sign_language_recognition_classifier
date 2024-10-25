# data_processing.py

import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from sklearn.model_selection import train_test_split

class SignLanguageDataset:
    def __init__(self, data_path: str, hand_detector):
        """Initialize dataset handler."""
        self.data_path = Path(data_path)
        self.hand_detector = hand_detector
        
        # Load data annotations
        print("Loading dataset...")
        self.train_data = self._load_split('train')
        self.valid_data = self._load_split('valid')
        self.test_data = self._load_split('test')
        
        # Store category mapping
        self.categories = self._get_categories()
        print(f"Found {len(self.categories)} categories")
    
    def _load_split(self, split: str) -> Dict:
        """Load data annotations for a specific split."""
        annotation_file = self.data_path / split / '_annotations.coco.json'
        with open(annotation_file, 'r') as f:
            return json.load(f)
    
    def _get_categories(self) -> Dict[int, str]:
        """Get mapping of category IDs to names."""
        categories = self.train_data['categories']
        return {cat['id']: cat['name'] for cat in categories}
    
    def process_images(self, split: str = 'train') -> Tuple[np.ndarray, np.ndarray]:
        """Process images and extract hand landmarks."""
        # Get data for requested split
        if split == 'train':
            data = self.train_data
        elif split == 'valid':
            data = self.valid_data
        else:
            data = self.test_data
            
        features = []
        labels = []
        processed = 0
        skipped = 0
        
        # Create image ID to filename mapping
        image_map = {img['id']: img['file_name'] for img in data['images']}
        
        # Process each annotation
        print(f"Processing {split} images...")
        for ann in data['annotations']:
            try:
                # Load image
                image_path = self.data_path / split / image_map[ann['image_id']]
                image = cv2.imread(str(image_path))
                if image is None:
                    print(f"Could not load image: {image_path}")
                    skipped += 1
                    continue
                
                # Detect landmarks
                landmarks, success = self.hand_detector.detect_landmarks(image)
                
                if success:
                    features.append(landmarks)
                    labels.append(ann['category_id'])
                    processed += 1
                else:
                    skipped += 1
                    
            except Exception as e:
                print(f"Error processing image {image_path}: {str(e)}")
                skipped += 1
        
        print(f"Processed {processed} images, skipped {skipped} images")
        
        if processed == 0:
            raise ValueError("No images were successfully processed. Check image paths and hand detection.")
            
        return np.array(features), np.array(labels)
    
    def prepare_data(self, valid_size: float = 0.15, test_size: float = 0.15) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Prepare dataset splits."""
        print("Processing training data...")
        features, labels = self.process_images('train')
        
        print("Creating data splits...")
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=42
        )
        
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train, y_train, test_size=valid_size/(1-test_size), random_state=42
        )
        
        return {
            'train': (X_train, y_train),
            'valid': (X_valid, y_valid),
            'test': (X_test, y_test)
        }

__all__ = ['SignLanguageDataset']