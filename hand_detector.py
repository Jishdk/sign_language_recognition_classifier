# hand_detector.py

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple

class HandDetector:
    def __init__(self, max_num_hands: int = 1,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5):
        """Initialize MediaPipe hand detector."""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils

    def detect_landmarks(self, image: np.ndarray) -> Tuple[List, bool]:
        """
        Detect hand landmarks in an image.
        
        Args:
            image: Input image (BGR format)
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get landmarks
        results = self.hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            # Get first hand's landmarks
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Extract coordinates
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            return landmarks, True
        
        return [], False

__all__ = ['HandDetector']