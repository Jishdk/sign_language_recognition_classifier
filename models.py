# models.py

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Tuple

class SignNN(nn.Module):
    """Neural Network for sign language recognition."""
    
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)

class SignClassifier:
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        y_pred = self.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        
        # Ensure we have all classes represented in confusion matrix
        max_class = max(np.max(y_test), np.max(y_pred))
        conf_matrix = np.zeros((max_class + 1, max_class + 1), dtype=int)
        for true, pred in zip(y_test, y_pred):
            conf_matrix[true][pred] += 1
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix
        }

class RandomForest(SignClassifier):
    def __init__(self, n_trees: int = 100):
        self.model = RandomForestClassifier(
            n_estimators=n_trees,
            random_state=42
        )
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        print("Training Random Forest...")
        # Verify labels
        unique_labels = np.unique(y_train)
        print(f"Training with {len(unique_labels)} classes: {sorted(unique_labels)}")
        self.model.fit(X_train, y_train)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

class NeuralNetwork(SignClassifier):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int,
                 learning_rate: float = 0.001):
        # Force CPU usage for stability
        self.device = torch.device('cpu')
        self.model = SignNN(input_size, hidden_size, num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.num_classes = num_classes
        print(f"Neural Network initialized with {num_classes} output classes")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              batch_size: int = 32, num_epochs: int = 100) -> Dict:
        print("Training Neural Network...")
        
        # Verify labels are within bounds
        if np.max(y_train) >= self.num_classes or np.min(y_train) < 0:
            raise ValueError(f"Training labels must be in range 0-{self.num_classes-1}, "
                           f"found range {np.min(y_train)}-{np.max(y_train)}")
        if X_val is not None and (np.max(y_val) >= self.num_classes or np.min(y_val) < 0):
            raise ValueError(f"Validation labels must be in range 0-{self.num_classes-1}, "
                           f"found range {np.min(y_val)}-{np.max(y_val)}")
        
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.LongTensor(y_train).to(self.device)
        
        if X_val is not None:
            X_val = torch.FloatTensor(X_val).to(self.device)
            y_val = torch.LongTensor(y_val).to(self.device)
        
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        # Training loop
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            # Process mini-batches
            num_batches = (len(X_train) + batch_size - 1) // batch_size  # Ceiling division
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i + batch_size]
                batch_y = y_train[i:i + batch_size]
                
                # Forward pass
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Track metrics
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()
            
            # Calculate epoch metrics
            avg_loss = total_loss / num_batches
            accuracy = 100. * correct / total
            
            history['train_loss'].append(avg_loss)
            history['train_acc'].append(accuracy)
            
            # Validation phase
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val)
                    val_loss = self.criterion(val_outputs, y_val).item()
                    _, predicted = val_outputs.max(1)
                    val_acc = 100. * predicted.eq(y_val).sum().item() / len(y_val)
                    
                    history['val_loss'].append(val_loss)
                    history['val_acc'].append(val_acc)
                    
                print(f'Epoch {epoch:3d}: loss={avg_loss:.4f}, acc={accuracy:6.2f}%, '
                      f'val_loss={val_loss:.4f}, val_acc={val_acc:6.2f}%')
            else:
                print(f'Epoch {epoch:3d}: loss={avg_loss:.4f}, acc={accuracy:6.2f}%')
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X)
            _, predicted = outputs.max(1)
            
        return predicted.cpu().numpy()

__all__ = ['RandomForest', 'NeuralNetwork']