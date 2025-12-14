"""
Logistic Regression Implementation from Scratch using NumPy

This module implements logistic regression with:
- Binary cross-entropy loss
- Sigmoid activation function
- Gradient descent optimization
- L2 regularization
"""

import numpy as np
from typing import Optional, Tuple, List


class LogisticRegression:
    """
    Logistic Regression model implemented from scratch.
    
    The model learns weights w and bias b such that:
    p = sigmoid(X @ w + b)
    y_pred = 1 if p >= 0.5, else 0
    
    Training uses gradient descent to minimize binary cross-entropy loss.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        regularization: float = 0.0,
        verbose: bool = False
    ):
        """
        Initialize Logistic Regression model.
        
        Parameters:
        -----------
        learning_rate : float
            Step size for gradient descent
        max_iterations : int
            Maximum number of gradient descent iterations
        tolerance : float
            Convergence threshold
        regularization : float
            L2 regularization strength (lambda)
        verbose : bool
            Print training progress
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.regularization = regularization
        self.verbose = verbose
        
        # Model parameters
        self.weights: Optional[np.ndarray] = None
        self.bias: Optional[float] = None
        
        # Training history
        self.loss_history: List[float] = []
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Compute sigmoid activation function.
        
        σ(z) = 1 / (1 + exp(-z))
        
        Parameters:
        -----------
        z : np.ndarray
            Input values
        
        Returns:
        --------
        np.ndarray
            Sigmoid output (values in [0, 1])
        """
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _initialize_parameters(self, n_features: int):
        """Initialize weights and bias to small random values."""
        self.weights = np.random.normal(0, 0.01, size=(n_features,))
        self.bias = 0.0
    
    def _compute_loss(
        self,
        X: np.ndarray,
        y: np.ndarray,
        probabilities: np.ndarray
    ) -> float:
        """
        Compute binary cross-entropy loss with optional L2 regularization.
        
        Loss = -(1/n) * [y*log(p) + (1-y)*log(1-p)] + (λ/2) * ||w||²
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        y : np.ndarray
            True binary labels (n_samples,)
        probabilities : np.ndarray
            Predicted probabilities (n_samples,)
        
        Returns:
        --------
        float : Loss value
        """
        n_samples = X.shape[0]
        
        # Clip probabilities to avoid log(0)
        probabilities = np.clip(probabilities, 1e-15, 1 - 1e-15)
        
        # Binary cross-entropy component
        bce = -np.mean(y * np.log(probabilities) + (1 - y) * np.log(1 - probabilities))
        
        # L2 regularization component
        l2_reg = (self.regularization / 2) * np.sum(self.weights ** 2)
        
        return bce + l2_reg
    
    def _compute_gradients(
        self,
        X: np.ndarray,
        y: np.ndarray,
        probabilities: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Compute gradients of loss with respect to weights and bias.
        
        ∂L/∂w = (1/n) * X^T @ (p - y) + λ * w
        ∂L/∂b = (1/n) * sum(p - y)
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        y : np.ndarray
            True binary labels (n_samples,)
        probabilities : np.ndarray
            Predicted probabilities (n_samples,)
        
        Returns:
        --------
        Tuple[np.ndarray, float] : (weight_gradients, bias_gradient)
        """
        n_samples = X.shape[0]
        
        # Error term
        error = probabilities - y
        
        # Gradient w.r.t. weights
        weight_grad = (1 / n_samples) * X.T @ error + self.regularization * self.weights
        
        # Gradient w.r.t. bias
        bias_grad = (1 / n_samples) * np.sum(error)
        
        return weight_grad, bias_grad
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegression':
        """
        Train the logistic regression model using gradient descent.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        y : np.ndarray
            Binary target labels (n_samples,)
        
        Returns:
        --------
        self : LogisticRegression
            Returns self for method chaining
        """
        # Ensure inputs are numpy arrays
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        
        # Validate binary labels
        unique_labels = np.unique(y)
        if not np.all(np.isin(unique_labels, [0, 1])):
            raise ValueError("Labels must be binary (0 or 1)")
        
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self._initialize_parameters(n_features)
        
        # Training loop
        self.loss_history = []
        prev_loss = float('inf')
        
        for iteration in range(self.max_iterations):
            # Forward pass: compute logits and probabilities
            logits = X @ self.weights + self.bias
            probabilities = self._sigmoid(logits)
            
            # Compute loss
            loss = self._compute_loss(X, y, probabilities)
            self.loss_history.append(loss)
            
            # Check convergence
            if abs(prev_loss - loss) < self.tolerance:
                if self.verbose:
                    print(f"Converged at iteration {iteration}")
                break
            
            # Compute gradients
            weight_grad, bias_grad = self._compute_gradients(X, y, probabilities)
            
            # Update parameters using gradient descent
            self.weights -= self.learning_rate * weight_grad
            self.bias -= self.learning_rate * bias_grad
            
            prev_loss = loss
            
            # Print progress
            if self.verbose and (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iterations}, Loss: {loss:.6f}")
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        
        Returns:
        --------
        np.ndarray
            Predicted probabilities (n_samples,)
        """
        if self.weights is None:
            raise ValueError("Model must be fitted before making predictions")
        
        X = np.asarray(X)
        logits = X @ self.weights + self.bias
        return self._sigmoid(logits)
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary class labels.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        threshold : float
            Decision threshold (default: 0.5)
        
        Returns:
        --------
        np.ndarray
            Predicted binary labels (n_samples,)
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute accuracy score.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            True binary labels
        
        Returns:
        --------
        float
            Accuracy score
        """
        y_pred = self.predict(X)
        y = np.asarray(y).ravel()
        return np.mean(y_pred == y)

