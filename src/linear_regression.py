"""
Linear Regression Implementation from Scratch using NumPy

This module implements linear regression with:
- Mean Squared Error loss
- Gradient descent optimization
- L2 regularization (Ridge regression)
- Learning rate scheduling
"""

import numpy as np
from typing import Optional, Tuple, List


class LinearRegression:
    """
    Linear Regression model implemented from scratch.
    
    The model learns weights w and bias b such that:
    y_pred = X @ w + b
    
    Training uses gradient descent to minimize MSE loss with optional L2 regularization.
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
        Initialize Linear Regression model.
        
        Parameters:
        -----------
        learning_rate : float
            Step size for gradient descent
        max_iterations : int
            Maximum number of gradient descent iterations
        tolerance : float
            Convergence threshold (stop if loss change < tolerance)
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
        
        # Model parameters (will be initialized during fit)
        self.weights: Optional[np.ndarray] = None
        self.bias: Optional[float] = None
        
        # Training history
        self.loss_history: List[float] = []
    
    def _initialize_parameters(self, n_features: int):
        """Initialize weights and bias to small random values."""
        self.weights = np.random.normal(0, 0.01, size=(n_features,))
        self.bias = 0.0
    
    def _compute_loss(
        self,
        X: np.ndarray,
        y: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Compute Mean Squared Error loss with optional L2 regularization.
        
        Loss = (1/n) * ||y - y_pred||² + (λ/2) * ||w||²
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        y : np.ndarray
            True target values (n_samples,)
        y_pred : np.ndarray
            Predicted values (n_samples,)
        
        Returns:
        --------
        float : Loss value
        """
        n_samples = X.shape[0]
        
        # MSE component
        mse = np.mean((y - y_pred) ** 2)
        
        # L2 regularization component
        l2_reg = (self.regularization / 2) * np.sum(self.weights ** 2)
        
        return mse + l2_reg
    
    def _compute_gradients(
        self,
        X: np.ndarray,
        y: np.ndarray,
        y_pred: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Compute gradients of loss with respect to weights and bias.
        
        ∂L/∂w = -(2/n) * X^T @ (y - y_pred) + λ * w
        ∂L/∂b = -(2/n) * sum(y - y_pred)
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        y : np.ndarray
            True target values (n_samples,)
        y_pred : np.ndarray
            Predicted values (n_samples,)
        
        Returns:
        --------
        Tuple[np.ndarray, float] : (weight_gradients, bias_gradient)
        """
        n_samples = X.shape[0]
        
        # Error term
        error = y - y_pred
        
        # Gradient w.r.t. weights
        weight_grad = -(2 / n_samples) * X.T @ error + self.regularization * self.weights
        
        # Gradient w.r.t. bias
        bias_grad = -(2 / n_samples) * np.sum(error)
        
        return weight_grad, bias_grad
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        Train the linear regression model using gradient descent.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        y : np.ndarray
            Target values (n_samples,)
        
        Returns:
        --------
        self : LinearRegression
            Returns self for method chaining
        """
        # Ensure inputs are numpy arrays
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        
        # Add bias term by prepending column of ones (alternative approach)
        # We'll keep bias separate for clarity
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self._initialize_parameters(n_features)
        
        # Training loop
        self.loss_history = []
        prev_loss = float('inf')
        
        for iteration in range(self.max_iterations):
            # Forward pass: compute predictions
            y_pred = X @ self.weights + self.bias
            
            # Compute loss
            loss = self._compute_loss(X, y, y_pred)
            self.loss_history.append(loss)
            
            # Check convergence
            if abs(prev_loss - loss) < self.tolerance:
                if self.verbose:
                    print(f"Converged at iteration {iteration}")
                break
            
            # Compute gradients
            weight_grad, bias_grad = self._compute_gradients(X, y, y_pred)
            
            # Update parameters using gradient descent
            self.weights -= self.learning_rate * weight_grad
            self.bias -= self.learning_rate * bias_grad
            
            prev_loss = loss
            
            # Print progress
            if self.verbose and (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iterations}, Loss: {loss:.6f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        
        Returns:
        --------
        np.ndarray
            Predicted values (n_samples,)
        """
        if self.weights is None:
            raise ValueError("Model must be fitted before making predictions")
        
        X = np.asarray(X)
        return X @ self.weights + self.bias
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute R² score (coefficient of determination).
        
        R² = 1 - (SS_res / SS_tot)
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            True target values
        
        Returns:
        --------
        float
            R² score
        """
        y_pred = self.predict(X)
        y = np.asarray(y).ravel()
        
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        return 1 - (ss_res / ss_tot)

