# Linear and Logistic Regression From Scratch Using NumPy

## Project Description

This project implements fundamental machine learning algorithms from scratch using only NumPy, providing deep insight into the mathematical foundations of linear and logistic regression. By building these algorithms without high-level libraries, we gain a thorough understanding of gradient descent, loss functions, and regularization.

**Why This Project Matters**:
- **Deep Understanding**: Implementing algorithms from scratch reveals the underlying mathematics and computational details that are often hidden in high-level libraries
- **Educational Value**: Perfect for learning how machine learning algorithms actually work under the hood
- **Foundation Building**: These are fundamental algorithms that form the basis for more complex models
- **Debugging Skills**: Understanding the implementation helps debug issues in production ML systems
- **Customization**: Building from scratch allows for custom modifications and optimizations

**Key Learning Outcomes**:
- Mathematical derivations of gradient descent
- Vectorized operations for efficient computation
- Loss function implementations and their properties
- Regularization techniques and their impact
- Convergence analysis and optimization strategies

## Dataset Description

**Dataset Name**: California Housing Dataset

**Source**: Scikit-learn datasets (originally from 1990 US Census, StatLib repository)

**Dataset Details**:
- **Number of samples**: 20,640 housing districts from California
- **Number of features**: 8 numerical features
  - **MedInc**: Median income in block group
  - **HouseAge**: Median house age in block group
  - **AveRooms**: Average number of rooms per household
  - **AveBedrms**: Average number of bedrooms per household
  - **Population**: Block group population
  - **AveOccup**: Average number of household members
  - **Latitude**: Block group latitude
  - **Longitude**: Block group longitude
- **Target variable**: Median house value (continuous, in hundreds of thousands of dollars)
- **Task**: Regression (predicting median house value)
- **Data quality**: Clean dataset with no missing values

**Why This Dataset**:
- **Well-understood problem**: Predicting median house values is a classic regression task
- **Tabular data**: Clean, structured data perfect for linear models
- **Real-world relevance**: Housing prices are influenced by multiple factors, making it a practical application
- **Size**: Large enough to demonstrate scalability concerns (~20,000 samples)
- **Feature diversity**: Mix of geographical, demographic, and structural features
- **Standard benchmark**: Widely used in ML education and research

**Data Loading**:
```python
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing(as_frame=True)
X, y = housing.data, housing.target
```

**IMPORTANT**: No synthetic or hard-coded data is used in this project. All experiments use the real California Housing dataset loaded from scikit-learn's `fetch_california_housing()` function.

## Project Structure

```
project1_ml_from_scratch/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_linear_regression_scratch.ipynb
│   ├── 03_logistic_regression_scratch.ipynb
│   └── 04_comparison_sklearn.ipynb
└── src/
    ├── linear_regression.py
    └── logistic_regression.py
```

## Key Implementations

### Linear Regression
- Mean Squared Error (MSE) loss function
- Gradient descent optimization
- Learning rate scheduling
- L2 regularization (Ridge regression)
- Convergence monitoring

### Logistic Regression
- Binary cross-entropy loss function
- Sigmoid activation function
- Gradient descent with regularization
- Decision boundary visualization

## Learning Objectives

1. **Mathematical Foundations**: Understand the mathematical derivations behind gradient descent
2. **Implementation Details**: Learn how to efficiently compute gradients using vectorized operations
3. **Hyperparameter Tuning**: Experience the impact of learning rate and regularization strength
4. **Comparison Analysis**: Validate implementations against sklearn's optimized versions

## Results

The notebooks demonstrate:
- Loss convergence curves
- Prediction accuracy comparisons
- Overfitting behavior with/without regularization
- Performance benchmarks vs sklearn

## Usage

1. Install dependencies: `pip install -r requirements.txt`
2. Open Jupyter notebooks in order (01 → 04)
3. Run all cells to reproduce experiments
4. Modify hyperparameters to explore different behaviors

## Requirements

- Python 3.8+
- NumPy
- Matplotlib
- Scikit-learn (for comparison)
- Pandas
- Jupyter

