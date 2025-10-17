import numpy as np

def create_highly_complex_dataset(n_samples=500000, n_features=15, noise_level=0.3, random_state=42):
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate base data with different distributions
    X = np.column_stack([
        np.random.normal(0, 1, n_samples),      # Normal
        np.random.uniform(-2, 2, n_samples),    # Uniform
        np.random.exponential(1, n_samples),    # Exponential
        np.random.lognormal(0, 1, n_samples),   # Log-normal
        np.random.beta(2, 5, n_samples),        # Beta
        np.random.gamma(2, 2, n_samples),       # Gamma
        np.random.chisquare(3, n_samples),      # Chi-squared
        np.random.normal(1, 0.5, n_samples),    # Shifted normal
        np.random.uniform(-3, 3, n_samples),    # Wider uniform
        np.random.normal(0, 2, n_samples),      # Higher variance normal
        np.random.logistic(0, 1, n_samples),    # Logistic
        np.random.rayleigh(1, n_samples),       # Rayleigh
        np.random.poisson(3, n_samples),        # Poisson
        np.random.geometric(0.3, n_samples),    # Geometric
        np.random.weibull(1.5, n_samples)       # Weibull
    ])
    
    # Extremely complex target with multiple non-linear transformations
    complex_target = (
        # Trigonometric interactions
        np.sin(X[:, 0] * X[:, 1]) * np.cos(X[:, 2]) +
        np.tan(X[:, 3] * 0.5) * np.arctan(X[:, 4]) +
        
        # Exponential and logarithmic interactions
        np.exp(X[:, 5] * 0.3) * np.log1p(np.abs(X[:, 6])) +
        np.power(X[:, 7], 2) * np.sqrt(np.abs(X[:, 8])) +
        
        # Polynomial interactions (high degree)
        X[:, 0]**3 * X[:, 1]**2 +
        X[:, 2]**4 * X[:, 3] +
        X[:, 4]**2 * X[:, 5]**3 +
        
        # Conditional relationships
        np.where(X[:, 6] > 0, X[:, 7]**2, -X[:, 7]**2) +
        np.where(X[:, 8] < 0, np.sin(X[:, 9]), np.cos(X[:, 9])) +
        
        # Multi-feature interactions
        X[:, 0] * X[:, 1] * X[:, 2] +
        X[:, 3] * X[:, 4] * X[:, 5] +
        X[:, 6] * X[:, 7] * X[:, 8] +
        
        # Complex periodic patterns
        np.sin(X[:, 0] * 2 * np.pi) * np.cos(X[:, 1] * 3 * np.pi) +
        np.sin(X[:, 2] + X[:, 3]) * np.cos(X[:, 4] - X[:, 5]) +
        
        # Saturation effects
        np.tanh(X[:, 9] * 2) * np.arctan(X[:, 10] * 3) +
        
        # Piecewise linear with multiple breakpoints
        np.piecewise(X[:, 11], 
                    [X[:, 11] < -1, (X[:, 11] >= -1) & (X[:, 11] < 1), X[:, 11] >= 1],
                    [lambda x: -x**2, lambda x: x**3, lambda x: np.sqrt(x)]) +
        
        # Random feature combinations
        X[:, 12] * np.sin(X[:, 13]) * np.cos(X[:, 14]) +
        X[:, 13] * np.tanh(X[:, 12]) * np.arctan(X[:, 11]) +
        
        # Highly non-linear transformations
        np.log1p(np.abs(X[:, 0] * X[:, 1] * X[:, 2])) +
        np.exp(np.sin(X[:, 3]) + np.cos(X[:, 4])) +
        
        # Interaction with feature products
        (X[:, 5] * X[:, 6]) / (1 + np.abs(X[:, 7] * X[:, 8])) +
        np.sin(X[:, 9] * X[:, 10]) * np.cos(X[:, 11] * X[:, 12])
    )
    
    # Add significant noise
    y = complex_target + np.random.normal(0, noise_level * np.std(complex_target), n_samples)
    
    return X, y
