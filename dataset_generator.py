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

def create_extended_dataset(n_samples=500000, random_state=42):
    if random_state is not None:
        np.random.seed(random_state)
    
    # Original 15 complex features
    X_original, y = create_highly_complex_dataset(n_samples, random_state=random_state)
    
    # 5 numeric features with moderate complexity
    X_moderate = np.column_stack([
        # Moderate complexity features - some non-linear relationships
        np.sin(X_original[:, 0] * 0.5) + np.cos(X_original[:, 1] * 0.3),
        np.log1p(np.abs(X_original[:, 2] * X_original[:, 3])),
        np.tanh(X_original[:, 4] * 0.7) * np.arctan(X_original[:, 5] * 0.4),
        np.sqrt(np.abs(X_original[:, 6])) + X_original[:, 7] * 0.2,
        np.exp(X_original[:, 8] * 0.1) - np.exp(X_original[:, 9] * -0.1)
    ])
    
    # 5 simple numeric features
    X_simple = np.column_stack([
        np.random.normal(0, 1, n_samples),
        np.random.uniform(-1, 1, n_samples),
        np.random.exponential(0.5, n_samples),
        np.random.normal(0.5, 0.3, n_samples),
        np.random.beta(1, 1, n_samples)
    ])
    
    # 15 categorical features with different number of categories
    categorical_features = []
    n_categories_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 50]
    
    for n_categories in n_categories_list:
        # Generate categorical features with different distributions
        if n_categories <= 5:
            # More balanced categories
            cat_feature = np.random.randint(0, n_categories, n_samples)
        else:
            # Some categories more frequent than others
            probs = np.random.dirichlet(np.ones(n_categories) * 2)
            cat_feature = np.random.choice(n_categories, n_samples, p=probs)
        
        categorical_features.append(cat_feature)
    
    X_categorical = np.column_stack(categorical_features)
    
    # Combine all features
    X_combined = np.column_stack([X_original, X_moderate, X_simple, X_categorical])
    
    y_updated = y + (
        # Add some moderate influence from new numeric features
        X_moderate[:, 0] * 0.3 +
        X_moderate[:, 1] * 0.2 +
        X_simple[:, 0] * 0.1 +
        X_simple[:, 1] * 0.05 +
        # Add some categorical influence (using first 5 categorical features)
        (X_categorical[:, 0] / n_categories_list[0]) * 0.4 +
        (X_categorical[:, 1] / n_categories_list[1]) * 0.3 +
        (X_categorical[:, 2] / n_categories_list[2]) * 0.2
    )
    
    return X_combined, y_updated
