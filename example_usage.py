from dataset_generator import create_extended_dataset

# Generate dataset
X, y = create_extended_dataset(n_samples=50000, random_state=42)

print(f"Dataset shape: {X.shape}")
print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
print("Ready for model training!")
