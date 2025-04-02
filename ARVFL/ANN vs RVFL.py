import numpy as np
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime
from sklearn.datasets import make_moons
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Create a directory to store results
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# Generate timestamp for file naming
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Generate synthetic dataset
X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
scaler = StandardScaler()
X = scaler.fit_transform(X)

### TRAIN TRADITIONAL NN (MLP)
start_time = time.time()
mlp = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', max_iter=500, random_state=42)
mlp.fit(X, y)
mlp_time = time.time() - start_time  # Compute training time
y_pred_mlp = mlp.predict(X)
acc_mlp = accuracy_score(y, y_pred_mlp)

### TRAIN RVFL NETWORK
start_time = time.time()
np.random.seed(42)
input_dim = X.shape[1]
hidden_neurons = 10

# Generate fixed random weights and biases for the hidden layer
W = np.random.randn(input_dim, hidden_neurons)
b = np.random.randn(hidden_neurons)

# Compute hidden layer activation (sigmoid function)
H = 1 / (1 + np.exp(-(X @ W + b)))

# Train output layer using Ridge Regression
ridge = Ridge(alpha=1e-2)
ridge.fit(H, y)
rvfl_time = time.time() - start_time  # Compute training time
y_pred_rvfl = (ridge.predict(H) > 0.5).astype(int)
acc_rvfl = accuracy_score(y, y_pred_rvfl)

# Print Time Complexity Results
print(f"Traditional NN (MLP) - Training Time: {mlp_time:.4f} seconds, Accuracy: {acc_mlp:.2f}")
print(f"RVFL Network - Training Time: {rvfl_time:.4f} seconds, Accuracy: {acc_rvfl:.2f}")

# Plot and save decision boundaries
def plot_decision_boundary(model, X, y, title, filename, rvfl=False):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    
    if rvfl:
        H_test = 1 / (1 + np.exp(-(np.c_[xx.ravel(), yy.ravel()] @ W + b)))
        Z = (model.predict(H_test) > 0.5).astype(int)
    else:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Spectral)
    plt.title(title)
    
    # Save the plot
    save_path = os.path.join(results_dir, filename)
    plt.savefig(save_path, dpi=300)
    print(f"Saved plot: {save_path}")
    plt.close()

# Save decision boundary plots with timestamps
plot_decision_boundary(mlp, X, y, f"Traditional NN (MLP) - Accuracy: {acc_mlp:.2f}", f"MLP_{timestamp}.png")
plot_decision_boundary(ridge, X, y, f"RVFL Network - Accuracy: {acc_rvfl:.2f}", f"RVFL_{timestamp}.png", rvfl=True)


