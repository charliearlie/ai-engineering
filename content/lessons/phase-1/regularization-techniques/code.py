"""
Lesson 9: Regularization Techniques
Preventing overfitting and improving generalization
"""

import numpy as np
import matplotlib.pyplot as plt

print("="*60)
print("LESSON 9: REGULARIZATION TECHNIQUES")
print("="*60)

# -----------------------------------------------------------------------------
# PART 1: DEMONSTRATING OVERFITTING
# -----------------------------------------------------------------------------
print("\n1. WHAT IS OVERFITTING?")
print("-" * 40)

# Generate a simple dataset with noise
np.random.seed(42)
n_samples = 50

# True function: y = sin(x) + small noise
X_train = np.linspace(0, 4*np.pi, n_samples)
y_true = np.sin(X_train)
y_train = y_true + 0.1 * np.random.randn(n_samples)

# Test data (more points for smooth curve)
X_test = np.linspace(0, 4*np.pi, 200)
y_test_true = np.sin(X_test)

print(f"Training data: {n_samples} points")
print(f"True function: y = sin(x)")
print(f"Added noise: ~N(0, 0.1)")

# Fit polynomials of different degrees
degrees = [3, 9, 15]
plt.figure(figsize=(15, 5))

for idx, degree in enumerate(degrees):
    # Fit polynomial
    coeffs = np.polyfit(X_train, y_train, degree)
    poly = np.poly1d(coeffs)
    
    # Predictions
    y_train_pred = poly(X_train)
    y_test_pred = poly(X_test)
    
    # Calculate errors
    train_error = np.mean((y_train - y_train_pred)**2)
    test_error = np.mean((y_test_true - y_test_pred)**2)
    
    # Plot
    plt.subplot(1, 3, idx+1)
    plt.scatter(X_train, y_train, alpha=0.6, s=40, label='Training data')
    plt.plot(X_test, y_test_true, 'g--', linewidth=2, label='True function')
    plt.plot(X_test, y_test_pred, 'r-', linewidth=2, label=f'Polynomial deg={degree}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Degree {degree}\nTrain MSE: {train_error:.4f}, Test MSE: {test_error:.4f}')
    plt.legend()
    plt.ylim(-2, 2)
    plt.grid(True, alpha=0.3)

plt.suptitle('Overfitting: Higher Degree = Better Training Fit, Worse Generalization', fontsize=14)
plt.tight_layout()
plt.show()

print("\nNotice how:")
print("  - Low degree (3): Underfits - too simple")
print("  - High degree (15): Overfits - memorizes noise")
print("  - Sweet spot (9): Balances fit and generalization")

# -----------------------------------------------------------------------------
# PART 2: L2 REGULARIZATION (WEIGHT DECAY)
# -----------------------------------------------------------------------------
print("\n\n2. L2 REGULARIZATION (WEIGHT DECAY)")
print("-" * 40)

class LinearModelL2:
    """Linear model with L2 regularization"""
    
    def __init__(self, n_features, lambda_reg=0.01):
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0
        self.lambda_reg = lambda_reg
        
    def forward(self, X):
        return np.dot(X, self.weights) + self.bias
    
    def loss(self, X, y):
        predictions = self.forward(X)
        # MSE loss + L2 penalty
        mse_loss = np.mean((predictions - y)**2)
        l2_penalty = self.lambda_reg * np.sum(self.weights**2)
        return mse_loss + l2_penalty, mse_loss
    
    def train_step(self, X, y, learning_rate=0.01):
        # Forward pass
        predictions = self.forward(X)
        
        # Gradients (including L2 term)
        error = predictions - y
        grad_w = 2 * np.dot(X.T, error) / len(y) + 2 * self.lambda_reg * self.weights
        grad_b = 2 * np.mean(error)
        
        # Update
        self.weights -= learning_rate * grad_w
        self.bias -= learning_rate * grad_b

# Compare different L2 strengths
print("Training with different L2 regularization strengths...")

# Create polynomial features
def create_polynomial_features(X, degree):
    """Create polynomial features up to specified degree"""
    n = len(X)
    features = np.zeros((n, degree))
    for d in range(degree):
        features[:, d] = X ** (d + 1)
    return features

# High-degree polynomial features (prone to overfitting)
degree = 15
X_train_poly = create_polynomial_features(X_train, degree)
X_test_poly = create_polynomial_features(X_test, degree)

# Train with different regularization strengths
lambdas = [0, 0.001, 0.01, 0.1]
models = {}
histories = {}

for lambda_reg in lambdas:
    model = LinearModelL2(n_features=degree, lambda_reg=lambda_reg)
    history = {'total_loss': [], 'mse_loss': [], 'weights': []}
    
    # Train
    for epoch in range(1000):
        model.train_step(X_train_poly, y_train, learning_rate=0.1)
        
        if epoch % 50 == 0:
            total_loss, mse_loss = model.loss(X_train_poly, y_train)
            history['total_loss'].append(total_loss)
            history['mse_loss'].append(mse_loss)
            history['weights'].append(np.copy(model.weights))
    
    models[lambda_reg] = model
    histories[lambda_reg] = history

# Visualize effects of L2 regularization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for idx, lambda_reg in enumerate(lambdas):
    model = models[lambda_reg]
    
    # Predictions
    y_test_pred = model.forward(X_test_poly)
    
    # Plot fit
    ax = axes.flatten()[idx]
    ax.scatter(X_train, y_train, alpha=0.6, s=40, label='Training data')
    ax.plot(X_test, y_test_true, 'g--', linewidth=2, label='True function')
    ax.plot(X_test, y_test_pred, 'r-', linewidth=2, label='Model prediction')
    
    # Calculate test error
    test_mse = np.mean((y_test_true - y_test_pred)**2)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'λ = {lambda_reg}\nTest MSE: {test_mse:.4f}, ||w||²: {np.sum(model.weights**2):.2f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-2, 2)

plt.suptitle('L2 Regularization: Larger λ → Smaller Weights → Simpler Functions', fontsize=14)
plt.tight_layout()
plt.show()

print("\nL2 Regularization effects:")
print("  - λ = 0: No regularization, overfits")
print("  - λ = 0.001: Slight smoothing")  
print("  - λ = 0.01: Good balance")
print("  - λ = 0.1: May underfit")

# Show weight magnitudes
print("\nWeight magnitudes (||w||²):")
for lambda_reg in lambdas:
    weight_norm = np.sum(models[lambda_reg].weights**2)
    print(f"  λ = {lambda_reg}: {weight_norm:.4f}")

# -----------------------------------------------------------------------------
# PART 3: L1 REGULARIZATION (SPARSITY)
# -----------------------------------------------------------------------------
print("\n\n3. L1 REGULARIZATION (LASSO)")
print("-" * 40)

class LinearModelL1:
    """Linear model with L1 regularization"""
    
    def __init__(self, n_features, lambda_reg=0.01):
        self.weights = np.random.randn(n_features) * 0.1
        self.bias = 0.0
        self.lambda_reg = lambda_reg
        
    def forward(self, X):
        return np.dot(X, self.weights) + self.bias
    
    def train_step(self, X, y, learning_rate=0.01):
        # Forward pass
        predictions = self.forward(X)
        
        # Gradients (including L1 term)
        error = predictions - y
        grad_w = 2 * np.dot(X.T, error) / len(y) + self.lambda_reg * np.sign(self.weights)
        grad_b = 2 * np.mean(error)
        
        # Update
        self.weights -= learning_rate * grad_w
        self.bias -= learning_rate * grad_b

# Create dataset with many irrelevant features
n_features = 20
n_relevant = 5  # Only first 5 features are relevant

# Generate features
X_train_sparse = np.random.randn(n_samples, n_features)
# Make output depend only on first few features
true_weights = np.zeros(n_features)
true_weights[:n_relevant] = np.random.randn(n_relevant)
y_train_sparse = np.dot(X_train_sparse, true_weights) + 0.1 * np.random.randn(n_samples)

print(f"Dataset: {n_features} features, but only {n_relevant} are relevant")
print(f"True relevant features: {np.where(true_weights != 0)[0]}")

# Train with L1 and L2 for comparison
l1_model = LinearModelL1(n_features, lambda_reg=0.1)
l2_model = LinearModelL2(n_features, lambda_reg=0.1)

# Train both
for epoch in range(500):
    l1_model.train_step(X_train_sparse, y_train_sparse, learning_rate=0.01)
    l2_model.train_step(X_train_sparse, y_train_sparse, learning_rate=0.01)

# Compare weights
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(range(n_features), np.abs(l1_model.weights), alpha=0.6, label='L1 weights')
plt.axvline(x=n_relevant-0.5, color='r', linestyle='--', label='Relevant/Irrelevant boundary')
plt.xlabel('Feature index')
plt.ylabel('|Weight|')
plt.title('L1 Regularization: Sparse Weights')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.bar(range(n_features), np.abs(l2_model.weights), alpha=0.6, label='L2 weights')
plt.axvline(x=n_relevant-0.5, color='r', linestyle='--', label='Relevant/Irrelevant boundary')
plt.xlabel('Feature index')
plt.ylabel('|Weight|')
plt.title('L2 Regularization: Small but Non-zero Weights')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Count zero weights
l1_zeros = np.sum(np.abs(l1_model.weights) < 0.01)
l2_zeros = np.sum(np.abs(l2_model.weights) < 0.01)

print(f"\nNumber of near-zero weights (<0.01):")
print(f"  L1: {l1_zeros}/{n_features} weights")
print(f"  L2: {l2_zeros}/{n_features} weights")
print("\nL1 creates sparsity - many weights become exactly zero!")

# -----------------------------------------------------------------------------
# PART 4: DROPOUT
# -----------------------------------------------------------------------------
print("\n\n4. DROPOUT - RANDOM DEACTIVATION")
print("-" * 40)

class DropoutLayer:
    """Dropout layer implementation"""
    
    def __init__(self, dropout_rate=0.5):
        self.dropout_rate = dropout_rate
        self.mask = None
        
    def forward(self, x, training=True):
        if training:
            # Create random mask
            self.mask = np.random.random(x.shape) > self.dropout_rate
            # Apply mask and scale
            return x * self.mask / (1 - self.dropout_rate)
        else:
            # No dropout during testing
            return x
    
    def backward(self, grad_output):
        # Gradient only flows through active neurons
        return grad_output * self.mask / (1 - self.dropout_rate)

# Demonstrate dropout on a simple network
print("Demonstrating dropout behavior...")

# Create sample activations
activations = np.ones((1, 10))  # 10 neurons, all outputting 1
dropout = DropoutLayer(dropout_rate=0.5)

# Show multiple forward passes during training
print("\nTraining mode (different neurons drop each time):")
for i in range(3):
    output = dropout.forward(activations, training=True)
    active_neurons = np.where(output[0] > 0)[0]
    print(f"  Pass {i+1}: Active neurons: {active_neurons}, Mean output: {np.mean(output):.2f}")

# Testing mode
print("\nTesting mode (all neurons active):")
output = dropout.forward(activations, training=False)
print(f"  All neurons active, Mean output: {np.mean(output):.2f}")

# Visualize dropout effect on overfitting
# Create a network prone to overfitting
class SimpleNetworkWithDropout:
    """Two-layer network with optional dropout"""
    
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.0):
        # Initialize weights with proper scaling
        self.W1 = np.random.randn(hidden_size, input_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(output_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(output_size)
        
        # Dropout layer
        self.dropout = DropoutLayer(dropout_rate)
        self.dropout_rate = dropout_rate
        
    def forward(self, X, training=True):
        # First layer
        self.z1 = np.dot(X, self.W1.T) + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU
        
        # Apply dropout
        self.a1_dropped = self.dropout.forward(self.a1, training=training)
        
        # Second layer
        self.z2 = np.dot(self.a1_dropped, self.W2.T) + self.b2
        return self.z2

# Create a classification dataset
from sklearn.datasets import make_moons
X_moon, y_moon = make_moons(n_samples=200, noise=0.3, random_state=42)
# Split into train/test
split = 150
X_train_moon = X_moon[:split]
y_train_moon = y_moon[:split]
X_test_moon = X_moon[split:]
y_test_moon = y_moon[split:]

print(f"\nMoons dataset: {split} training, {len(X_test_moon)} test samples")

# Train networks with and without dropout
networks = {
    'No Dropout': SimpleNetworkWithDropout(2, 100, 1, dropout_rate=0.0),
    'Dropout 0.5': SimpleNetworkWithDropout(2, 100, 1, dropout_rate=0.5)
}

# Simple training (for demonstration)
from scipy.special import expit  # sigmoid function

for name, net in networks.items():
    print(f"\nTraining {name}...")
    
    for epoch in range(500):
        # Forward pass
        logits = net.forward(X_train_moon, training=True)
        probs = expit(logits.flatten())
        
        # Binary cross-entropy gradient
        grad = probs - y_train_moon
        
        # Backward pass (simplified)
        # Output layer
        dW2 = np.outer(grad, net.a1_dropped.mean(axis=0))
        db2 = np.mean(grad)
        
        # Update
        net.W2 -= 0.1 * dW2
        net.b2 -= 0.1 * db2

# Visualize decision boundaries
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for idx, (name, net) in enumerate(networks.items()):
    ax = axes[idx]
    
    # Create mesh
    h = 0.02
    x_min, x_max = X_moon[:, 0].min() - 0.5, X_moon[:, 0].max() + 0.5
    y_min, y_max = X_moon[:, 1].min() - 0.5, X_moon[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict on mesh
    Z = net.forward(np.c_[xx.ravel(), yy.ravel()], training=False)
    Z = expit(Z).reshape(xx.shape)
    
    # Plot
    ax.contourf(xx, yy, Z, levels=[0, 0.5, 1], colors=['lightcoral', 'lightblue'], alpha=0.8)
    ax.scatter(X_train_moon[y_train_moon==0, 0], X_train_moon[y_train_moon==0, 1], 
              c='red', s=50, edgecolor='k', label='Class 0 (train)')
    ax.scatter(X_train_moon[y_train_moon==1, 0], X_train_moon[y_train_moon==1, 1], 
              c='blue', s=50, edgecolor='k', label='Class 1 (train)')
    ax.scatter(X_test_moon[:, 0], X_test_moon[:, 1], c='green', marker='^', 
              s=50, edgecolor='k', label='Test data')
    
    ax.set_title(f'{name}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle('Dropout Effect: Smoother Decision Boundaries', fontsize=14)
plt.tight_layout()
plt.show()

print("\nDropout benefits:")
print("  - Prevents complex, overfitted boundaries")
print("  - Forces network to be robust")
print("  - Acts like training multiple networks")

# -----------------------------------------------------------------------------
# PART 5: EARLY STOPPING
# -----------------------------------------------------------------------------
print("\n\n5. EARLY STOPPING")
print("-" * 40)

# Simulate training with validation monitoring
np.random.seed(42)
epochs = 100

# Simulated loss curves (typical overfitting pattern)
train_losses = []
val_losses = []

for epoch in range(epochs):
    # Training loss decreases continuously
    train_loss = 2.0 * np.exp(-0.05 * epoch) + 0.1 + 0.01 * np.random.randn()
    train_losses.append(train_loss)
    
    # Validation loss decreases then increases
    if epoch < 30:
        val_loss = 2.0 * np.exp(-0.04 * epoch) + 0.15 + 0.02 * np.random.randn()
    else:
        # Start overfitting
        val_loss = val_losses[-1] + 0.002 * (epoch - 30) + 0.02 * np.random.randn()
    val_losses.append(val_loss)

# Find best epoch (minimum validation loss)
best_epoch = np.argmin(val_losses)
best_val_loss = val_losses[best_epoch]

# Implement early stopping with patience
def early_stopping(val_losses, patience=10):
    """Find when to stop training"""
    best_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    for epoch, val_loss in enumerate(val_losses):
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            return best_epoch, epoch  # best_epoch, stop_epoch
    
    return best_epoch, len(val_losses) - 1

best_epoch_patience, stop_epoch = early_stopping(val_losses, patience=10)

# Visualize
plt.figure(figsize=(10, 6))
plt.plot(train_losses, 'b-', linewidth=2, label='Training loss')
plt.plot(val_losses, 'r-', linewidth=2, label='Validation loss')
plt.axvline(x=best_epoch, color='g', linestyle='--', linewidth=2, 
           label=f'Best epoch ({best_epoch})')
plt.axvline(x=stop_epoch, color='orange', linestyle='--', linewidth=2, 
           label=f'Early stop ({stop_epoch})')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Early Stopping: Stop When Validation Loss Stops Improving')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
print(f"Early stopping at epoch {stop_epoch} (patience=10)")
print(f"Avoided {epochs - stop_epoch} epochs of overfitting!")

# -----------------------------------------------------------------------------
# PART 6: DATA AUGMENTATION
# -----------------------------------------------------------------------------
print("\n\n6. DATA AUGMENTATION")
print("-" * 40)

# Simple image augmentation demonstration
def create_sample_image():
    """Create a simple 10x10 'image' with a pattern"""
    img = np.zeros((10, 10))
    # Add a simple L shape
    img[2:8, 2] = 1
    img[7, 2:6] = 1
    return img

def augment_image(img, augmentation_type):
    """Apply different augmentations"""
    if augmentation_type == 'original':
        return img
    elif augmentation_type == 'flip_horizontal':
        return np.fliplr(img)
    elif augmentation_type == 'flip_vertical':
        return np.flipud(img)
    elif augmentation_type == 'rotate_90':
        return np.rot90(img)
    elif augmentation_type == 'add_noise':
        noise = np.random.randn(*img.shape) * 0.1
        return np.clip(img + noise, 0, 1)
    elif augmentation_type == 'shift':
        shifted = np.zeros_like(img)
        shifted[1:, 1:] = img[:-1, :-1]
        return shifted

# Create and augment
original_img = create_sample_image()
augmentations = ['original', 'flip_horizontal', 'flip_vertical', 
                'rotate_90', 'add_noise', 'shift']

plt.figure(figsize=(15, 3))
for idx, aug_type in enumerate(augmentations):
    aug_img = augment_image(original_img, aug_type)
    
    plt.subplot(1, 6, idx+1)
    plt.imshow(aug_img, cmap='gray', vmin=0, vmax=1)
    plt.title(aug_type.replace('_', ' ').title())
    plt.axis('off')

plt.suptitle('Data Augmentation: Creating Variations of Training Data', fontsize=14)
plt.tight_layout()
plt.show()

print("Data augmentation benefits:")
print("  - Increases effective dataset size")
print("  - Forces learning of invariant features")
print("  - Reduces overfitting to specific examples")
print("  - Free performance boost!")

# -----------------------------------------------------------------------------
# PART 7: COMPARING ALL TECHNIQUES
# -----------------------------------------------------------------------------
print("\n\n7. COMPARING REGULARIZATION TECHNIQUES")
print("-" * 40)

# Summary visualization
techniques = ['None', 'L2 Reg', 'L1 Reg', 'Dropout', 'Early Stop', 'Data Aug']
effectiveness = [2, 7, 6, 8, 7, 9]  # Subjective effectiveness scores
ease_of_use = [10, 9, 8, 7, 9, 6]   # Ease of implementation

plt.figure(figsize=(10, 6))
x = np.arange(len(techniques))
width = 0.35

plt.bar(x - width/2, effectiveness, width, label='Effectiveness', alpha=0.8)
plt.bar(x + width/2, ease_of_use, width, label='Ease of Use', alpha=0.8)

plt.xlabel('Technique')
plt.ylabel('Score (1-10)')
plt.title('Regularization Techniques: Effectiveness vs Ease of Use')
plt.xticks(x, techniques)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

print("Best practices:")
print("  1. Start with L2 regularization (simple and effective)")
print("  2. Add dropout for deep networks")
print("  3. Always use early stopping (free and easy)")
print("  4. Data augmentation for image/audio tasks")
print("  5. Combine multiple techniques for best results")

# -----------------------------------------------------------------------------
# EXERCISES
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print("EXERCISES - Master Regularization!")
print("="*60)

print("\n📝 Exercise 1: Implement Elastic Net")
print("Combine L1 and L2 regularization")

class ElasticNet:
    """
    Implement Elastic Net regularization.
    Loss = MSE + α * (ρ * L1 + (1-ρ) * L2)
    where α controls overall regularization strength
    and ρ balances between L1 and L2
    """
    def __init__(self, n_features, alpha=0.01, rho=0.5):
        self.weights = np.random.randn(n_features) * 0.01
        self.alpha = alpha
        self.rho = rho
        
    def regularization_loss(self):
        """
        Calculate the elastic net penalty.
        Should return: α * (ρ * Σ|w| + (1-ρ) * Σw²)
        """
        # YOUR CODE HERE
        l1_penalty = None  # Calculate L1 penalty
        l2_penalty = None  # Calculate L2 penalty
        
        return None  # Return combined penalty
    
    def gradient_penalty(self):
        """
        Calculate gradient of elastic net penalty.
        Should return: α * (ρ * sign(w) + (1-ρ) * 2w)
        """
        # YOUR CODE HERE
        return None

# Test your implementation
try:
    elastic = ElasticNet(n_features=5, alpha=0.1, rho=0.7)
    elastic.weights = np.array([0.5, -0.3, 0.0, 0.8, -0.1])
    
    reg_loss = elastic.regularization_loss()
    if reg_loss is None:
        print("⚠️  Not implemented yet")
    else:
        print(f"Weights: {elastic.weights}")
        print(f"Regularization loss: {reg_loss:.4f}")
        # Expected ≈ 0.1 * (0.7 * 1.7 + 0.3 * 0.79) = 0.143
except Exception as e:
    print(f"⚠️  Error: {e}")

print("\n📝 Exercise 2: Implement Dropout Forward and Backward")
print("Complete the dropout implementation")

class MyDropout:
    """
    Implement dropout with proper forward and backward passes.
    Remember to scale appropriately!
    """
    def __init__(self, p=0.5):
        self.p = p  # Dropout probability
        self.mask = None
        
    def forward(self, x, training=True):
        """
        Forward pass with dropout.
        During training: randomly drop neurons and scale
        During testing: return input unchanged
        """
        if not training:
            return x
            
        # YOUR CODE HERE
        # 1. Create random mask (Hint: use np.random.rand)
        # 2. Apply mask and scale by 1/(1-p)
        
        return x  # Replace with your implementation
    
    def backward(self, grad_output):
        """
        Backward pass: gradients only flow through kept neurons
        """
        # YOUR CODE HERE
        # Apply the same mask and scaling as forward pass
        
        return grad_output  # Replace with your implementation

# Test dropout
try:
    my_dropout = MyDropout(p=0.3)
    x = np.ones((2, 4))  # 2 samples, 4 features
    
    # Forward pass
    out = my_dropout.forward(x, training=True)
    print(f"Input shape: {x.shape}")
    print(f"Output (training): \n{out}")
    print(f"Mean output (should be ≈1): {np.mean(out):.2f}")
    
    # Test mode
    out_test = my_dropout.forward(x, training=False)
    print(f"\nOutput (testing): \n{out_test}")
except:
    print("⚠️  Error in implementation")

print("\n📝 Exercise 3: Implement Early Stopping Monitor")
print("Create a class to monitor training and stop appropriately")

class EarlyStoppingMonitor:
    """
    Monitor validation loss and determine when to stop training.
    """
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.counter = 0
        self.should_stop = False
        
    def update(self, val_loss, epoch):
        """
        Update with new validation loss.
        Return True if training should stop.
        """
        # YOUR CODE HERE
        # 1. Check if val_loss is better than best_loss (by at least min_delta)
        # 2. If yes: update best_loss, reset counter
        # 3. If no: increment counter
        # 4. If counter >= patience: set should_stop = True
        
        return self.should_stop

# Test early stopping
try:
    monitor = EarlyStoppingMonitor(patience=3, min_delta=0.01)
    
    # Simulate validation losses
    val_losses = [1.0, 0.8, 0.7, 0.69, 0.68, 0.68, 0.67]
    
    for epoch, loss in enumerate(val_losses):
        should_stop = monitor.update(loss, epoch)
        print(f"Epoch {epoch}: val_loss={loss:.2f}, should_stop={should_stop}")
        if should_stop:
            print(f"Early stopping triggered at epoch {epoch}!")
            break
except:
    print("⚠️  Error in implementation")

print("\n🎉 Congratulations! You've mastered regularization!")
print("\nKey achievements:")
print("  ✓ Understood overfitting and how to detect it")
print("  ✓ Implemented L1 and L2 regularization")
print("  ✓ Built dropout from scratch")
print("  ✓ Created early stopping monitors")
print("  ✓ Learned data augmentation techniques")
print("\nNext: Put it all together in a complete MNIST project!")