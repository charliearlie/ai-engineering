"""
Lesson 8: Optimization and Learning
Training neural networks effectively with modern optimizers
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

print("="*60)
print("LESSON 8: OPTIMIZATION AND LEARNING")
print("="*60)

# -----------------------------------------------------------------------------
# PART 1: VISUALIZING THE OPTIMIZATION LANDSCAPE
# -----------------------------------------------------------------------------
print("\n1. THE OPTIMIZATION LANDSCAPE")
print("-" * 40)

# Create a simple 2D loss landscape for visualization
def create_loss_landscape(x_range, y_range):
    """Create a non-convex loss landscape with interesting features"""
    X, Y = np.meshgrid(x_range, y_range)
    
    # Create a landscape with multiple local minima and saddle points
    Z = (0.5 * (X - 1)**2 + 0.5 * (Y - 1)**2 +  # Global minimum at (1, 1)
         0.3 * np.sin(3 * X) * np.sin(3 * Y) +   # Add some local minima
         0.1 * X * Y)                             # Add some coupling
    
    return X, Y, Z

# Create the landscape
x_range = np.linspace(-2, 4, 100)
y_range = np.linspace(-2, 4, 100)
X, Y, Z = create_loss_landscape(x_range, y_range)

# Visualize the landscape
fig = plt.figure(figsize=(15, 5))

# 3D view
ax1 = fig.add_subplot(131, projection='3d')
surf = ax1.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.8)
ax1.set_xlabel('Weight 1')
ax1.set_ylabel('Weight 2')
ax1.set_zlabel('Loss')
ax1.set_title('3D Loss Landscape')

# Contour plot
ax2 = fig.add_subplot(132)
contour = ax2.contour(X, Y, Z, levels=20)
ax2.clabel(contour, inline=True, fontsize=8)
ax2.set_xlabel('Weight 1')
ax2.set_ylabel('Weight 2')
ax2.set_title('Contour View')
ax2.grid(True, alpha=0.3)

# Heatmap
ax3 = fig.add_subplot(133)
im = ax3.imshow(Z, extent=[-2, 4, -2, 4], origin='lower', cmap='viridis', aspect='auto')
plt.colorbar(im, ax=ax3, label='Loss')
ax3.set_xlabel('Weight 1')
ax3.set_ylabel('Weight 2')
ax3.set_title('Heatmap View')

plt.tight_layout()
plt.show()

print("Notice the complex landscape:")
print("  - Global minimum around (1, 1)")
print("  - Multiple local minima (dark blue spots)")
print("  - Saddle points (flat regions)")
print("  - Ravines (narrow valleys)")

# -----------------------------------------------------------------------------
# PART 2: COMPARING BASIC GRADIENT DESCENT VS MOMENTUM
# -----------------------------------------------------------------------------
print("\n\n2. GRADIENT DESCENT VS MOMENTUM")
print("-" * 40)

class Optimizer:
    """Base class for optimizers"""
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    def update(self, params, gradients):
        """Update parameters given gradients"""
        raise NotImplementedError

class GradientDescent(Optimizer):
    """Basic gradient descent"""
    def update(self, params, gradients):
        # Simple update: params = params - learning_rate * gradients
        return params - self.learning_rate * gradients

class Momentum(Optimizer):
    """Gradient descent with momentum"""
    def __init__(self, learning_rate=0.01, momentum=0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = None
    
    def update(self, params, gradients):
        # Initialize velocity if needed
        if self.velocity is None:
            self.velocity = np.zeros_like(params)
        
        # Update velocity: v = momentum * v - learning_rate * gradients
        self.velocity = self.momentum * self.velocity - self.learning_rate * gradients
        
        # Update params: params = params + v
        return params + self.velocity

# Function to compute gradients of our loss landscape
def compute_gradient(x, y):
    """Compute gradient at a point (x, y)"""
    # Analytical gradient of our loss function
    dx = (x - 1) + 0.9 * np.cos(3 * x) * np.sin(3 * y) + 0.1 * y
    dy = (y - 1) + 0.9 * np.sin(3 * x) * np.cos(3 * y) + 0.1 * x
    return np.array([dx, dy])

# Function to optimize and track path
def optimize_path(optimizer, start_point, num_steps=50):
    """Run optimizer and return path taken"""
    path = [start_point.copy()]
    point = start_point.copy()
    
    for _ in range(num_steps):
        # Compute gradient at current point
        grad = compute_gradient(point[0], point[1])
        
        # Update point
        point = optimizer.update(point, grad)
        path.append(point.copy())
    
    return np.array(path)

# Compare optimizers from same starting point
start = np.array([3.5, 3.5])
print(f"Starting point: {start}")

# Run both optimizers
gd_optimizer = GradientDescent(learning_rate=0.1)
momentum_optimizer = Momentum(learning_rate=0.1, momentum=0.9)

gd_path = optimize_path(gd_optimizer, start, num_steps=50)
momentum_path = optimize_path(momentum_optimizer, start, num_steps=50)

# Visualize paths
plt.figure(figsize=(12, 5))

# Gradient Descent path
plt.subplot(121)
plt.contour(X, Y, Z, levels=20, alpha=0.6)
plt.plot(gd_path[:, 0], gd_path[:, 1], 'r.-', linewidth=2, markersize=8, label='Path')
plt.plot(gd_path[0, 0], gd_path[0, 1], 'go', markersize=10, label='Start')
plt.plot(gd_path[-1, 0], gd_path[-1, 1], 'ro', markersize=10, label='End')
plt.xlabel('Weight 1')
plt.ylabel('Weight 2')
plt.title('Gradient Descent Path')
plt.legend()
plt.grid(True, alpha=0.3)

# Momentum path
plt.subplot(122)
plt.contour(X, Y, Z, levels=20, alpha=0.6)
plt.plot(momentum_path[:, 0], momentum_path[:, 1], 'b.-', linewidth=2, markersize=8, label='Path')
plt.plot(momentum_path[0, 0], momentum_path[0, 1], 'go', markersize=10, label='Start')
plt.plot(momentum_path[-1, 0], momentum_path[-1, 1], 'bo', markersize=10, label='End')
plt.xlabel('Weight 1')
plt.ylabel('Weight 2')
plt.title('Momentum Path')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nNotice the differences:")
print("  - Gradient descent: More zigzagging, follows gradient exactly")
print("  - Momentum: Smoother path, overshoots but corrects")
print("  - Momentum often reaches better solutions faster!")

# -----------------------------------------------------------------------------
# PART 3: LEARNING RATE SCHEDULES
# -----------------------------------------------------------------------------
print("\n\n3. LEARNING RATE SCHEDULES")
print("-" * 40)

# Define different learning rate schedules
def constant_lr(initial_lr, epoch):
    """Constant learning rate"""
    return initial_lr

def step_decay_lr(initial_lr, epoch, drop_rate=0.5, epochs_per_drop=10):
    """Step decay: drop by drop_rate every epochs_per_drop epochs"""
    drops = epoch // epochs_per_drop
    return initial_lr * (drop_rate ** drops)

def exponential_decay_lr(initial_lr, epoch, decay_rate=0.95):
    """Exponential decay: lr = initial_lr * decay_rate^epoch"""
    return initial_lr * (decay_rate ** epoch)

def cosine_annealing_lr(initial_lr, epoch, total_epochs=100):
    """Cosine annealing: follows a cosine curve"""
    return initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))

# Visualize schedules
epochs = np.arange(100)
initial_lr = 0.1

plt.figure(figsize=(12, 8))

schedules = [
    ("Constant", [constant_lr(initial_lr, e) for e in epochs]),
    ("Step Decay", [step_decay_lr(initial_lr, e) for e in epochs]),
    ("Exponential Decay", [exponential_decay_lr(initial_lr, e) for e in epochs]),
    ("Cosine Annealing", [cosine_annealing_lr(initial_lr, e) for e in epochs])
]

for i, (name, lr_values) in enumerate(schedules):
    plt.subplot(2, 2, i+1)
    plt.plot(epochs, lr_values, 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title(f'{name} Schedule')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, initial_lr * 1.1)
    
    # Add annotations
    if name == "Step Decay":
        plt.axvline(x=10, color='r', linestyle='--', alpha=0.5)
        plt.axvline(x=20, color='r', linestyle='--', alpha=0.5)
        plt.axvline(x=30, color='r', linestyle='--', alpha=0.5)
        plt.text(5, 0.08, 'Drop points', color='r')

plt.tight_layout()
plt.show()

print("Learning rate schedules help by:")
print("  - Starting with large steps for exploration")
print("  - Gradually reducing for fine-tuning")
print("  - Preventing overshooting near convergence")

# -----------------------------------------------------------------------------
# PART 4: ADAPTIVE LEARNING RATES - ADAGRAD AND RMSPROP
# -----------------------------------------------------------------------------
print("\n\n4. ADAPTIVE LEARNING RATES")
print("-" * 40)

class AdaGrad(Optimizer):
    """Adaptive Gradient Algorithm"""
    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        super().__init__(learning_rate)
        self.epsilon = epsilon
        self.accumulated_gradients = None
    
    def update(self, params, gradients):
        # Initialize accumulator if needed
        if self.accumulated_gradients is None:
            self.accumulated_gradients = np.zeros_like(params)
        
        # Accumulate squared gradients
        self.accumulated_gradients += gradients ** 2
        
        # Update with adaptive learning rate
        adapted_lr = self.learning_rate / (np.sqrt(self.accumulated_gradients) + self.epsilon)
        return params - adapted_lr * gradients

class RMSprop(Optimizer):
    """Root Mean Square Propagation"""
    def __init__(self, learning_rate=0.01, decay_rate=0.9, epsilon=1e-8):
        super().__init__(learning_rate)
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.squared_gradients = None
    
    def update(self, params, gradients):
        # Initialize if needed
        if self.squared_gradients is None:
            self.squared_gradients = np.zeros_like(params)
        
        # Update moving average of squared gradients
        self.squared_gradients = (self.decay_rate * self.squared_gradients + 
                                 (1 - self.decay_rate) * gradients ** 2)
        
        # Update with adaptive learning rate
        adapted_lr = self.learning_rate / (np.sqrt(self.squared_gradients) + self.epsilon)
        return params - adapted_lr * gradients

# Compare adaptive methods on a problem with different gradient scales
print("Demonstrating adaptive learning rates...")

# Create a problem where parameters have very different gradient magnitudes
def scaled_gradient(params):
    """Gradient where first parameter has much larger gradients"""
    return np.array([10 * (params[0] - 1),  # Large gradients
                     0.1 * (params[1] - 1)])  # Small gradients

# Track parameter values
start_point = np.array([0.0, 0.0])
num_steps = 100

# Regular gradient descent
gd = GradientDescent(learning_rate=0.01)
gd_params = [start_point.copy()]
param = start_point.copy()
for _ in range(num_steps):
    grad = scaled_gradient(param)
    param = gd.update(param, grad)
    gd_params.append(param.copy())
gd_params = np.array(gd_params)

# AdaGrad
adagrad = AdaGrad(learning_rate=0.1)
adagrad_params = [start_point.copy()]
param = start_point.copy()
for _ in range(num_steps):
    grad = scaled_gradient(param)
    param = adagrad.update(param, grad)
    adagrad_params.append(param.copy())
adagrad_params = np.array(adagrad_params)

# Plot convergence
plt.figure(figsize=(12, 5))

plt.subplot(121)
plt.plot(gd_params[:, 0], label='Parameter 1 (large gradients)', linewidth=2)
plt.plot(gd_params[:, 1], label='Parameter 2 (small gradients)', linewidth=2)
plt.axhline(y=1, color='k', linestyle='--', alpha=0.5)
plt.xlabel('Steps')
plt.ylabel('Parameter Value')
plt.title('Gradient Descent: Same learning rate for all')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(122)
plt.plot(adagrad_params[:, 0], label='Parameter 1 (large gradients)', linewidth=2)
plt.plot(adagrad_params[:, 1], label='Parameter 2 (small gradients)', linewidth=2)
plt.axhline(y=1, color='k', linestyle='--', alpha=0.5)
plt.xlabel('Steps')
plt.ylabel('Parameter Value')
plt.title('AdaGrad: Adaptive learning rates')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nNotice how AdaGrad:")
print("  - Gives smaller learning rates to parameters with large gradients")
print("  - Gives larger learning rates to parameters with small gradients")
print("  - Both parameters converge at similar speeds!")

# -----------------------------------------------------------------------------
# PART 5: ADAM OPTIMIZER
# -----------------------------------------------------------------------------
print("\n\n5. ADAM - THE COMPLETE PACKAGE")
print("-" * 40)

class Adam(Optimizer):
    """Adaptive Moment Estimation"""
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1  # Momentum decay
        self.beta2 = beta2  # RMSprop decay
        self.epsilon = epsilon
        self.m = None  # First moment (momentum)
        self.v = None  # Second moment (RMSprop)
        self.t = 0     # Time step
    
    def update(self, params, gradients):
        # Initialize moments if needed
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        
        # Increment time step
        self.t += 1
        
        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        
        # Update biased second raw moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradients**2
        
        # Compute bias-corrected first moment estimate
        m_hat = self.m / (1 - self.beta1**self.t)
        
        # Compute bias-corrected second raw moment estimate
        v_hat = self.v / (1 - self.beta2**self.t)
        
        # Update parameters
        return params - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

# Compare all optimizers on the same problem
print("Comparing all optimizers on our loss landscape...")

optimizers = {
    'Gradient Descent': GradientDescent(learning_rate=0.1),
    'Momentum': Momentum(learning_rate=0.1, momentum=0.9),
    'AdaGrad': AdaGrad(learning_rate=0.5),
    'RMSprop': RMSprop(learning_rate=0.1),
    'Adam': Adam(learning_rate=0.1)
}

# Run all optimizers from same starting point
start = np.array([3.5, 3.5])
paths = {}
losses = {}

for name, optimizer in optimizers.items():
    path = [start.copy()]
    point = start.copy()
    loss_history = []
    
    for _ in range(100):
        # Compute gradient
        grad = compute_gradient(point[0], point[1])
        
        # Update
        point = optimizer.update(point, grad)
        path.append(point.copy())
        
        # Compute loss at new point
        loss = (0.5 * (point[0] - 1)**2 + 0.5 * (point[1] - 1)**2 + 
                0.3 * np.sin(3 * point[0]) * np.sin(3 * point[1]) + 
                0.1 * point[0] * point[1])
        loss_history.append(loss)
    
    paths[name] = np.array(path)
    losses[name] = loss_history

# Visualize all paths
plt.figure(figsize=(15, 10))

# Plot paths on landscape
plt.subplot(2, 3, 1)
plt.contour(X, Y, Z, levels=20, alpha=0.6)
colors = ['red', 'blue', 'green', 'orange', 'purple']
for (name, path), color in zip(paths.items(), colors):
    plt.plot(path[:, 0], path[:, 1], '-', color=color, linewidth=2, 
             alpha=0.7, label=name)
plt.plot(start[0], start[1], 'ko', markersize=10, label='Start')
plt.xlabel('Weight 1')
plt.ylabel('Weight 2')
plt.title('All Optimizer Paths')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

# Plot individual paths
for i, (name, path) in enumerate(paths.items()):
    plt.subplot(2, 3, i+2)
    plt.contour(X, Y, Z, levels=20, alpha=0.6)
    plt.plot(path[:, 0], path[:, 1], '.-', linewidth=2, markersize=4)
    plt.plot(path[0, 0], path[0, 1], 'go', markersize=8)
    plt.plot(path[-1, 0], path[-1, 1], 'ro', markersize=8)
    plt.xlabel('Weight 1')
    plt.ylabel('Weight 2')
    plt.title(name)
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Plot loss curves
plt.figure(figsize=(10, 6))
for name, loss_history in losses.items():
    plt.plot(loss_history, linewidth=2, label=name)
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Loss Curves for Different Optimizers')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.show()

print("\nOptimizer characteristics:")
print("  - GD: Simple but slow, lots of zigzagging")
print("  - Momentum: Faster, smoother path")
print("  - AdaGrad: Adaptive but can stop learning")
print("  - RMSprop: Adaptive without stopping")
print("  - Adam: Best of all - fast and adaptive!")

# -----------------------------------------------------------------------------
# PART 6: BATCH SIZE EFFECTS
# -----------------------------------------------------------------------------
print("\n\n6. BATCH SIZE EFFECTS")
print("-" * 40)

# Simulate training with different batch sizes
np.random.seed(42)

# Generate synthetic data
n_samples = 1000
X_data = np.random.randn(n_samples, 10)
true_weights = np.random.randn(10)
y_data = np.dot(X_data, true_weights) + 0.1 * np.random.randn(n_samples)

# Function to train with different batch sizes
def train_with_batch_size(batch_size, num_epochs=20):
    """Train a simple linear model with specified batch size"""
    weights = np.random.randn(10) * 0.1
    losses = []
    gradient_norms = []
    
    optimizer = Adam(learning_rate=0.01)
    
    for epoch in range(num_epochs):
        epoch_losses = []
        epoch_grad_norms = []
        
        # Shuffle data
        indices = np.random.permutation(n_samples)
        
        # Process mini-batches
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            X_batch = X_data[batch_indices]
            y_batch = y_data[batch_indices]
            
            # Forward pass
            predictions = np.dot(X_batch, weights)
            loss = np.mean((predictions - y_batch)**2)
            epoch_losses.append(loss)
            
            # Compute gradients
            gradients = 2 * np.dot(X_batch.T, predictions - y_batch) / len(batch_indices)
            epoch_grad_norms.append(np.linalg.norm(gradients))
            
            # Update weights
            weights = optimizer.update(weights, gradients)
        
        losses.append(np.mean(epoch_losses))
        gradient_norms.append(np.mean(epoch_grad_norms))
    
    return losses, gradient_norms

# Train with different batch sizes
batch_sizes = [1, 16, 64, 256]
results = {}

print("Training with different batch sizes...")
for batch_size in batch_sizes:
    print(f"  Batch size: {batch_size}")
    losses, grad_norms = train_with_batch_size(batch_size)
    results[batch_size] = (losses, grad_norms)

# Visualize results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Loss curves
for batch_size, (losses, _) in results.items():
    ax1.plot(losses, linewidth=2, label=f'Batch size: {batch_size}')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Loss vs Batch Size')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# Gradient norm variance
for batch_size, (_, grad_norms) in results.items():
    ax2.plot(grad_norms, linewidth=2, label=f'Batch size: {batch_size}')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Gradient Norm')
ax2.set_title('Gradient Stability vs Batch Size')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nBatch size effects:")
print("  - Small batch (1): Noisy gradients, can escape local minima")
print("  - Medium batch (16-64): Good balance of noise and stability")
print("  - Large batch (256+): Smooth gradients, faster per epoch")
print("  - Trade-off: Noise helps exploration but slows convergence")

# -----------------------------------------------------------------------------
# EXERCISES
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print("EXERCISES - Master Optimization!")
print("="*60)

print("\n📝 Exercise 1: Implement Learning Rate Decay")
print("Create an optimizer with learning rate decay")

class SGDWithDecay(Optimizer):
    """
    SGD with learning rate decay.
    Implement exponential decay: lr = initial_lr * decay_rate^epoch
    """
    def __init__(self, initial_lr=0.01, decay_rate=0.95):
        super().__init__(initial_lr)
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
        self.epoch = 0
        
    def update(self, params, gradients):
        """
        Update parameters with decayed learning rate.
        Call step_epoch() after each epoch to decay the rate.
        """
        # YOUR CODE HERE
        # Use the current learning rate (which should decay over time)
        return params  # Replace with your implementation
    
    def step_epoch(self):
        """Call this after each epoch to decay learning rate"""
        # YOUR CODE HERE
        # Update self.learning_rate based on decay formula
        pass

# Test your implementation
try:
    sgd_decay = SGDWithDecay(initial_lr=0.1, decay_rate=0.9)
    
    # Simulate 5 epochs
    lr_history = []
    for epoch in range(5):
        lr_history.append(sgd_decay.learning_rate)
        sgd_decay.step_epoch()
    
    if lr_history[0] == 0.01:  # Using default from parent class
        print("⚠️  Not implemented yet")
    else:
        print("Learning rates over epochs:", [f"{lr:.4f}" for lr in lr_history])
        print("Expected: [0.1000, 0.0900, 0.0810, 0.0729, 0.0656]")
except Exception as e:
    print(f"⚠️  Error: {e}")

print("\n📝 Exercise 2: Implement Nesterov Momentum")
print("Implement the 'look-ahead' version of momentum")

class NesterovMomentum(Optimizer):
    """
    Nesterov Accelerated Gradient (NAG).
    Look ahead by momentum before computing gradient.
    """
    def __init__(self, learning_rate=0.01, momentum=0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = None
        
    def update(self, params, gradients_fn):
        """
        Nesterov momentum requires gradient at lookahead position.
        gradients_fn: function that computes gradients given parameters
        """
        if self.velocity is None:
            self.velocity = np.zeros_like(params)
        
        # YOUR CODE HERE
        # 1. Lookahead: temp_params = params + momentum * velocity
        # 2. Compute gradient at lookahead position
        # 3. Update velocity with this gradient
        # 4. Update params with velocity
        
        return params  # Replace with your implementation

# Test hint
print("\nHint: Nesterov looks at where momentum would take you,")
print("      then computes gradient there instead of current position.")

print("\n📝 Exercise 3: Implement Gradient Clipping")
print("Prevent exploding gradients by clipping")

def clip_gradients(gradients, max_norm=1.0):
    """
    Clip gradients to have maximum L2 norm of max_norm.
    
    If ||gradients|| > max_norm:
        gradients = gradients * max_norm / ||gradients||
    """
    # YOUR CODE HERE
    # 1. Compute L2 norm of gradients
    # 2. If norm > max_norm, scale gradients down
    
    return gradients  # Replace with your implementation

# Test gradient clipping
test_grad = np.array([3.0, 4.0])  # Norm = 5
try:
    clipped = clip_gradients(test_grad, max_norm=1.0)
    if np.array_equal(clipped, test_grad):
        print("⚠️  Not implemented yet")
    else:
        print(f"Original gradient: {test_grad}, norm: {np.linalg.norm(test_grad):.2f}")
        print(f"Clipped gradient: {clipped}, norm: {np.linalg.norm(clipped):.2f}")
        print(f"Expected norm after clipping: 1.0")
except:
    print("⚠️  Error in implementation")

print("\n📝 Exercise 4: Implement a Custom Optimizer")
print("Combine ideas to create your own optimizer!")

class MyOptimizer(Optimizer):
    """
    Create your own optimizer!
    Ideas to try:
    - Combine momentum with adaptive learning rates
    - Add gradient clipping
    - Use different decay rates for different parameters
    - Add warm-up period for learning rate
    """
    def __init__(self, learning_rate=0.01):
        super().__init__(learning_rate)
        # YOUR CODE HERE
        # Add any state variables you need
        
    def update(self, params, gradients):
        """Implement your custom update rule"""
        # YOUR CODE HERE
        # Be creative! Try combining different ideas
        
        return params - self.learning_rate * gradients  # Basic version

# Test your optimizer
print("\nCreate an optimizer that combines your favorite features!")
print("Ideas: momentum + adaptive rates, warm-up period, per-parameter rates, etc.")

print("\n🎉 Congratulations! You've mastered optimization!")
print("\nKey achievements:")
print("  ✓ Understood optimization landscapes")
print("  ✓ Compared different optimizers")
print("  ✓ Implemented learning rate schedules")
print("  ✓ Built adaptive optimizers")
print("  ✓ Analyzed batch size effects")
print("\nNext: Regularization techniques to prevent overfitting!")