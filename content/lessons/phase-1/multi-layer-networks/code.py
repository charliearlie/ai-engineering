"""
Lesson 6: Multi-layer Networks
Building networks that can solve complex problems!
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

print("="*60)
print("LESSON 6: MULTI-LAYER NETWORKS")
print("="*60)

# -----------------------------------------------------------------------------
# PART 1: A TWO-LAYER NETWORK CLASS
# -----------------------------------------------------------------------------
print("\n1. BUILDING A TWO-LAYER NETWORK")
print("-" * 40)

class TwoLayerNetwork:
    """A neural network with one hidden layer"""
    
    def __init__(self, input_size, hidden_size, output_size, activation='relu'):
        """Initialize network with random weights"""
        # Layer 1: input → hidden
        self.W1 = np.random.randn(hidden_size, input_size) * 0.1
        self.b1 = np.zeros(hidden_size)
        
        # Layer 2: hidden → output  
        self.W2 = np.random.randn(output_size, hidden_size) * 0.1
        self.b2 = np.zeros(output_size)
        
        # Activation function
        if activation == 'relu':
            self.activation = lambda x: np.maximum(0, x)
            self.activation_grad = lambda x: (x > 0).astype(float)
        elif activation == 'sigmoid':
            self.activation = lambda x: 1 / (1 + np.exp(-x))
            self.activation_grad = lambda x: self.activation(x) * (1 - self.activation(x))
        else:
            raise ValueError(f"Unknown activation: {activation}")
            
        # Store for backprop
        self.cache = {}
        
    def forward(self, X):
        """Forward pass through the network"""
        # Layer 1
        z1 = np.dot(X, self.W1.T) + self.b1  # Note: X is batch × features
        a1 = self.activation(z1)
        
        # Layer 2
        z2 = np.dot(a1, self.W2.T) + self.b2
        
        # For classification, apply sigmoid to output
        output = 1 / (1 + np.exp(-z2))
        
        # Cache for backprop
        self.cache = {
            'X': X, 'z1': z1, 'a1': a1, 'z2': z2, 'output': output
        }
        
        return output
    
    def backward(self, y, learning_rate=0.1):
        """Backward pass - compute gradients and update weights"""
        m = y.shape[0]  # batch size
        
        # Output layer gradients
        dz2 = self.cache['output'] - y  # For logistic loss
        dW2 = np.dot(dz2.T, self.cache['a1']) / m
        db2 = np.mean(dz2, axis=0)
        
        # Hidden layer gradients  
        da1 = np.dot(dz2, self.W2)
        dz1 = da1 * self.activation_grad(self.cache['z1'])
        dW1 = np.dot(dz1.T, self.cache['X']) / m
        db1 = np.mean(dz1, axis=0)
        
        # Update weights
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        
    def train(self, X, y, epochs=1000, learning_rate=0.1, verbose=True):
        """Train the network"""
        losses = []
        
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Compute loss (binary cross-entropy)
            loss = -np.mean(y * np.log(output + 1e-8) + 
                           (1 - y) * np.log(1 - output + 1e-8))
            losses.append(loss)
            
            # Backward pass
            self.backward(y, learning_rate)
            
            # Print progress
            if verbose and epoch % 100 == 0:
                accuracy = np.mean((output > 0.5) == y)
                print(f"  Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {accuracy:.2%}")
                
        return losses

# Test on XOR problem
print("Training 2-layer network on XOR problem:")

X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([[0], [1], [1], [0]])

# Create and train network
network = TwoLayerNetwork(input_size=2, hidden_size=4, output_size=1)
losses = network.train(X_xor, y_xor, epochs=1000, learning_rate=1.0)

# Test final predictions
print("\nFinal predictions:")
predictions = network.forward(X_xor)
for i in range(len(X_xor)):
    print(f"  {X_xor[i]} → {predictions[i,0]:.3f} (target: {y_xor[i,0]})")

# -----------------------------------------------------------------------------
# PART 2: VISUALIZING HIDDEN LAYER REPRESENTATIONS
# -----------------------------------------------------------------------------
print("\n\n2. WHAT HIDDEN LAYERS LEARN")
print("-" * 40)

# Get hidden layer activations
z1 = np.dot(X_xor, network.W1.T) + network.b1
hidden_activations = network.activation(z1)

print("Hidden layer activations for each input:")
print("  Input [0,0]:", hidden_activations[0])
print("  Input [0,1]:", hidden_activations[1])
print("  Input [1,0]:", hidden_activations[2])
print("  Input [1,1]:", hidden_activations[3])

print("\nNotice: Hidden layer transforms XOR into a linearly separable problem!")

# Visualize the transformation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Original XOR problem
ax1.scatter(X_xor[y_xor[:,0]==0, 0], X_xor[y_xor[:,0]==0, 1], 
           c='red', s=200, marker='o', edgecolor='black', linewidth=2, label='Class 0')
ax1.scatter(X_xor[y_xor[:,0]==1, 0], X_xor[y_xor[:,0]==1, 1], 
           c='blue', s=200, marker='s', edgecolor='black', linewidth=2, label='Class 1')
ax1.set_xlim(-0.5, 1.5)
ax1.set_ylim(-0.5, 1.5)
ax1.set_xlabel('Input 1')
ax1.set_ylabel('Input 2')
ax1.set_title('Original XOR Problem (Not Linearly Separable)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Hidden layer representation (show first 2 dimensions)
ax2.scatter(hidden_activations[y_xor[:,0]==0, 0], hidden_activations[y_xor[:,0]==0, 1], 
           c='red', s=200, marker='o', edgecolor='black', linewidth=2, label='Class 0')
ax2.scatter(hidden_activations[y_xor[:,0]==1, 0], hidden_activations[y_xor[:,0]==1, 1], 
           c='blue', s=200, marker='s', edgecolor='black', linewidth=2, label='Class 1')
ax2.set_xlabel('Hidden Neuron 1')
ax2.set_ylabel('Hidden Neuron 2')
ax2.set_title('Hidden Layer Representation (Now Linearly Separable!)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# PART 3: DECISION BOUNDARIES
# -----------------------------------------------------------------------------
print("\n\n3. COMPLEX DECISION BOUNDARIES")
print("-" * 40)

def plot_decision_boundary(network, X, y, title):
    """Plot the decision boundary of a network"""
    h = 0.01
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Get predictions for all points
    Z = network.forward(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], colors=['lightcoral', 'lightblue'], alpha=0.8)
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
    
    # Plot points
    colors = ['red', 'blue']
    markers = ['o', 's']
    for i in range(2):
        mask = y[:, 0] == i
        plt.scatter(X[mask, 0], X[mask, 1], c=colors[i], marker=markers[i],
                   s=200, edgecolor='black', linewidth=2, label=f'Class {i}')
    
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

# Plot XOR decision boundary
plot_decision_boundary(network, X_xor, y_xor, 
                      "XOR Decision Boundary (2-Layer Network)")
plt.show()

# -----------------------------------------------------------------------------
# PART 4: COMPARING ARCHITECTURES
# -----------------------------------------------------------------------------
print("\n\n4. ARCHITECTURE COMPARISON")
print("-" * 40)

# Create a more complex dataset - two circles
np.random.seed(42)
n_samples = 200

# Inner circle (class 0)
r_inner = np.random.uniform(0, 2, n_samples//2)
theta_inner = np.random.uniform(0, 2*np.pi, n_samples//2)
X_inner = np.column_stack([r_inner * np.cos(theta_inner),
                          r_inner * np.sin(theta_inner)])

# Outer circle (class 1)
r_outer = np.random.uniform(3, 5, n_samples//2)
theta_outer = np.random.uniform(0, 2*np.pi, n_samples//2)
X_outer = np.column_stack([r_outer * np.cos(theta_outer),
                          r_outer * np.sin(theta_outer)])

X_circles = np.vstack([X_inner, X_outer])
y_circles = np.vstack([np.zeros((n_samples//2, 1)), 
                      np.ones((n_samples//2, 1))])

# Shuffle
shuffle_idx = np.random.permutation(n_samples)
X_circles = X_circles[shuffle_idx]
y_circles = y_circles[shuffle_idx]

# Train networks with different architectures
architectures = [
    (2, "Very Small (2 hidden)"),
    (8, "Small (8 hidden)"),
    (32, "Medium (32 hidden)"),
    (128, "Large (128 hidden)")
]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for idx, (hidden_size, name) in enumerate(architectures):
    print(f"\nTraining {name}...")
    net = TwoLayerNetwork(input_size=2, hidden_size=hidden_size, output_size=1)
    losses = net.train(X_circles, y_circles, epochs=500, learning_rate=0.5, verbose=False)
    
    # Final accuracy
    predictions = net.forward(X_circles)
    accuracy = np.mean((predictions > 0.5) == y_circles)
    print(f"  Final accuracy: {accuracy:.2%}")
    
    # Plot decision boundary
    ax = axes[idx]
    h = 0.1
    x_min, x_max = X_circles[:, 0].min() - 1, X_circles[:, 0].max() + 1
    y_min, y_max = X_circles[:, 1].min() - 1, X_circles[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = net.forward(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, levels=20, cmap='RdBu', alpha=0.8)
    ax.scatter(X_circles[y_circles[:,0]==0, 0], X_circles[y_circles[:,0]==0, 1],
              c='red', s=50, edgecolor='black', linewidth=1)
    ax.scatter(X_circles[y_circles[:,0]==1, 0], X_circles[y_circles[:,0]==1, 1],
              c='blue', s=50, edgecolor='black', linewidth=1)
    ax.set_title(f'{name}\nAccuracy: {accuracy:.1%}')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

plt.suptitle('Effect of Hidden Layer Size on Decision Boundary', fontsize=16)
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# PART 5: DEEP NETWORKS
# -----------------------------------------------------------------------------
print("\n\n5. BUILDING DEEPER NETWORKS")
print("-" * 40)

class DeepNetwork:
    """A neural network with multiple hidden layers"""
    
    def __init__(self, layer_sizes):
        """
        Initialize network with given architecture.
        layer_sizes: list of layer sizes, e.g., [2, 16, 8, 1]
        """
        self.layers = []
        self.layer_sizes = layer_sizes
        
        # Create weight matrices for each layer
        for i in range(len(layer_sizes) - 1):
            W = np.random.randn(layer_sizes[i+1], layer_sizes[i]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros(layer_sizes[i+1])
            self.layers.append({'W': W, 'b': b})
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, X):
        """Forward pass through all layers"""
        a = X
        activations = [a]
        
        for i, layer in enumerate(self.layers):
            z = np.dot(a, layer['W'].T) + layer['b']
            
            # Use ReLU for hidden layers, sigmoid for output
            if i < len(self.layers) - 1:
                a = self.relu(z)
            else:
                a = self.sigmoid(z)
                
            activations.append(a)
        
        return a, activations
    
    def predict(self, X):
        """Get predictions"""
        output, _ = self.forward(X)
        return output

# Create networks with different depths
print("Comparing network depths on XOR:")

depths = [
    [2, 4, 1],           # 2 layers (what we used before)
    [2, 4, 4, 1],        # 3 layers
    [2, 4, 4, 4, 1],     # 4 layers
    [2, 4, 4, 4, 4, 1]   # 5 layers
]

for depth in depths:
    net = DeepNetwork(depth)
    # Note: We're not training these, just showing the architecture
    print(f"  {len(depth)-1} layers: {' → '.join(map(str, depth))}")

# -----------------------------------------------------------------------------
# PART 6: UNIVERSAL APPROXIMATION DEMO
# -----------------------------------------------------------------------------
print("\n\n6. UNIVERSAL APPROXIMATION")
print("-" * 40)

# Create a complex 1D function to approximate
x = np.linspace(-2, 2, 200).reshape(-1, 1)
y = np.sin(2 * np.pi * x) + 0.1 * np.sin(8 * np.pi * x)

# Try different network widths
widths = [5, 10, 50, 100]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for idx, width in enumerate(widths):
    # Create and train network
    net = TwoLayerNetwork(input_size=1, hidden_size=width, output_size=1, activation='sigmoid')
    
    # Simple training (not optimal, just for demonstration)
    for _ in range(2000):
        output = net.forward(x)
        net.backward(y, learning_rate=0.5)
    
    # Get predictions
    y_pred = net.forward(x)
    
    # Plot
    ax = axes[idx]
    ax.plot(x, y, 'b-', label='True function', linewidth=2)
    ax.plot(x, y_pred, 'r--', label='Network approximation', linewidth=2)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'{width} Hidden Neurons')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle('Universal Approximation: More Neurons = Better Approximation', fontsize=16)
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# EXERCISES
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print("EXERCISES - Design and Build Networks!")
print("="*60)

print("\n📝 Exercise 1: Three-Layer Network")
print("Implement a network with two hidden layers")

class ThreeLayerNetwork:
    """
    Network with two hidden layers.
    Architecture: input → hidden1 → hidden2 → output
    """
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        # YOUR CODE HERE
        # Initialize weights for all three layers
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        self.W3 = None
        self.b3 = None
    
    def forward(self, X):
        """
        Forward pass through all layers.
        Use ReLU for hidden layers, sigmoid for output.
        """
        # YOUR CODE HERE
        return None

# Test your implementation
try:
    net3 = ThreeLayerNetwork(2, 8, 4, 1)
    if net3.W1 is None:
        print("⚠️  Not implemented yet")
    else:
        # Test on XOR
        output = net3.forward(X_xor)
        print("Output shape:", output.shape)
        print("Sample outputs:", output[:2].ravel())
except Exception as e:
    print(f"⚠️  Error: {e}")

print("\n📝 Exercise 2: Flexible Deep Network")
print("Create a network that can have any number of layers")

def create_deep_network(layer_sizes, activation='relu'):
    """
    Create a network with arbitrary depth.
    
    Args:
        layer_sizes: List of layer sizes [input, hidden1, ..., output]
        activation: Activation function name
    
    Returns:
        Dictionary with weights and activation function
    """
    # YOUR CODE HERE
    network = {
        'layers': [],
        'activation': activation
    }
    
    # Initialize weights for each layer
    
    return network

# Test flexible network
try:
    deep_net = create_deep_network([2, 16, 8, 4, 1])
    if not deep_net['layers']:
        print("⚠️  Not implemented yet")
    else:
        print(f"Created network with {len(deep_net['layers'])} weight matrices")
        for i, layer in enumerate(deep_net['layers']):
            print(f"  Layer {i+1}: {layer['W'].shape}")
except:
    print("⚠️  Error in implementation")

print("\n📝 Exercise 3: Activation Function Comparison")
print("Compare different activation functions on the same problem")

def train_with_activation(X, y, activation, hidden_size=16, epochs=1000):
    """
    Train a 2-layer network with specified activation.
    Return final accuracy and decision boundary.
    """
    # YOUR CODE HERE
    # Create network with given activation
    # Train it
    # Return accuracy
    
    accuracy = None
    return accuracy

# Test different activations
activations = ['relu', 'sigmoid']
for act in activations:
    try:
        acc = train_with_activation(X_circles[:100], y_circles[:100], act)
        if acc is None:
            print(f"⚠️  {act}: Not implemented yet")
        else:
            print(f"✅ {act} accuracy: {acc:.2%}")
    except:
        print(f"⚠️  {act}: Error in implementation")

print("\n🎉 Congratulations! You've mastered multi-layer networks!")
print("\nKey achievements:")
print("  ✓ Built networks that solve XOR")
print("  ✓ Visualized hidden layer representations")
print("  ✓ Compared different architectures")
print("  ✓ Understood universal approximation")
print("  ✓ Designed deep networks")
print("\nNext: Backpropagation - how these networks actually learn!")