"""
Lesson 7: Backpropagation Demystified
Understanding and implementing the algorithm that powers deep learning
"""

import numpy as np
import matplotlib.pyplot as plt

print("="*60)
print("LESSON 7: BACKPROPAGATION DEMYSTIFIED")
print("="*60)

# -----------------------------------------------------------------------------
# PART 1: MANUAL BACKPROPAGATION
# -----------------------------------------------------------------------------
print("\n1. MANUAL BACKPROPAGATION - Step by Step")
print("-" * 40)

# Simple 2-layer network example
# Input (2) → Hidden (2) → Output (1)

# Sample input and target
x = np.array([1.0, 0.5])
y_true = 1.0

# Initialize small network
W1 = np.array([[0.5, -0.6],
               [0.3, 0.8]])  # 2x2: 2 hidden neurons, 2 inputs each
b1 = np.array([0.1, -0.1])

W2 = np.array([[0.4, -0.5]])  # 1x2: 1 output, 2 hidden inputs
b2 = np.array([0.2])

print("Network architecture:")
print(f"  Input: {x}")
print(f"  W1:\n{W1}")
print(f"  b1: {b1}")
print(f"  W2: {W2}")
print(f"  b2: {b2}")

# FORWARD PASS - save everything!
print("\n--- FORWARD PASS ---")

# Layer 1
z1 = np.dot(W1, x) + b1
a1 = np.maximum(0, z1)  # ReLU

print(f"Layer 1:")
print(f"  z1 = W1·x + b1 = {z1}")
print(f"  a1 = ReLU(z1) = {a1}")

# Layer 2
z2 = np.dot(W2, a1) + b2
a2 = 1 / (1 + np.exp(-z2))  # Sigmoid

print(f"\nLayer 2:")
print(f"  z2 = W2·a1 + b2 = {z2}")
print(f"  a2 = sigmoid(z2) = {a2}")

# Loss
loss = -y_true * np.log(a2) - (1 - y_true) * np.log(1 - a2)
print(f"\nLoss (binary cross-entropy): {loss[0]:.4f}")

# BACKWARD PASS
print("\n--- BACKWARD PASS ---")

# Output layer gradients
da2 = -(y_true / a2) + (1 - y_true) / (1 - a2)  # d(loss)/d(a2)
dz2 = da2 * a2 * (1 - a2)  # d(loss)/d(z2) = da2 * sigmoid'(z2)

print(f"Output layer:")
print(f"  da2 = d(loss)/d(a2) = {da2[0]:.4f}")
print(f"  dz2 = d(loss)/d(z2) = {dz2[0]:.4f}")

# Gradients for W2 and b2
dW2 = np.outer(dz2, a1)  # dz2 * a1.T
db2 = dz2

print(f"\nWeight gradients (layer 2):")
print(f"  dW2 = dz2 × a1 = {dW2}")
print(f"  db2 = dz2 = {db2}")

# Hidden layer gradients
da1 = np.dot(W2.T, dz2)  # Gradient flows back through W2
dz1 = da1 * (z1 > 0)  # ReLU derivative

print(f"\nHidden layer:")
print(f"  da1 = W2.T × dz2 = {da1}")
print(f"  dz1 = da1 × ReLU'(z1) = {dz1}")

# Gradients for W1 and b1
dW1 = np.outer(dz1, x)
db1 = dz1

print(f"\nWeight gradients (layer 1):")
print(f"  dW1 = dz1 × x =\n{dW1}")
print(f"  db1 = dz1 = {db1}")

# -----------------------------------------------------------------------------
# PART 2: GRADIENT CHECKING
# -----------------------------------------------------------------------------
print("\n\n2. GRADIENT CHECKING - Verifying Backprop")
print("-" * 40)

def numerical_gradient(f, x, h=1e-5):
    """Compute gradient numerically using finite differences"""
    grad = np.zeros_like(x)
    
    # Iterate through each dimension
    for i in range(x.size):
        # Save original value
        old_value = x.flat[i]
        
        # f(x + h)
        x.flat[i] = old_value + h
        fxh_pos = f()
        
        # f(x - h)
        x.flat[i] = old_value - h
        fxh_neg = f()
        
        # Gradient
        grad.flat[i] = (fxh_pos - fxh_neg) / (2 * h)
        
        # Restore original value
        x.flat[i] = old_value
    
    return grad

# Define forward pass as a function
def forward_pass():
    z1 = np.dot(W1, x) + b1
    a1 = np.maximum(0, z1)
    z2 = np.dot(W2, a1) + b2
    a2 = 1 / (1 + np.exp(-z2))
    loss = -y_true * np.log(a2) - (1 - y_true) * np.log(1 - a2)
    return loss[0]

# Check W2 gradient
print("Gradient checking for W2:")
numerical_dW2 = numerical_gradient(forward_pass, W2)
print(f"  Analytical gradient: {dW2}")
print(f"  Numerical gradient:  {numerical_dW2}")
print(f"  Difference: {np.abs(dW2 - numerical_dW2).max():.2e}")

# Check W1 gradient
print("\nGradient checking for W1:")
numerical_dW1 = numerical_gradient(forward_pass, W1)
print(f"  Analytical gradient:\n{dW1}")
print(f"  Numerical gradient:\n{numerical_dW1}")
print(f"  Difference: {np.abs(dW1 - numerical_dW1).max():.2e}")

print("\n✅ Small differences (< 1e-6) mean backprop is correct!")

# -----------------------------------------------------------------------------
# PART 3: BACKPROP WITH BATCHES
# -----------------------------------------------------------------------------
print("\n\n3. BATCH BACKPROPAGATION")
print("-" * 40)

class NeuralNetwork:
    """Neural network with proper backpropagation"""
    
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights with proper scaling
        self.W1 = np.random.randn(hidden_size, input_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(output_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(output_size)
        
        # For storing activations during forward pass
        self.cache = {}
        
    def forward(self, X):
        """
        Forward pass
        X: (batch_size, input_size)
        """
        # Layer 1
        self.cache['X'] = X
        self.cache['Z1'] = np.dot(X, self.W1.T) + self.b1
        self.cache['A1'] = np.maximum(0, self.cache['Z1'])  # ReLU
        
        # Layer 2
        self.cache['Z2'] = np.dot(self.cache['A1'], self.W2.T) + self.b2
        self.cache['A2'] = 1 / (1 + np.exp(-self.cache['Z2']))  # Sigmoid
        
        return self.cache['A2']
    
    def backward(self, y, learning_rate=0.01):
        """
        Backward pass
        y: (batch_size, output_size)
        """
        m = y.shape[0]  # batch size
        
        # Output layer gradients
        dZ2 = self.cache['A2'] - y  # For binary cross-entropy + sigmoid
        dW2 = (1/m) * np.dot(dZ2.T, self.cache['A1'])
        db2 = (1/m) * np.sum(dZ2, axis=0)
        
        # Hidden layer gradients
        dA1 = np.dot(dZ2, self.W2)
        dZ1 = dA1 * (self.cache['Z1'] > 0)  # ReLU derivative
        dW1 = (1/m) * np.dot(dZ1.T, self.cache['X'])
        db1 = (1/m) * np.sum(dZ1, axis=0)
        
        # Update weights
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        
        # Return gradients for visualization
        return {'dW1': dW1, 'dW2': dW2, 'dZ1': dZ1, 'dZ2': dZ2}

# Create sample batch data
np.random.seed(42)
batch_size = 4
X_batch = np.random.randn(batch_size, 2)
y_batch = np.array([[1], [0], [1], [0]])

print(f"Batch data shape: X={X_batch.shape}, y={y_batch.shape}")

# Create and test network
net = NeuralNetwork(input_size=2, hidden_size=3, output_size=1)

# Forward pass
output = net.forward(X_batch)
print(f"\nForward pass output:\n{output}")

# Backward pass
gradients = net.backward(y_batch)
print(f"\nGradient shapes:")
print(f"  dW1: {gradients['dW1'].shape}")
print(f"  dW2: {gradients['dW2'].shape}")

# -----------------------------------------------------------------------------
# PART 4: VISUALIZING GRADIENT FLOW
# -----------------------------------------------------------------------------
print("\n\n4. VISUALIZING GRADIENT FLOW")
print("-" * 40)

# Train on XOR to visualize gradient magnitudes
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([[0], [1], [1], [0]])

# Create network
net_viz = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)

# Track gradient magnitudes during training
gradient_history = {'dW1': [], 'dW2': [], 'layer1': [], 'layer2': []}

print("Training and tracking gradients...")
for epoch in range(100):
    # Forward pass
    output = net_viz.forward(X_xor)
    
    # Backward pass
    grads = net_viz.backward(y_xor, learning_rate=0.5)
    
    # Store gradient magnitudes
    gradient_history['dW1'].append(np.abs(grads['dW1']).mean())
    gradient_history['dW2'].append(np.abs(grads['dW2']).mean())
    gradient_history['layer1'].append(np.abs(grads['dZ1']).mean())
    gradient_history['layer2'].append(np.abs(grads['dZ2']).mean())

# Plot gradient flow
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Weight gradients
ax1.plot(gradient_history['dW1'], label='dW1 (First layer)', linewidth=2)
ax1.plot(gradient_history['dW2'], label='dW2 (Second layer)', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Average Gradient Magnitude')
ax1.set_title('Weight Gradient Magnitudes During Training')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# Layer gradients
ax2.plot(gradient_history['layer1'], label='Hidden layer', linewidth=2)
ax2.plot(gradient_history['layer2'], label='Output layer', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Average Gradient Magnitude')
ax2.set_title('Layer Gradient Magnitudes During Training')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')

plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# PART 5: VANISHING GRADIENT DEMONSTRATION
# -----------------------------------------------------------------------------
print("\n\n5. VANISHING GRADIENT PROBLEM")
print("-" * 40)

# Compare gradients with different activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu_derivative(x):
    return (x > 0).astype(float)

# Simulate deep network gradient flow
depths = [1, 5, 10, 20]
x_range = np.linspace(-5, 5, 100)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for idx, depth in enumerate(depths):
    # Start with gradient of 1
    gradient_sigmoid = np.ones_like(x_range)
    gradient_relu = np.ones_like(x_range)
    
    # Propagate through layers
    for _ in range(depth):
        gradient_sigmoid *= sigmoid_derivative(x_range)
        gradient_relu *= relu_derivative(x_range)
    
    ax = axes[idx]
    ax.plot(x_range, gradient_sigmoid, 'b-', label='Sigmoid', linewidth=2)
    ax.plot(x_range, gradient_relu, 'r-', label='ReLU', linewidth=2)
    ax.set_xlabel('Input value')
    ax.set_ylabel('Gradient magnitude')
    ax.set_title(f'Gradient after {depth} layers')
    ax.set_yscale('log')
    ax.set_ylim([1e-10, 2])
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.suptitle('Vanishing Gradient: Sigmoid vs ReLU', fontsize=16)
plt.tight_layout()
plt.show()

print("Notice how sigmoid gradients vanish exponentially with depth!")

# -----------------------------------------------------------------------------
# PART 6: COMPUTATIONAL GRAPH
# -----------------------------------------------------------------------------
print("\n\n6. COMPUTATIONAL GRAPH VISUALIZATION")
print("-" * 40)

class ComputationalGraph:
    """Simple computational graph for automatic differentiation"""
    
    def __init__(self):
        self.nodes = []
        self.gradients = {}
    
    def add_node(self, value, name, parents=None):
        """Add a node to the graph"""
        node = {
            'value': value,
            'name': name,
            'parents': parents or [],
            'grad_fn': None
        }
        self.nodes.append(node)
        return node
    
    def backward(self, loss_node):
        """Compute gradients via backpropagation"""
        # Initialize gradient of loss w.r.t itself = 1
        self.gradients[loss_node['name']] = 1.0
        
        # Traverse graph in reverse order
        for node in reversed(self.nodes):
            if node['grad_fn']:
                # Compute gradients for parent nodes
                node['grad_fn'](self.gradients)

# Example: Simple computation
print("Building computational graph for: loss = (w*x + b - y)²")

# Create graph
graph = ComputationalGraph()

# Values
w_val, x_val, b_val, y_val = 2.0, 3.0, 1.0, 5.0

# Build graph
w = graph.add_node(w_val, 'w')
x = graph.add_node(x_val, 'x')
b = graph.add_node(b_val, 'b')
y = graph.add_node(y_val, 'y')

# w * x
wx = graph.add_node(w_val * x_val, 'wx', parents=[w, x])
wx['grad_fn'] = lambda g: (
    g.update({'w': g.get('w', 0) + g['wx'] * x_val}),
    g.update({'x': g.get('x', 0) + g['wx'] * w_val})
)

# wx + b
wx_plus_b = graph.add_node(wx['value'] + b_val, 'wx_plus_b', parents=[wx, b])
wx_plus_b['grad_fn'] = lambda g: (
    g.update({'wx': g.get('wx', 0) + g['wx_plus_b']}),
    g.update({'b': g.get('b', 0) + g['wx_plus_b']})
)

# wx_plus_b - y
diff = graph.add_node(wx_plus_b['value'] - y_val, 'diff', parents=[wx_plus_b, y])
diff['grad_fn'] = lambda g: (
    g.update({'wx_plus_b': g.get('wx_plus_b', 0) + g['diff']}),
    g.update({'y': g.get('y', 0) - g['diff']})
)

# diff²
loss = graph.add_node(diff['value']**2, 'loss', parents=[diff])
loss['grad_fn'] = lambda g: g.update({'diff': g.get('diff', 0) + g['loss'] * 2 * diff['value']})

# Run backward pass
graph.backward(loss)

print(f"\nForward pass:")
print(f"  w={w_val}, x={x_val}, b={b_val}, y={y_val}")
print(f"  wx = {wx['value']}")
print(f"  wx + b = {wx_plus_b['value']}")
print(f"  (wx + b) - y = {diff['value']}")
print(f"  loss = {loss['value']}")

print(f"\nGradients (via backprop):")
for name, grad in graph.gradients.items():
    print(f"  d(loss)/d({name}) = {grad}")

# -----------------------------------------------------------------------------
# EXERCISES
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print("EXERCISES - Master Backpropagation!")
print("="*60)

print("\n📝 Exercise 1: Implement Backward Pass")
print("Complete the backward pass for a 3-layer network")

class ThreeLayerNet:
    """Network with two hidden layers"""
    
    def __init__(self, sizes):
        """sizes = [input_size, hidden1, hidden2, output_size]"""
        self.W1 = np.random.randn(sizes[1], sizes[0]) * 0.1
        self.b1 = np.zeros(sizes[1])
        self.W2 = np.random.randn(sizes[2], sizes[1]) * 0.1
        self.b2 = np.zeros(sizes[2])
        self.W3 = np.random.randn(sizes[3], sizes[2]) * 0.1
        self.b3 = np.zeros(sizes[3])
        
    def forward(self, X):
        """Forward pass - implemented for you"""
        self.X = X
        self.Z1 = np.dot(X, self.W1.T) + self.b1
        self.A1 = np.maximum(0, self.Z1)
        self.Z2 = np.dot(self.A1, self.W2.T) + self.b2
        self.A2 = np.maximum(0, self.Z2)
        self.Z3 = np.dot(self.A2, self.W3.T) + self.b3
        self.A3 = 1 / (1 + np.exp(-self.Z3))
        return self.A3
    
    def backward(self, y, learning_rate=0.01):
        """
        TODO: Implement backward pass for 3-layer network
        Return dictionary of gradients
        """
        m = y.shape[0]
        
        # YOUR CODE HERE
        # Start with output layer and work backward
        
        # Output layer (layer 3)
        dZ3 = None  # Compute this
        dW3 = None
        db3 = None
        
        # Hidden layer 2
        dA2 = None
        dZ2 = None
        dW2 = None
        db2 = None
        
        # Hidden layer 1
        dA1 = None
        dZ1 = None
        dW1 = None
        db1 = None
        
        return {'dW1': dW1, 'dW2': dW2, 'dW3': dW3}

# Test your implementation
try:
    net3 = ThreeLayerNet([2, 4, 3, 1])
    X_test = np.random.randn(5, 2)
    y_test = np.random.randint(0, 2, (5, 1))
    
    output = net3.forward(X_test)
    grads = net3.backward(y_test)
    
    if grads['dW1'] is None:
        print("⚠️  Not implemented yet")
    else:
        print("✅ Gradients computed!")
        print(f"   dW1 shape: {grads['dW1'].shape}")
        print(f"   dW2 shape: {grads['dW2'].shape}")
        print(f"   dW3 shape: {grads['dW3'].shape}")
except Exception as e:
    print(f"⚠️  Error: {e}")

print("\n📝 Exercise 2: Gradient Clipping")
print("Implement gradient clipping to prevent exploding gradients")

def clip_gradients(gradients, max_norm=1.0):
    """
    Clip gradients to prevent exploding gradients.
    
    Args:
        gradients: Dictionary of gradients
        max_norm: Maximum allowed norm
    
    Returns:
        Clipped gradients
    """
    # YOUR CODE HERE
    # 1. Compute total norm of all gradients
    # 2. If norm > max_norm, scale all gradients down
    
    return gradients

# Test gradient clipping
test_grads = {
    'dW1': np.array([[10., 20.], [30., 40.]]),
    'dW2': np.array([[50., 60.]])
}

try:
    clipped = clip_gradients(test_grads, max_norm=5.0)
    if clipped is test_grads:
        print("⚠️  Not implemented yet")
    else:
        print("Original gradient norms:")
        for name, grad in test_grads.items():
            print(f"  {name}: {np.linalg.norm(grad):.2f}")
        print("\nClipped gradient norms:")
        for name, grad in clipped.items():
            print(f"  {name}: {np.linalg.norm(grad):.2f}")
except:
    print("⚠️  Error in implementation")

print("\n📝 Exercise 3: Custom Activation Backward")
print("Implement backward pass for Leaky ReLU")

def leaky_relu(x, alpha=0.01):
    """Leaky ReLU: f(x) = x if x > 0 else alpha * x"""
    return np.where(x > 0, x, alpha * x)

def leaky_relu_backward(x, grad_output, alpha=0.01):
    """
    Backward pass for Leaky ReLU.
    
    Args:
        x: Input to leaky ReLU
        grad_output: Gradient from next layer
        alpha: Leak parameter
    
    Returns:
        Gradient w.r.t input
    """
    # YOUR CODE HERE
    # Derivative is 1 if x > 0, else alpha
    
    return None

# Test Leaky ReLU backward
x_test = np.array([-2, -1, 0, 1, 2])
grad_out = np.ones_like(x_test)

try:
    grad_in = leaky_relu_backward(x_test, grad_out)
    if grad_in is None:
        print("⚠️  Not implemented yet")
    else:
        print("Input:", x_test)
        print("Gradient out:", grad_out)
        print("Gradient in:", grad_in)
        print("Expected: [0.01, 0.01, 0.01, 1., 1.]")
except:
    print("⚠️  Error in implementation")

print("\n🎉 Congratulations! You understand backpropagation!")
print("\nKey achievements:")
print("  ✓ Manually computed gradients step by step")
print("  ✓ Implemented batch backpropagation")
print("  ✓ Visualized gradient flow")
print("  ✓ Understood vanishing gradients")
print("  ✓ Built computational graphs")
print("\nNext: Advanced optimization algorithms!")