"""
Lesson 4: Building a Neuron from Scratch
Let's build a working neuron that can learn!
"""

import numpy as np
import matplotlib.pyplot as plt

print("="*60)
print("LESSON 4: BUILDING A NEURON FROM SCRATCH")
print("="*60)

# -----------------------------------------------------------------------------
# PART 1: ACTIVATION FUNCTIONS
# -----------------------------------------------------------------------------
print("\n1. ACTIVATION FUNCTIONS - Adding Non-linearity")
print("-" * 40)

# Define activation functions and their derivatives
def sigmoid(x):
    """Sigmoid: squashes input to [0, 1]"""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Derivative of sigmoid for backpropagation"""
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    """ReLU: passes positive values, blocks negative"""
    return np.maximum(0, x)

def relu_derivative(x):
    """Derivative of ReLU"""
    return (x > 0).astype(float)

def tanh(x):
    """Tanh: squashes input to [-1, 1]"""
    return np.tanh(x)

def tanh_derivative(x):
    """Derivative of tanh"""
    return 1 - np.tanh(x)**2

# Visualize activation functions
x = np.linspace(-5, 5, 100)

print("Common activation functions:")
print("  - Sigmoid: Smooth, outputs [0,1], good for probabilities")
print("  - ReLU: Simple, fast, most popular for hidden layers")
print("  - Tanh: Like sigmoid but centered at zero, outputs [-1,1]")

# -----------------------------------------------------------------------------
# PART 2: A SIMPLE NEURON CLASS
# -----------------------------------------------------------------------------
print("\n\n2. BUILDING A NEURON CLASS")
print("-" * 40)

class Neuron:
    """A single neuron with configurable activation"""
    
    def __init__(self, n_inputs, activation='sigmoid'):
        """Initialize neuron with random weights"""
        # Initialize weights (small random values)
        self.weights = np.random.randn(n_inputs) * 0.1
        self.bias = 0.0
        
        # Set activation function
        self.activation_name = activation
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif activation == 'relu':
            self.activation = relu
            self.activation_derivative = relu_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_derivative = tanh_derivative
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # For storing values during forward pass (needed for backprop)
        self.last_input = None
        self.last_output = None
        self.last_pre_activation = None
    
    def forward(self, inputs):
        """Forward pass: compute neuron output"""
        # Store for backpropagation
        self.last_input = inputs
        
        # Linear combination: w·x + b
        self.last_pre_activation = np.dot(self.weights, inputs) + self.bias
        
        # Apply activation function
        self.last_output = self.activation(self.last_pre_activation)
        
        return self.last_output
    
    def backward(self, error):
        """Backward pass: compute gradients"""
        # Gradient of activation function
        activation_gradient = self.activation_derivative(self.last_pre_activation)
        
        # Error after activation
        delta = error * activation_gradient
        
        # Gradients for weights and bias
        weight_gradients = delta * self.last_input
        bias_gradient = delta
        
        return weight_gradients, bias_gradient
    
    def update_weights(self, weight_gradients, bias_gradient, learning_rate):
        """Update weights using gradients"""
        self.weights -= learning_rate * weight_gradients
        self.bias -= learning_rate * bias_gradient

# Create and test a neuron
print("Creating a neuron with 3 inputs...")
neuron = Neuron(n_inputs=3, activation='sigmoid')

# Test forward pass
test_input = np.array([1.0, 0.5, -0.3])
output = neuron.forward(test_input)

print(f"\nTest forward pass:")
print(f"  Inputs: {test_input}")
print(f"  Weights: {neuron.weights}")
print(f"  Bias: {neuron.bias:.3f}")
print(f"  Output: {output:.3f}")

# -----------------------------------------------------------------------------
# PART 3: TRAINING A NEURON - AND Gate
# -----------------------------------------------------------------------------
print("\n\n3. TRAINING A NEURON - Learning AND Gate")
print("-" * 40)

# AND gate training data
# AND: outputs 1 only when both inputs are 1
X_and = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
y_and = np.array([0, 0, 0, 1])

print("AND gate truth table:")
print("  A  B  | Output")
print("  ------+-------")
for i in range(len(X_and)):
    print(f"  {X_and[i,0]}  {X_and[i,1]}  |   {y_and[i]}")

# Train neuron
def train_neuron(neuron, X, y, epochs=1000, learning_rate=0.1):
    """Train a neuron using gradient descent"""
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        
        # Train on each example
        for i in range(len(X)):
            # Forward pass
            prediction = neuron.forward(X[i])
            
            # Calculate error (for binary classification)
            error = prediction - y[i]
            epoch_loss += error**2
            
            # Backward pass
            weight_grads, bias_grad = neuron.backward(error)
            
            # Update weights
            neuron.update_weights(weight_grads, bias_grad, learning_rate)
        
        # Average loss
        losses.append(epoch_loss / len(X))
        
        # Print progress
        if epoch % 100 == 0:
            print(f"  Epoch {epoch}: Loss = {losses[-1]:.4f}")
    
    return losses

# Create and train neuron
print("\nTraining neuron on AND gate...")
and_neuron = Neuron(n_inputs=2, activation='sigmoid')
losses = train_neuron(and_neuron, X_and, y_and, epochs=1000, learning_rate=1.0)

# Test the trained neuron
print("\nTesting trained neuron:")
for i in range(len(X_and)):
    prediction = and_neuron.forward(X_and[i])
    print(f"  Input: {X_and[i]} → Output: {prediction:.3f} (target: {y_and[i]})")

print(f"\nLearned weights: {and_neuron.weights}")
print(f"Learned bias: {and_neuron.bias:.3f}")

# -----------------------------------------------------------------------------
# PART 4: VISUALIZING THE DECISION BOUNDARY
# -----------------------------------------------------------------------------
print("\n\n4. VISUALIZING WHAT THE NEURON LEARNED")
print("-" * 40)

# Create a grid of points
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 100),
                     np.linspace(-0.5, 1.5, 100))
grid_points = np.c_[xx.ravel(), yy.ravel()]

# Get neuron predictions for all points
Z = np.array([and_neuron.forward(point) for point in grid_points])
Z = Z.reshape(xx.shape)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Decision boundary
contour = ax1.contourf(xx, yy, Z, levels=20, cmap='RdBu', alpha=0.8)
ax1.scatter(X_and[:, 0], X_and[:, 1], c=y_and, cmap='RdBu', 
            s=200, edgecolor='black', linewidth=2)

# Add labels for data points
for i in range(len(X_and)):
    ax1.annotate(f'({X_and[i,0]},{X_and[i,1]})', 
                xy=(X_and[i,0], X_and[i,1]), 
                xytext=(5, 5), textcoords='offset points')

ax1.set_xlabel('Input A')
ax1.set_ylabel('Input B')
ax1.set_title('AND Gate Decision Boundary')
ax1.grid(True, alpha=0.3)
plt.colorbar(contour, ax=ax1, label='Neuron Output')

# Plot 2: Training loss
ax2.plot(losses, 'b-', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Mean Squared Error')
ax2.set_title('Training Loss Over Time')
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')

plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# PART 5: DIFFERENT ACTIVATION FUNCTIONS
# -----------------------------------------------------------------------------
print("\n\n5. COMPARING ACTIVATION FUNCTIONS")
print("-" * 40)

# Train neurons with different activations
activations = ['sigmoid', 'tanh', 'relu']
trained_neurons = {}

for activation in activations:
    print(f"\nTraining with {activation} activation...")
    neuron = Neuron(n_inputs=2, activation=activation)
    
    # ReLU needs different learning rate
    lr = 0.01 if activation == 'relu' else 1.0
    
    losses = train_neuron(neuron, X_and, y_and, epochs=1000, learning_rate=lr)
    trained_neurons[activation] = neuron
    
    # Test
    print(f"Results with {activation}:")
    for i in range(len(X_and)):
        pred = neuron.forward(X_and[i])
        print(f"  {X_and[i]} → {pred:.3f}")

# Visualize all activation functions
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Row 1: Activation functions
x_range = np.linspace(-3, 3, 100)
activations_plot = [
    ('Sigmoid', sigmoid(x_range), axes[0, 0]),
    ('Tanh', tanh(x_range), axes[0, 1]),
    ('ReLU', relu(x_range), axes[0, 2])
]

for name, y_values, ax in activations_plot:
    ax.plot(x_range, y_values, 'b-', linewidth=3)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'{name} Activation', fontsize=14)
    ax.set_xlabel('Input')
    ax.set_ylabel('Output')
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)

# Row 2: Decision boundaries
for idx, (name, neuron) in enumerate(trained_neurons.items()):
    ax = axes[1, idx]
    
    # Calculate decision boundary
    Z = np.array([neuron.forward(point) for point in grid_points])
    Z = Z.reshape(xx.shape)
    
    # Plot
    contour = ax.contourf(xx, yy, Z, levels=20, cmap='RdBu', alpha=0.8)
    ax.scatter(X_and[:, 0], X_and[:, 1], c=y_and, cmap='RdBu', 
              s=200, edgecolor='black', linewidth=2)
    
    ax.set_title(f'{name} Decision Boundary', fontsize=14)
    ax.set_xlabel('Input A')
    ax.set_ylabel('Input B')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# EXERCISES
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print("EXERCISES - Build your own neurons!")
print("="*60)

print("\n📝 Exercise 1: Implement OR Gate")
print("Train a neuron to learn the OR gate")

# OR gate data
X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_or = np.array([0, 1, 1, 1])  # OR: 1 if either input is 1

def train_or_gate():
    """
    Create and train a neuron to implement OR gate.
    Return the trained neuron.
    """
    # YOUR CODE HERE
    neuron = None  # Create a neuron
    # Train it on X_or, y_or
    
    return neuron

# Test your implementation
try:
    or_neuron = train_or_gate()
    if or_neuron is None:
        print("⚠️  Not implemented yet")
    else:
        print("Testing OR gate:")
        correct = 0
        for i in range(len(X_or)):
            pred = or_neuron.forward(X_or[i])
            target = y_or[i]
            is_correct = (pred > 0.5) == target
            correct += is_correct
            print(f"  {X_or[i]} → {pred:.3f} (target: {target}) {'✓' if is_correct else '✗'}")
        print(f"Accuracy: {correct}/4")
except:
    print("⚠️  Error in implementation")

print("\n📝 Exercise 2: Multi-class Neuron")
print("Implement a neuron that can output values for 3 classes")

def softmax(x):
    """Softmax activation for multi-class"""
    exp_x = np.exp(x - np.max(x))  # Stability trick
    return exp_x / np.sum(exp_x)

class MultiClassNeuron:
    """
    A neuron that outputs probabilities for multiple classes.
    
    TODO: Implement forward pass using softmax activation
    """
    def __init__(self, n_inputs, n_classes):
        # YOUR CODE HERE
        # Initialize weights for each class
        self.weights = None  # Should be shape (n_classes, n_inputs)
        self.bias = None     # Should be shape (n_classes,)
    
    def forward(self, inputs):
        """
        Forward pass with softmax activation.
        Returns probability distribution over classes.
        """
        # YOUR CODE HERE
        # 1. Compute linear combination for each class
        # 2. Apply softmax
        return None

# Test your implementation
try:
    mc_neuron = MultiClassNeuron(n_inputs=2, n_classes=3)
    if mc_neuron.weights is None:
        print("⚠️  Not implemented yet")
    else:
        test_input = np.array([0.5, 0.5])
        output = mc_neuron.forward(test_input)
        print(f"Input: {test_input}")
        print(f"Output probabilities: {output}")
        print(f"Sum of probabilities: {np.sum(output):.3f} (should be 1.0)")
except:
    print("⚠️  Error in implementation")

print("\n📝 Exercise 3: Custom Loss Function")
print("Implement a neuron with custom loss for regression")

def train_regression_neuron(X, y, epochs=100):
    """
    Train a neuron for regression (no activation function).
    Use Mean Absolute Error (MAE) instead of MSE.
    
    Hint: MAE gradient = sign(prediction - target)
    """
    neuron = Neuron(n_inputs=X.shape[1], activation='relu')
    losses = []
    
    # YOUR CODE HERE
    # Implement training loop with MAE loss
    
    return neuron, losses

# Test on simple regression data
X_reg = np.array([[1], [2], [3], [4], [5]])
y_reg = np.array([2, 4, 6, 8, 10])  # y = 2x

try:
    reg_neuron, reg_losses = train_regression_neuron(X_reg, y_reg)
    if reg_neuron is None:
        print("⚠️  Not implemented yet")
    else:
        print("Testing regression:")
        for i in range(len(X_reg)):
            pred = reg_neuron.forward(X_reg[i])
            print(f"  x={X_reg[i,0]} → prediction={pred:.2f} (target={y_reg[i]})")
except:
    print("⚠️  Error in implementation")

print("\n🎉 Congratulations! You've built neurons from scratch!")
print("\nKey achievements:")
print("  ✓ Implemented forward and backward passes")
print("  ✓ Trained neurons with gradient descent")
print("  ✓ Visualized decision boundaries")
print("  ✓ Compared different activation functions")
print("  ✓ Solved real classification problems")
print("\nNext: The Perceptron algorithm - a classic approach to learning!")