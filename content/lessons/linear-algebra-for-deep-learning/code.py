"""
Lesson 2: Linear Algebra for Deep Learning
Understanding vectors, matrices, and the operations that power neural networks
"""

import numpy as np
import matplotlib.pyplot as plt

print("="*60)
print("LESSON 2: LINEAR ALGEBRA FOR DEEP LEARNING")
print("="*60)

# -----------------------------------------------------------------------------
# PART 1: VECTORS - Your Data's Format
# -----------------------------------------------------------------------------
print("\n1. VECTORS - Lists of numbers")
print("-" * 40)

# Creating vectors
x = np.array([1, 2, 3])
print(f"Vector x: {x}")
print(f"Shape: {x.shape} (means {len(x)} elements)")
print(f"Type: {type(x)}")

# Vector operations
y = np.array([4, 5, 6])
print(f"\nVector y: {y}")

# Element-wise operations
print(f"\nElement-wise addition (x + y): {x + y}")
print(f"Element-wise multiplication (x * y): {x * y}")
print(f"Scalar multiplication (x * 2): {x * 2}")

# Real-world example
print("\n📊 Real Example: RGB Color")
color_vector = np.array([255, 128, 0])  # Orange color
print(f"RGB color vector: {color_vector}")
print(f"This represents: Red={color_vector[0]}, Green={color_vector[1]}, Blue={color_vector[2]}")

# -----------------------------------------------------------------------------
# PART 2: DOT PRODUCT - The Heart of Neural Networks
# -----------------------------------------------------------------------------
print("\n\n2. DOT PRODUCT - The fundamental operation")
print("-" * 40)

# Calculate dot product
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
dot_result = np.dot(v1, v2)

print(f"v1: {v1}")
print(f"v2: {v2}")
print(f"\nDot product: {dot_result}")

# Show the calculation step by step
print("\nStep by step:")
for i in range(len(v1)):
    print(f"  {v1[i]} × {v2[i]} = {v1[i] * v2[i]}")
print(f"  Sum: {' + '.join([str(v1[i]*v2[i]) for i in range(len(v1))])} = {dot_result}")

# Neural network example
print("\n🧠 Neural Network Example:")
inputs = np.array([0.5, 0.8, 0.2])  # e.g., normalized pixel values
weights = np.array([0.3, -0.5, 0.7])  # learned weights
bias = 0.1

output = np.dot(inputs, weights) + bias
print(f"Inputs: {inputs}")
print(f"Weights: {weights}")
print(f"Bias: {bias}")
print(f"Neuron output: {output:.3f}")
print(f"Activated (ReLU): {max(0, output):.3f}")

# -----------------------------------------------------------------------------
# PART 3: MATRICES - Weight Storage
# -----------------------------------------------------------------------------
print("\n\n3. MATRICES - Tables of weights")
print("-" * 40)

# Create a weight matrix (3 inputs → 2 neurons)
W = np.array([[0.5, -0.3, 0.2],    # Neuron 1's weights
              [0.1, 0.4, -0.5]])    # Neuron 2's weights

print("Weight matrix W:")
print(W)
print(f"Shape: {W.shape} (2 neurons, 3 weights each)")

# Accessing elements
print(f"\nNeuron 1's weights: {W[0]}")
print(f"Neuron 2's weights: {W[1]}")
print(f"Weight from input 2 to neuron 1: {W[0, 1]}")

# -----------------------------------------------------------------------------
# PART 4: MATRIX-VECTOR MULTIPLICATION - Layer Computation
# -----------------------------------------------------------------------------
print("\n\n4. MATRIX-VECTOR MULTIPLICATION")
print("-" * 40)

# Input vector
x = np.array([1, 2, 3])
print(f"Input x: {x}")
print(f"Weight matrix W:")
print(W)

# Calculate output for all neurons at once!
output = np.dot(W, x)
print(f"\nOutput = W @ x: {output}")

# Show what happened
print("\nBreaking it down:")
for i in range(W.shape[0]):
    calculation = ' + '.join([f"({W[i,j]}×{x[j]})" for j in range(len(x))])
    result = np.dot(W[i], x)
    print(f"  Neuron {i+1}: {calculation} = {result:.1f}")

# Add bias and activation
bias = np.array([0.1, -0.2])
activated = np.maximum(0, output + bias)  # ReLU activation
print(f"\nWith bias {bias}: {output + bias}")
print(f"After ReLU activation: {activated}")

# -----------------------------------------------------------------------------
# PART 5: BATCH PROCESSING - Multiple Inputs
# -----------------------------------------------------------------------------
print("\n\n5. BATCH PROCESSING - Multiple inputs at once")
print("-" * 40)

# Batch of 3 samples, each with 3 features
X_batch = np.array([[1, 2, 3],     # Sample 1
                    [4, 5, 6],     # Sample 2
                    [7, 8, 9]])    # Sample 3

print("Batch of inputs X:")
print(X_batch)
print(f"Shape: {X_batch.shape}")

# Process entire batch (note: we use X @ W.T for batch processing)
Y_batch = np.dot(X_batch, W.T)  # Transpose W for correct dimensions
print(f"\nBatch output Y = X @ W.T:")
print(Y_batch)
print(f"Output shape: {Y_batch.shape} (3 samples, 2 neurons each)")

# -----------------------------------------------------------------------------
# PART 6: TRANSPOSE - Flipping Matrices
# -----------------------------------------------------------------------------
print("\n\n6. TRANSPOSE - Flipping rows and columns")
print("-" * 40)

# Original matrix
A = np.array([[1, 2, 3],
              [4, 5, 6]])

print("Original matrix A:")
print(A)
print(f"Shape: {A.shape}")

print("\nTransposed A.T:")
print(A.T)
print(f"Shape: {A.T.shape}")

# Why transpose matters
print("\n💡 Why we need transpose:")
print("- Batch processing: X @ W.T")
print("- Backpropagation: gradients flow backward")
print("- Changing operation order: (A @ B).T = B.T @ A.T")

# -----------------------------------------------------------------------------
# VISUALIZATION: Understanding Shapes
# -----------------------------------------------------------------------------
print("\n\n7. VISUALIZING MATRIX OPERATIONS")
print("-" * 40)

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Vector visualization
ax1 = axes[0, 0]
vector = np.array([3, 4])
ax1.arrow(0, 0, vector[0], vector[1], head_width=0.3, head_length=0.2, 
          fc='blue', ec='blue', linewidth=2)
ax1.set_xlim(-1, 5)
ax1.set_ylim(-1, 5)
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')
ax1.set_title('Vector [3, 4]', fontsize=14)
ax1.set_xlabel('x')
ax1.set_ylabel('y')

# 2. Dot product visualization
ax2 = axes[0, 1]
v1 = np.array([4, 0])
v2 = np.array([3, 3])
ax2.arrow(0, 0, v1[0], v1[1], head_width=0.3, head_length=0.2,
          fc='blue', ec='blue', linewidth=2, label='v1=[4,0]')
ax2.arrow(0, 0, v2[0], v2[1], head_width=0.3, head_length=0.2,
          fc='red', ec='red', linewidth=2, label='v2=[3,3]')
ax2.set_xlim(-1, 5)
ax2.set_ylim(-1, 5)
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')
dot_prod = np.dot(v1, v2)
ax2.set_title(f'Dot Product: {v1} • {v2} = {dot_prod}', fontsize=14)
ax2.legend()

# 3. Matrix as transformation
ax3 = axes[1, 0]
# Show how a matrix transforms the unit square
unit_square = np.array([[0, 1, 1, 0, 0],
                       [0, 0, 1, 1, 0]])
transform_matrix = np.array([[2, 0.5],
                            [0.5, 1.5]])
transformed = transform_matrix @ unit_square

ax3.plot(unit_square[0], unit_square[1], 'b-', linewidth=2, label='Original')
ax3.plot(transformed[0], transformed[1], 'r-', linewidth=2, label='Transformed')
ax3.fill(unit_square[0], unit_square[1], alpha=0.3, color='blue')
ax3.fill(transformed[0], transformed[1], alpha=0.3, color='red')
ax3.set_xlim(-1, 3)
ax3.set_ylim(-1, 3)
ax3.grid(True, alpha=0.3)
ax3.set_aspect('equal')
ax3.set_title('Matrix as Transformation', fontsize=14)
ax3.legend()

# 4. Neural network layer
ax4 = axes[1, 1]
ax4.text(0.5, 0.9, 'Neural Network Layer', ha='center', fontsize=16, 
         transform=ax4.transAxes, weight='bold')

# Draw simple network
input_y = [0.7, 0.5, 0.3]
hidden_y = [0.6, 0.4]
input_x = [0.2] * 3
hidden_x = [0.8] * 2

# Draw neurons
for i, y in enumerate(input_y):
    ax4.scatter(input_x[i], y, s=200, c='lightblue', edgecolor='blue', linewidth=2)
    ax4.text(input_x[i], y, f'x{i}', ha='center', va='center')

for i, y in enumerate(hidden_y):
    ax4.scatter(hidden_x[i], y, s=200, c='lightcoral', edgecolor='red', linewidth=2)
    ax4.text(hidden_x[i], y, f'h{i}', ha='center', va='center')

# Draw connections
for i, iy in enumerate(input_y):
    for j, hy in enumerate(hidden_y):
        ax4.plot([input_x[i], hidden_x[j]], [iy, hy], 'gray', alpha=0.5)

ax4.text(0.2, 0.1, 'Input\n(3 neurons)', ha='center', transform=ax4.transAxes)
ax4.text(0.8, 0.1, 'Hidden\n(2 neurons)', ha='center', transform=ax4.transAxes)
ax4.text(0.5, 0.05, 'Weights: 3×2 matrix', ha='center', transform=ax4.transAxes,
         style='italic')
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.axis('off')

plt.tight_layout()
plt.suptitle('Linear Algebra in Neural Networks', fontsize=16, y=1.02)
plt.show()

# -----------------------------------------------------------------------------
# EXERCISES
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print("EXERCISES - Practice what you learned!")
print("="*60)

print("\n📝 Exercise 1: Calculate Neuron Output")
print("Complete the function to calculate a neuron's output")

def neuron_output(inputs, weights, bias):
    """
    Calculate the output of a single neuron.
    
    Args:
        inputs: numpy array of input values
        weights: numpy array of weights (same length as inputs)
        bias: single number
    
    Returns:
        The neuron's output value
    
    Formula: output = dot(inputs, weights) + bias
    """
    # YOUR CODE HERE
    output = None  # Replace with your calculation
    
    return output

# Test your function
test_inputs = np.array([1.0, 0.5, -0.5])
test_weights = np.array([0.5, -1.0, 0.5])
test_bias = 0.2

try:
    result = neuron_output(test_inputs, test_weights, test_bias)
    expected = np.dot(test_inputs, test_weights) + test_bias
    
    if result is None:
        print("⚠️  Not implemented yet")
    elif np.isclose(result, expected):
        print(f"✅ Correct! Output = {result}")
    else:
        print(f"❌ Not quite. Expected {expected}, got {result}")
except Exception as e:
    print(f"⚠️  Error: {e}")

print("\n📝 Exercise 2: Matrix Layer")
print("Implement a neural network layer")

def neural_layer(inputs, weights, biases):
    """
    Calculate the output of an entire neural network layer.
    
    Args:
        inputs: numpy array of shape (n_inputs,)
        weights: numpy array of shape (n_neurons, n_inputs)
        biases: numpy array of shape (n_neurons,)
    
    Returns:
        Output array of shape (n_neurons,)
    
    Hint: Use matrix multiplication and add biases
    """
    # YOUR CODE HERE
    output = None  # Replace with your calculation
    
    return output

# Test your layer
test_inputs = np.array([1.0, 2.0])
test_weights = np.array([[0.5, -0.5],   # Neuron 1
                        [1.0, 0.0],     # Neuron 2
                        [-0.5, 0.5]])   # Neuron 3
test_biases = np.array([0.1, 0.2, -0.1])

try:
    result = neural_layer(test_inputs, test_weights, test_biases)
    expected = np.dot(test_weights, test_inputs) + test_biases
    
    if result is None:
        print("⚠️  Not implemented yet")
    elif np.allclose(result, expected):
        print(f"✅ Correct! Layer output = {result}")
    else:
        print(f"❌ Not quite. Expected {expected}, got {result}")
except Exception as e:
    print(f"⚠️  Error: {e}")

print("\n📝 Exercise 3: Batch Processing")
print("Process multiple inputs through a layer")

def batch_neural_layer(batch_inputs, weights, biases):
    """
    Process a batch of inputs through a neural layer.
    
    Args:
        batch_inputs: numpy array of shape (batch_size, n_inputs)
        weights: numpy array of shape (n_neurons, n_inputs)
        biases: numpy array of shape (n_neurons,)
    
    Returns:
        Output array of shape (batch_size, n_neurons)
    
    Hint: For batch processing, use batch_inputs @ weights.T + biases
    """
    # YOUR CODE HERE
    output = None  # Replace with your calculation
    
    return output

# Test batch processing
batch = np.array([[1.0, 2.0],
                  [0.5, 1.5],
                  [2.0, 0.5]])

try:
    result = batch_neural_layer(batch, test_weights, test_biases)
    expected = np.dot(batch, test_weights.T) + test_biases
    
    if result is None:
        print("⚠️  Not implemented yet")
    elif np.allclose(result, expected):
        print("✅ Correct! Batch output shape:", result.shape)
        print("First sample output:", result[0])
    else:
        print("❌ Not quite. Check your matrix dimensions!")
except Exception as e:
    print(f"⚠️  Error: {e}")

print("\n🎉 Congratulations! You've mastered the linear algebra for neural networks!")
print("\nKey skills you've learned:")
print("  ✓ Vectors represent data flowing through networks")
print("  ✓ Dot products calculate neuron outputs")
print("  ✓ Matrices store layer weights efficiently")
print("  ✓ Matrix multiplication processes entire layers")
print("  ✓ Batch processing handles multiple inputs")
print("\nNext lesson: Just enough calculus to understand backpropagation!")