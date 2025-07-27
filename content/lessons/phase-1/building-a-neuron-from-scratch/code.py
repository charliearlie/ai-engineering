"""
Lesson 3: Calculus Essentials for Neural Networks
Understanding derivatives, gradients, and how networks learn
"""

import numpy as np
import matplotlib.pyplot as plt

print("="*60)
print("LESSON 3: CALCULUS ESSENTIALS FOR NEURAL NETWORKS")
print("="*60)

# -----------------------------------------------------------------------------
# PART 1: UNDERSTANDING DERIVATIVES
# -----------------------------------------------------------------------------
print("\n1. DERIVATIVES - Measuring Change")
print("-" * 40)

# Let's look at a simple function: f(x) = x²
def f(x):
    return x**2

# Calculate derivative numerically (the slope at a point)
def numerical_derivative(func, x, h=0.0001):
    """Calculate derivative using tiny differences"""
    return (func(x + h) - func(x)) / h

# Test at different points
test_points = [-2, -1, 0, 1, 2]
print("Function: f(x) = x²")
print("\nDerivatives at different points:")
for x in test_points:
    derivative = numerical_derivative(f, x)
    print(f"  At x={x:2d}: f(x)={f(x):4d}, slope={derivative:5.1f}")

print("\n💡 Notice: The derivative of x² is 2x")
print("   At x=3, slope=6. At x=-2, slope=-4.")

# -----------------------------------------------------------------------------
# PART 2: GRADIENT DESCENT VISUALIZATION
# -----------------------------------------------------------------------------
print("\n\n2. GRADIENT DESCENT - Finding the Minimum")
print("-" * 40)

# Simple loss function: L(w) = (w-3)² + 1
def loss_function(w):
    return (w - 3)**2 + 1

def loss_gradient(w):
    return 2 * (w - 3)

# Gradient descent implementation
def gradient_descent_demo(starting_point=0, learning_rate=0.1, steps=20):
    """Demonstrate gradient descent step by step"""
    w = starting_point
    history = [w]
    
    print(f"Starting at w={w}, Loss={loss_function(w):.3f}")
    print(f"Learning rate: {learning_rate}")
    print("\nStep-by-step descent:")
    
    for step in range(steps):
        # Calculate gradient
        gradient = loss_gradient(w)
        
        # Update weight
        w_new = w - learning_rate * gradient
        
        if step < 5:  # Show first 5 steps in detail
            print(f"  Step {step+1}: w={w:.3f}, gradient={gradient:.3f}, "
                  f"new_w={w_new:.3f}, loss={loss_function(w_new):.3f}")
        
        w = w_new
        history.append(w)
        
        # Stop if we're close enough
        if abs(gradient) < 0.001:
            print(f"\n✅ Converged at step {step+1}!")
            break
    
    print(f"\nFinal: w={w:.3f}, Loss={loss_function(w):.3f}")
    print(f"(Optimal w=3, minimum loss=1)")
    
    return history

# Run gradient descent
history = gradient_descent_demo()

# -----------------------------------------------------------------------------
# PART 3: LEARNING RATE COMPARISON
# -----------------------------------------------------------------------------
print("\n\n3. LEARNING RATE - Finding the Right Step Size")
print("-" * 40)

# Compare different learning rates
learning_rates = [0.01, 0.1, 0.5, 0.9]
histories = {}

for lr in learning_rates:
    w = 0  # Start at 0
    hist = [w]
    
    for _ in range(50):
        gradient = loss_gradient(w)
        w = w - lr * gradient
        hist.append(w)
        
        # Stop if diverging
        if abs(w) > 100:
            break
    
    histories[lr] = hist
    
    # Print summary
    final_w = hist[-1]
    final_loss = loss_function(final_w)
    status = "✅ Good" if abs(final_w - 3) < 0.1 else "❌ Poor"
    print(f"  LR={lr}: Final w={final_w:.3f}, Loss={final_loss:.3f} {status}")

# -----------------------------------------------------------------------------
# PART 4: BACKPROPAGATION CONCEPT
# -----------------------------------------------------------------------------
print("\n\n4. BACKPROPAGATION - Chain Rule in Action")
print("-" * 40)

# Simple 2-layer network example
print("Simple network: Input → Hidden → Output")
print("Let's trace how gradients flow backward:")

# Forward pass
x = 2.0  # Input
w1 = 0.5  # First weight
w2 = -0.3  # Second weight

# Layer 1: h = w1 * x
h = w1 * x
print(f"\nForward pass:")
print(f"  Input x = {x}")
print(f"  Hidden h = w1 * x = {w1} * {x} = {h}")

# Layer 2: y = w2 * h
y = w2 * h
print(f"  Output y = w2 * h = {w2} * {h} = {y}")

# Suppose target = 1.0
target = 1.0
loss = (y - target)**2
print(f"\n  Target = {target}")
print(f"  Loss = (y - target)² = ({y} - {target})² = {loss:.3f}")

# Backward pass (chain rule)
print(f"\nBackward pass (chain rule):")

# Gradient of loss with respect to y
dL_dy = 2 * (y - target)
print(f"  dL/dy = 2(y - target) = {dL_dy:.3f}")

# Gradient with respect to w2 (direct connection)
dL_dw2 = dL_dy * h  # Chain rule: dL/dw2 = dL/dy * dy/dw2
print(f"  dL/dw2 = dL/dy * h = {dL_dy:.3f} * {h} = {dL_dw2:.3f}")

# Gradient with respect to h (going deeper)
dL_dh = dL_dy * w2  # Chain rule: dL/dh = dL/dy * dy/dh
print(f"  dL/dh = dL/dy * w2 = {dL_dy:.3f} * {w2} = {dL_dh:.3f}")

# Gradient with respect to w1 (through the chain)
dL_dw1 = dL_dh * x  # Chain rule: dL/dw1 = dL/dh * dh/dw1
print(f"  dL/dw1 = dL/dh * x = {dL_dh:.3f} * {x} = {dL_dw1:.3f}")

print("\n💡 Notice how we multiply gradients backward through the network!")

# -----------------------------------------------------------------------------
# PART 5: VISUALIZATIONS
# -----------------------------------------------------------------------------
print("\n\n5. VISUALIZING THE CONCEPTS")
print("-" * 40)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Function and derivative
ax1 = axes[0, 0]
x = np.linspace(-3, 3, 100)
y = x**2
y_derivative = 2*x

ax1.plot(x, y, 'b-', linewidth=2, label='f(x) = x²')
ax1.plot(x, y_derivative, 'r--', linewidth=2, label="f'(x) = 2x (derivative)")
ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax1.axvline(x=0, color='k', linestyle='-', alpha=0.3)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Function and Its Derivative', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Gradient descent path
ax2 = axes[0, 1]
w_range = np.linspace(-2, 8, 100)
loss_curve = (w_range - 3)**2 + 1

ax2.plot(w_range, loss_curve, 'b-', linewidth=2, label='Loss function')
ax2.plot(history, [loss_function(w) for w in history], 'ro-', 
         linewidth=2, markersize=6, label='Gradient descent path')
ax2.axvline(x=3, color='g', linestyle='--', alpha=0.5, label='Optimal w=3')
ax2.set_xlabel('Weight (w)')
ax2.set_ylabel('Loss')
ax2.set_title('Gradient Descent Finding Minimum', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Learning rate comparison
ax3 = axes[1, 0]
for lr, hist in histories.items():
    steps = range(len(hist))
    ax3.plot(steps, hist, label=f'LR={lr}', linewidth=2)

ax3.axhline(y=3, color='g', linestyle='--', alpha=0.5, label='Target w=3')
ax3.set_xlabel('Steps')
ax3.set_ylabel('Weight value')
ax3.set_title('Effect of Learning Rate', fontsize=14)
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_ylim(-5, 10)

# 4. Chain rule visualization
ax4 = axes[1, 1]
ax4.text(0.5, 0.9, 'Backpropagation Flow', ha='center', fontsize=16,
         transform=ax4.transAxes, weight='bold')

# Draw network
layers_x = [0.2, 0.5, 0.8]
layers_y = [0.5, 0.5, 0.5]
labels = ['Input\nx=2', 'Hidden\nh=1', 'Output\ny=-0.3']

for i, (x_pos, y_pos, label) in enumerate(zip(layers_x, layers_y, labels)):
    circle = plt.Circle((x_pos, y_pos), 0.08, color='lightblue', 
                       edgecolor='blue', linewidth=2)
    ax4.add_patch(circle)
    ax4.text(x_pos, y_pos, label, ha='center', va='center', fontsize=10)

# Draw connections with gradients
ax4.arrow(0.28, 0.5, 0.14, 0, head_width=0.03, head_length=0.02,
         fc='gray', ec='gray')
ax4.text(0.35, 0.55, 'w1=0.5', ha='center', fontsize=9)
ax4.text(0.35, 0.42, 'dL/dw1=-0.36', ha='center', fontsize=9, color='red')

ax4.arrow(0.58, 0.5, 0.14, 0, head_width=0.03, head_length=0.02,
         fc='gray', ec='gray')
ax4.text(0.65, 0.55, 'w2=-0.3', ha='center', fontsize=9)
ax4.text(0.65, 0.42, 'dL/dw2=-2.6', ha='center', fontsize=9, color='red')

# Gradient flow arrows (backward)
ax4.arrow(0.72, 0.38, -0.14, 0, head_width=0.03, head_length=0.02,
         fc='red', ec='red', alpha=0.6, linestyle='--')
ax4.arrow(0.42, 0.38, -0.14, 0, head_width=0.03, head_length=0.02,
         fc='red', ec='red', alpha=0.6, linestyle='--')

ax4.text(0.5, 0.25, 'Gradients flow backward', ha='center', 
         fontsize=11, color='red', style='italic')

ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.axis('off')

plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# EXERCISES
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print("EXERCISES - Implement gradient descent!")
print("="*60)

print("\n📝 Exercise 1: Calculate Gradient")
def calculate_gradient(w, x, y, target):
    """
    Calculate the gradient for a simple linear model.
    Model: y_pred = w * x
    Loss: L = (y_pred - target)²
    
    Return dL/dw (gradient with respect to weight)
    
    Hint: dL/dw = 2 * (y_pred - target) * x
    """
    # YOUR CODE HERE
    y_pred = None  # Calculate prediction
    gradient = None  # Calculate gradient
    
    return gradient

# Test
test_w = 0.5
test_x = 2.0
test_target = 3.0

try:
    grad = calculate_gradient(test_w, test_x, None, test_target)
    expected_pred = test_w * test_x
    expected_grad = 2 * (expected_pred - test_target) * test_x
    
    if grad is None:
        print("⚠️  Not implemented yet")
    elif np.isclose(grad, expected_grad):
        print(f"✅ Correct! Gradient = {grad}")
    else:
        print(f"❌ Not quite. Expected {expected_grad}, got {grad}")
except:
    print("⚠️  Error in implementation")

print("\n📝 Exercise 2: Gradient Descent Step")
def gradient_descent_step(w, gradient, learning_rate):
    """
    Perform one step of gradient descent.
    
    Return the updated weight.
    """
    # YOUR CODE HERE
    w_new = None  # Update the weight
    
    return w_new

# Test
test_w = 1.0
test_gradient = 0.5
test_lr = 0.1

try:
    new_w = gradient_descent_step(test_w, test_gradient, test_lr)
    expected = test_w - test_lr * test_gradient
    
    if new_w is None:
        print("⚠️  Not implemented yet")
    elif np.isclose(new_w, expected):
        print(f"✅ Correct! New weight = {new_w}")
    else:
        print(f"❌ Not quite. Expected {expected}, got {new_w}")
except:
    print("⚠️  Error in implementation")

print("\n📝 Exercise 3: Complete Learning Loop")
def train_simple_model(X, y, epochs=100, learning_rate=0.01):
    """
    Train a simple linear model y = w*x using gradient descent.
    
    Args:
        X: Input data (array)
        y: Target values (array)
        epochs: Number of training iterations
        learning_rate: Step size
    
    Returns:
        w: Learned weight
        loss_history: List of losses over time
    """
    # Initialize weight randomly
    w = np.random.randn()
    loss_history = []
    
    for epoch in range(epochs):
        # YOUR CODE HERE
        # 1. Calculate predictions for all X
        # 2. Calculate loss (mean squared error)
        # 3. Calculate gradient
        # 4. Update weight
        # 5. Store loss
        
        predictions = None  # Replace with your code
        loss = None  # Replace with your code
        gradient = None  # Replace with your code
        
        # Update weight (uncomment when ready)
        # w = w - learning_rate * gradient
        # loss_history.append(loss)
    
    return w, loss_history

# Test on simple data
X_train = np.array([1, 2, 3, 4, 5])
y_train = np.array([2, 4, 6, 8, 10])  # True relationship: y = 2*x

try:
    w_learned, losses = train_simple_model(X_train, y_train)
    
    if w_learned is None:
        print("⚠️  Not implemented yet")
    elif abs(w_learned - 2.0) < 0.1:
        print(f"✅ Excellent! Learned w = {w_learned:.3f} (true value = 2.0)")
        print(f"   Final loss: {losses[-1]:.6f}")
    else:
        print(f"❌ Not quite. Learned w = {w_learned:.3f}, expected ≈ 2.0")
except:
    print("⚠️  Error in implementation")

print("\n🎉 Congratulations! You understand how neural networks learn!")
print("\nKey concepts mastered:")
print("  ✓ Derivatives measure rate of change")
print("  ✓ Gradients point toward lower loss")
print("  ✓ Gradient descent iteratively improves weights")
print("  ✓ Learning rate controls convergence speed")
print("  ✓ Chain rule enables backpropagation")
print("\nNext: Building a complete neuron from scratch!")