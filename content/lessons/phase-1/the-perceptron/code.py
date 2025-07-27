"""
Lesson 5: The Perceptron Algorithm
Implementing the first machine learning algorithm!
"""

import numpy as np
import matplotlib.pyplot as plt

print("="*60)
print("LESSON 5: THE PERCEPTRON ALGORITHM")
print("="*60)

# -----------------------------------------------------------------------------
# PART 1: THE CLASSIC PERCEPTRON
# -----------------------------------------------------------------------------
print("\n1. IMPLEMENTING THE PERCEPTRON")
print("-" * 40)

class Perceptron:
    """The classic Perceptron algorithm"""
    
    def __init__(self, n_features, learning_rate=1.0):
        """Initialize with random weights"""
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0
        self.learning_rate = learning_rate
        
    def predict(self, x):
        """Make a binary prediction (0 or 1)"""
        # Calculate weighted sum
        activation = np.dot(self.weights, x) + self.bias
        # Step function: 1 if positive, 0 if negative
        return 1 if activation > 0 else 0
    
    def train_step(self, x, y):
        """Single training step - update only if wrong"""
        # Make prediction
        y_pred = self.predict(x)
        
        # Calculate error
        error = y - y_pred
        
        # Update only if wrong
        if error != 0:
            # Perceptron update rule
            self.weights += self.learning_rate * error * x
            self.bias += self.learning_rate * error
            return True  # Made an update
        return False  # No update needed
    
    def train(self, X, y, max_epochs=100):
        """Train until convergence or max epochs"""
        n_samples = len(X)
        converged = False
        errors_per_epoch = []
        
        for epoch in range(max_epochs):
            errors = 0
            updates = 0
            
            # Go through all training examples
            for i in range(n_samples):
                if self.train_step(X[i], y[i]):
                    updates += 1
                
                # Check accuracy on all samples
                if self.predict(X[i]) != y[i]:
                    errors += 1
            
            errors_per_epoch.append(errors)
            
            # Print progress
            if epoch % 10 == 0 or errors == 0:
                print(f"  Epoch {epoch}: {errors} errors, {updates} updates")
            
            # Check convergence
            if errors == 0:
                print(f"✅ Converged at epoch {epoch}!")
                converged = True
                break
        
        if not converged:
            print(f"⚠️  Did not converge after {max_epochs} epochs")
            
        return errors_per_epoch

# Test on AND gate
print("\nTraining Perceptron on AND gate:")
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])

perceptron_and = Perceptron(n_features=2)
errors = perceptron_and.train(X_and, y_and)

print("\nFinal weights:", perceptron_and.weights)
print("Final bias:", perceptron_and.bias)

# Test predictions
print("\nTesting learned AND gate:")
for i in range(len(X_and)):
    pred = perceptron_and.predict(X_and[i])
    print(f"  {X_and[i]} → {pred} (correct: {y_and[i]})")

# -----------------------------------------------------------------------------
# PART 2: VISUALIZING THE DECISION BOUNDARY
# -----------------------------------------------------------------------------
print("\n\n2. VISUALIZING PERCEPTRON LEARNING")
print("-" * 40)

def plot_perceptron_boundary(perceptron, X, y, title):
    """Plot data points and decision boundary"""
    # Create a mesh
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # Predict for each point in mesh
    Z = np.array([perceptron.predict([x, y]) 
                  for x, y in zip(xx.ravel(), yy.ravel())])
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], colors=['lightcoral', 'lightblue'], alpha=0.8)
    
    # Plot data points
    colors = ['red', 'blue']
    markers = ['o', 's']
    for i in range(2):
        mask = y == i
        plt.scatter(X[mask, 0], X[mask, 1], c=colors[i], marker=markers[i], 
                   s=200, edgecolor='black', linewidth=2, label=f'Class {i}')
    
    # Plot decision boundary line
    if perceptron.weights[1] != 0:
        x_boundary = np.linspace(x_min, x_max, 100)
        y_boundary = -(perceptron.weights[0] * x_boundary + perceptron.bias) / perceptron.weights[1]
        plt.plot(x_boundary, y_boundary, 'k--', linewidth=2, label='Decision boundary')
    
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

# Visualize AND gate
plot_perceptron_boundary(perceptron_and, X_and, y_and, "Perceptron: AND Gate")
plt.show()

# -----------------------------------------------------------------------------
# PART 3: OR GATE AND CONVERGENCE
# -----------------------------------------------------------------------------
print("\n\n3. LEARNING OR GATE")
print("-" * 40)

# OR gate data
X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_or = np.array([0, 1, 1, 1])

print("OR gate truth table:")
for i in range(len(X_or)):
    print(f"  {X_or[i,0]} OR {X_or[i,1]} = {y_or[i]}")

# Train perceptron
perceptron_or = Perceptron(n_features=2)
errors_or = perceptron_or.train(X_or, y_or)

# Visualize
plot_perceptron_boundary(perceptron_or, X_or, y_or, "Perceptron: OR Gate")
plt.show()

# -----------------------------------------------------------------------------
# PART 4: THE XOR PROBLEM
# -----------------------------------------------------------------------------
print("\n\n4. THE FAMOUS XOR PROBLEM")
print("-" * 40)

# XOR gate data
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

print("XOR gate truth table:")
for i in range(len(X_xor)):
    print(f"  {X_xor[i,0]} XOR {X_xor[i,1]} = {y_xor[i]}")

print("\n⚠️  XOR is NOT linearly separable!")
print("Let's see what happens when we try to learn it...")

# Try to train perceptron on XOR
perceptron_xor = Perceptron(n_features=2)
errors_xor = perceptron_xor.train(X_xor, y_xor, max_epochs=1000)

# Show it oscillates
print(f"\nErrors in last 10 epochs: {errors_xor[-10:]}")
print("Notice: The perceptron cannot converge on XOR!")

# Visualize the impossible task
plot_perceptron_boundary(perceptron_xor, X_xor, y_xor, 
                        "Perceptron on XOR: No Linear Solution!")
plt.show()

# -----------------------------------------------------------------------------
# PART 5: LEARNING RATE EFFECTS
# -----------------------------------------------------------------------------
print("\n\n5. EFFECT OF LEARNING RATE")
print("-" * 40)

# Create a more complex dataset
np.random.seed(42)
n_samples = 100

# Generate two classes
class1 = np.random.randn(n_samples//2, 2) + [-1, -1]
class2 = np.random.randn(n_samples//2, 2) + [1, 1]
X_complex = np.vstack([class1, class2])
y_complex = np.hstack([np.zeros(n_samples//2, dtype=int), 
                       np.ones(n_samples//2, dtype=int)])

# Shuffle the data
shuffle_idx = np.random.permutation(n_samples)
X_complex = X_complex[shuffle_idx]
y_complex = y_complex[shuffle_idx]

# Train with different learning rates
learning_rates = [0.01, 0.1, 1.0]
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, lr in enumerate(learning_rates):
    print(f"\nTraining with learning rate = {lr}")
    perceptron = Perceptron(n_features=2, learning_rate=lr)
    errors = perceptron.train(X_complex, y_complex, max_epochs=50)
    
    # Plot convergence
    ax = axes[idx]
    ax.plot(errors, 'b-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Number of Errors')
    ax.set_title(f'Learning Rate = {lr}')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(errors) + 5)

plt.suptitle('Perceptron Convergence with Different Learning Rates', fontsize=14)
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# PART 6: MULTI-CLASS PERCEPTRON
# -----------------------------------------------------------------------------
print("\n\n6. MULTI-CLASS CLASSIFICATION")
print("-" * 40)

class MultiClassPerceptron:
    """One-vs-All perceptron for multiple classes"""
    
    def __init__(self, n_features, n_classes):
        """Initialize one perceptron per class"""
        self.n_classes = n_classes
        self.perceptrons = [Perceptron(n_features) for _ in range(n_classes)]
    
    def train(self, X, y, max_epochs=100):
        """Train all perceptrons"""
        for class_idx in range(self.n_classes):
            print(f"\nTraining perceptron for class {class_idx}:")
            # Create binary labels: 1 for this class, 0 for others
            binary_y = (y == class_idx).astype(int)
            self.perceptrons[class_idx].train(X, binary_y, max_epochs)
    
    def predict(self, x):
        """Predict class with highest activation"""
        activations = []
        for perceptron in self.perceptrons:
            # Get raw activation (before step function)
            activation = np.dot(perceptron.weights, x) + perceptron.bias
            activations.append(activation)
        return np.argmax(activations)

# Create 3-class dataset
np.random.seed(42)
n_per_class = 30

# Three clusters
class0 = np.random.randn(n_per_class, 2) + [-2, 0]
class1 = np.random.randn(n_per_class, 2) + [2, 0]
class2 = np.random.randn(n_per_class, 2) + [0, 2]

X_multi = np.vstack([class0, class1, class2])
y_multi = np.hstack([np.zeros(n_per_class), 
                     np.ones(n_per_class), 
                     2 * np.ones(n_per_class)]).astype(int)

# Train multi-class perceptron
print("Training multi-class perceptron on 3 classes:")
mc_perceptron = MultiClassPerceptron(n_features=2, n_classes=3)
mc_perceptron.train(X_multi, y_multi, max_epochs=50)

# Test accuracy
correct = 0
for i in range(len(X_multi)):
    pred = mc_perceptron.predict(X_multi[i])
    if pred == y_multi[i]:
        correct += 1

print(f"\nMulti-class accuracy: {correct}/{len(X_multi)} = {100*correct/len(X_multi):.1f}%")

# Visualize multi-class boundaries
plt.figure(figsize=(8, 8))
colors = ['red', 'green', 'blue']
markers = ['o', 's', '^']

# Plot data points
for class_idx in range(3):
    mask = y_multi == class_idx
    plt.scatter(X_multi[mask, 0], X_multi[mask, 1], 
               c=colors[class_idx], marker=markers[class_idx], 
               s=100, edgecolor='black', linewidth=1, 
               label=f'Class {class_idx}', alpha=0.7)

# Create mesh for decision regions
x_min, x_max = X_multi[:, 0].min() - 1, X_multi[:, 0].max() + 1
y_min, y_max = X_multi[:, 1].min() - 1, X_multi[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

# Predict for each point
Z = np.array([mc_perceptron.predict([x, y]) 
              for x, y in zip(xx.ravel(), yy.ravel())])
Z = Z.reshape(xx.shape)

# Plot decision regions
plt.contourf(xx, yy, Z, alpha=0.3, levels=[-0.5, 0.5, 1.5, 2.5],
             colors=['lightcoral', 'lightgreen', 'lightblue'])

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Multi-Class Perceptron: One-vs-All Strategy')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# -----------------------------------------------------------------------------
# EXERCISES
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print("EXERCISES - Master the Perceptron!")
print("="*60)

print("\n📝 Exercise 1: Implement NAND Gate")
print("Train a perceptron to learn NAND (NOT AND)")

# NAND gate data
X_nand = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_nand = np.array([1, 1, 1, 0])  # Opposite of AND

def train_nand_perceptron():
    """
    Create and train a perceptron for NAND gate.
    Return the trained perceptron.
    """
    # YOUR CODE HERE
    perceptron = None  # Create and train perceptron
    
    return perceptron

# Test your implementation
try:
    nand_perceptron = train_nand_perceptron()
    if nand_perceptron is None:
        print("⚠️  Not implemented yet")
    else:
        print("\nTesting NAND gate:")
        correct = 0
        for i in range(len(X_nand)):
            pred = nand_perceptron.predict(X_nand[i])
            is_correct = pred == y_nand[i]
            correct += is_correct
            print(f"  {X_nand[i]} → {pred} (target: {y_nand[i]}) {'✓' if is_correct else '✗'}")
        print(f"Accuracy: {correct}/{len(X_nand)}")
except Exception as e:
    print(f"⚠️  Error: {e}")

print("\n📝 Exercise 2: Averaged Perceptron")
print("Implement an averaged perceptron for better generalization")

class AveragedPerceptron:
    """
    Perceptron that averages weights over all iterations.
    This often generalizes better than the final weights.
    """
    def __init__(self, n_features, learning_rate=1.0):
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0
        self.learning_rate = learning_rate
        
        # YOUR CODE HERE
        # Add variables to track averaged weights
        self.averaged_weights = None
        self.averaged_bias = None
        self.update_count = None
    
    def train(self, X, y, max_epochs=100):
        """
        Train and maintain averaged weights.
        
        Hint: Keep a running sum of all weight vectors
        """
        # YOUR CODE HERE
        pass
    
    def predict(self, x):
        """Use averaged weights for prediction"""
        # YOUR CODE HERE
        pass

# Test averaged perceptron
try:
    avg_perceptron = AveragedPerceptron(n_features=2)
    if avg_perceptron.averaged_weights is None:
        print("⚠️  Not implemented yet")
    else:
        # Train on noisy data
        avg_perceptron.train(X_complex[:50], y_complex[:50])
        
        # Test on remaining data
        correct = 0
        for i in range(50, 100):
            pred = avg_perceptron.predict(X_complex[i])
            if pred == y_complex[i]:
                correct += 1
        
        print(f"Averaged Perceptron Test Accuracy: {correct}/50 = {100*correct/50:.1f}%")
except:
    print("⚠️  Error in implementation")

print("\n📝 Exercise 3: Pocket Algorithm")
print("Keep the best weights seen during training")

def pocket_algorithm(X, y, max_epochs=100):
    """
    Implement the Pocket algorithm:
    1. Train a perceptron normally
    2. Keep track of the best weights (lowest error)
    3. Return the best weights, not the final weights
    
    Returns: (best_weights, best_bias, best_error_count)
    """
    # YOUR CODE HERE
    best_weights = None
    best_bias = None
    best_error_count = None
    
    return best_weights, best_bias, best_error_count

# Test pocket algorithm
try:
    # Create a dataset where perceptron might oscillate
    X_noisy = np.vstack([X_xor, X_xor + np.random.randn(4, 2) * 0.1])
    y_noisy = np.hstack([y_xor, y_xor])
    
    best_w, best_b, best_err = pocket_algorithm(X_noisy, y_noisy)
    
    if best_w is None:
        print("⚠️  Not implemented yet")
    else:
        print(f"Pocket algorithm found weights with {best_err} errors")
        print(f"Best weights: {best_w}")
        print(f"Best bias: {best_b}")
except:
    print("⚠️  Error in implementation")

print("\n🎉 You've mastered the Perceptron algorithm!")
print("\nKey achievements:")
print("  ✓ Implemented the classic perceptron")
print("  ✓ Visualized decision boundaries")
print("  ✓ Understood linear separability and XOR problem")
print("  ✓ Explored multi-class classification")
print("  ✓ Learned about perceptron variants")
print("\nNext: Connecting neurons into multi-layer networks!")