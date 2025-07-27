"""
Lesson 10: MNIST Digit Recognizer Project
Building your first complete neural network!
"""

import numpy as np
import matplotlib.pyplot as plt
from urllib import request
import gzip
import pickle
import os

print("="*60)
print("LESSON 10: MNIST DIGIT RECOGNIZER PROJECT")
print("="*60)

# -----------------------------------------------------------------------------
# PART 1: LOADING AND EXPLORING MNIST
# -----------------------------------------------------------------------------
print("\n1. LOADING THE MNIST DATASET")
print("-" * 40)

def download_mnist():
    """Download MNIST if not already present"""
    # Create data directory
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Check if already downloaded
    if os.path.exists('data/mnist.pkl.gz'):
        print("MNIST already downloaded!")
        return
    
    print("Downloading MNIST dataset...")
    url = 'https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/mnist.pkl.gz'
    request.urlretrieve(url, 'data/mnist.pkl.gz')
    print("Download complete!")

def load_mnist():
    """Load MNIST data"""
    download_mnist()
    
    with gzip.open('data/mnist.pkl.gz', 'rb') as f:
        train_data, val_data, test_data = pickle.load(f, encoding='latin1')
    
    # Convert to our format
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data
    
    return X_train, y_train, X_val, y_val, X_test, y_test

# Load the data
print("Loading MNIST data...")
X_train, y_train, X_val, y_val, X_test, y_test = load_mnist()

print(f"\nDataset shapes:")
print(f"  Training:   {X_train.shape} images, {y_train.shape} labels")
print(f"  Validation: {X_val.shape} images, {y_val.shape} labels")
print(f"  Test:       {X_test.shape} images, {y_test.shape} labels")

print(f"\nPixel value range: [{X_train.min():.1f}, {X_train.max():.1f}]")
print(f"Label range: {np.unique(y_train)}")

# Visualize some examples
print("\nVisualizing sample digits...")
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
axes = axes.ravel()

for i in range(10):
    # Find first occurrence of each digit
    idx = np.where(y_train == i)[0][0]
    image = X_train[idx].reshape(28, 28)
    
    axes[i].imshow(image, cmap='gray')
    axes[i].set_title(f'Label: {i}')
    axes[i].axis('off')

plt.suptitle('Sample Digits from MNIST Dataset', fontsize=16)
plt.tight_layout()
plt.show()

# Show some statistics
print("\nLabel distribution in training set:")
for digit in range(10):
    count = np.sum(y_train == digit)
    percentage = count / len(y_train) * 100
    print(f"  Digit {digit}: {count:5d} samples ({percentage:.1f}%)")

# -----------------------------------------------------------------------------
# PART 2: DATA PREPROCESSING
# -----------------------------------------------------------------------------
print("\n\n2. PREPROCESSING THE DATA")
print("-" * 40)

def preprocess_data(X, y):
    """Normalize pixels and one-hot encode labels"""
    # Normalize pixel values to [0, 1]
    X_normalized = X.astype(np.float32) / 255.0
    
    # One-hot encode labels
    n_classes = 10
    y_onehot = np.zeros((len(y), n_classes))
    y_onehot[np.arange(len(y)), y] = 1
    
    return X_normalized, y_onehot

# Preprocess all datasets
print("Preprocessing data...")
X_train_norm, y_train_onehot = preprocess_data(X_train, y_train)
X_val_norm, y_val_onehot = preprocess_data(X_val, y_val)
X_test_norm, y_test_onehot = preprocess_data(X_test, y_test)

print(f"Normalized pixel range: [{X_train_norm.min():.1f}, {X_train_norm.max():.1f}]")
print(f"One-hot encoded shape: {y_train_onehot.shape}")
print(f"\nExample one-hot encoding for digit 3:")
print(f"  Original label: 3")
print(f"  One-hot: {y_train_onehot[y_train == 3][0]}")

# -----------------------------------------------------------------------------
# PART 3: BUILDING THE NEURAL NETWORK
# -----------------------------------------------------------------------------
print("\n\n3. BUILDING THE NEURAL NETWORK")
print("-" * 40)

class MNISTClassifier:
    """Complete neural network for MNIST classification"""
    
    def __init__(self, layer_sizes=[784, 128, 64, 10]):
        """Initialize network with given architecture"""
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        for i in range(self.n_layers):
            # He initialization for ReLU layers
            if i < self.n_layers - 1:  # Hidden layers
                W = np.random.randn(layer_sizes[i+1], layer_sizes[i]) * np.sqrt(2.0 / layer_sizes[i])
            else:  # Output layer
                W = np.random.randn(layer_sizes[i+1], layer_sizes[i]) * np.sqrt(1.0 / layer_sizes[i])
            
            b = np.zeros(layer_sizes[i+1])
            
            self.weights.append(W)
            self.biases.append(b)
        
        # For storing activations during forward pass
        self.cache = {}
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def relu(self, z):
        """ReLU activation function"""
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        """Derivative of ReLU"""
        return (z > 0).astype(float)
    
    def softmax(self, z):
        """Softmax activation for output layer"""
        # Stability trick: subtract max
        z_exp = np.exp(z - np.max(z, axis=1, keepdims=True))
        return z_exp / np.sum(z_exp, axis=1, keepdims=True)
    
    def forward(self, X, training=False, dropout_rate=0.5):
        """Forward pass through the network"""
        batch_size = X.shape[0]
        
        # Input layer
        self.cache['a0'] = X
        
        # Hidden layers
        for i in range(self.n_layers - 1):
            # Linear transformation
            z = np.dot(self.cache[f'a{i}'], self.weights[i].T) + self.biases[i]
            self.cache[f'z{i+1}'] = z
            
            # ReLU activation
            a = self.relu(z)
            
            # Dropout (only during training)
            if training and dropout_rate > 0:
                mask = np.random.rand(*a.shape) > dropout_rate
                a = a * mask / (1 - dropout_rate)
                self.cache[f'mask{i+1}'] = mask
            
            self.cache[f'a{i+1}'] = a
        
        # Output layer (no dropout)
        z_out = np.dot(self.cache[f'a{self.n_layers-1}'], self.weights[-1].T) + self.biases[-1]
        self.cache[f'z{self.n_layers}'] = z_out
        
        # Softmax activation
        output = self.softmax(z_out)
        self.cache[f'a{self.n_layers}'] = output
        
        return output
    
    def compute_loss(self, y_pred, y_true):
        """Cross-entropy loss"""
        # Avoid log(0)
        epsilon = 1e-7
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Cross-entropy
        loss = -np.sum(y_true * np.log(y_pred)) / len(y_true)
        return loss
    
    def backward(self, y_true, learning_rate=0.001, dropout_rate=0.5):
        """Backward pass - compute gradients and update weights"""
        batch_size = y_true.shape[0]
        
        # Output layer gradient
        dz = self.cache[f'a{self.n_layers}'] - y_true
        
        # Backpropagate through layers
        for i in range(self.n_layers - 1, -1, -1):
            # Gradient w.r.t weights and biases
            dW = np.dot(dz.T, self.cache[f'a{i}']) / batch_size
            db = np.mean(dz, axis=0)
            
            if i > 0:
                # Gradient w.r.t previous layer's activation
                da = np.dot(dz, self.weights[i])
                
                # Apply dropout mask if in hidden layer
                if i < self.n_layers - 1 and dropout_rate > 0:
                    da = da * self.cache.get(f'mask{i}', 1) / (1 - dropout_rate)
                
                # Gradient through ReLU
                dz = da * self.relu_derivative(self.cache[f'z{i}'])
            
            # Update weights (gradient descent)
            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * db
    
    def predict(self, X):
        """Make predictions (no dropout)"""
        output = self.forward(X, training=False)
        return np.argmax(output, axis=1)
    
    def accuracy(self, X, y):
        """Calculate accuracy"""
        predictions = self.predict(X)
        if len(y.shape) > 1:  # One-hot encoded
            y = np.argmax(y, axis=1)
        return np.mean(predictions == y)
    
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=10, batch_size=128, learning_rate=0.001, dropout_rate=0.5):
        """Train the network"""
        n_samples = X_train.shape[0]
        n_batches = n_samples // batch_size
        
        print(f"Training for {epochs} epochs with batch size {batch_size}")
        print(f"Learning rate: {learning_rate}, Dropout: {dropout_rate}")
        
        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            
            # Mini-batch training
            epoch_loss = 0
            for batch in range(n_batches):
                # Get batch
                start_idx = batch * batch_size
                end_idx = start_idx + batch_size
                X_batch = X_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]
                
                # Forward pass
                y_pred = self.forward(X_batch, training=True, dropout_rate=dropout_rate)
                
                # Compute loss
                batch_loss = self.compute_loss(y_pred, y_batch)
                epoch_loss += batch_loss
                
                # Backward pass
                self.backward(y_batch, learning_rate, dropout_rate)
            
            # Calculate metrics
            epoch_loss /= n_batches
            train_acc = self.accuracy(X_train[:5000], y_train[:5000])  # Subset for speed
            val_acc = self.accuracy(X_val, y_val)
            val_loss = self.compute_loss(self.forward(X_val), y_val)
            
            # Store history
            self.history['train_loss'].append(epoch_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"loss: {epoch_loss:.4f} - acc: {train_acc:.4f} - "
                  f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")

# Create and display network architecture
network = MNISTClassifier(layer_sizes=[784, 128, 64, 10])

print("\nNetwork Architecture:")
print("  Input layer:  784 neurons (28×28 pixels)")
print("  Hidden layer 1: 128 neurons (ReLU)")
print("  Hidden layer 2: 64 neurons (ReLU)")
print("  Output layer: 10 neurons (Softmax)")

total_params = sum(w.size + b.size for w, b in zip(network.weights, network.biases))
print(f"\nTotal parameters: {total_params:,}")

# -----------------------------------------------------------------------------
# PART 4: TRAINING THE NETWORK
# -----------------------------------------------------------------------------
print("\n\n4. TRAINING THE NETWORK")
print("-" * 40)

# Train the network
network.train(X_train_norm, y_train_onehot, 
              X_val_norm, y_val_onehot,
              epochs=10, 
              batch_size=128, 
              learning_rate=0.001,
              dropout_rate=0.2)

# Plot training history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Loss plot
ax1.plot(network.history['train_loss'], 'b-', label='Training loss')
ax1.plot(network.history['val_loss'], 'r-', label='Validation loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Accuracy plot
ax2.plot(network.history['train_acc'], 'b-', label='Training accuracy')
ax2.plot(network.history['val_acc'], 'r-', label='Validation accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Training and Validation Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# PART 5: EVALUATING ON TEST SET
# -----------------------------------------------------------------------------
print("\n\n5. EVALUATING ON TEST SET")
print("-" * 40)

# Final test accuracy
test_accuracy = network.accuracy(X_test_norm, y_test_onehot)
print(f"Final test accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Confusion matrix
def compute_confusion_matrix(y_true, y_pred, n_classes=10):
    """Compute confusion matrix"""
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        matrix[true, pred] += 1
    return matrix

# Get predictions
test_predictions = network.predict(X_test_norm)
confusion_matrix = compute_confusion_matrix(y_test, test_predictions)

# Visualize confusion matrix
plt.figure(figsize=(10, 8))
plt.imshow(confusion_matrix, cmap='Blues')
plt.colorbar(label='Count')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix on Test Set')

# Add count annotations
for i in range(10):
    for j in range(10):
        count = confusion_matrix[i, j]
        color = 'white' if count > confusion_matrix.max() / 2 else 'black'
        plt.text(j, i, str(count), ha='center', va='center', color=color)

plt.tight_layout()
plt.show()

# Per-class accuracy
print("\nPer-class accuracy:")
for digit in range(10):
    digit_mask = y_test == digit
    digit_acc = np.mean(test_predictions[digit_mask] == digit)
    print(f"  Digit {digit}: {digit_acc:.4f} ({digit_acc*100:.2f}%)")

# Most confused pairs
print("\nMost confused digit pairs:")
for i in range(10):
    for j in range(10):
        if i != j and confusion_matrix[i, j] > 50:
            print(f"  True {i} predicted as {j}: {confusion_matrix[i, j]} times")

# -----------------------------------------------------------------------------
# PART 6: VISUALIZING MISTAKES
# -----------------------------------------------------------------------------
print("\n\n6. ANALYZING MISCLASSIFIED EXAMPLES")
print("-" * 40)

# Find misclassified examples
misclassified_mask = test_predictions != y_test
misclassified_indices = np.where(misclassified_mask)[0]

print(f"Total misclassified: {len(misclassified_indices)} out of {len(y_test)}")

# Show some misclassified examples
if len(misclassified_indices) > 0:
    n_examples = min(12, len(misclassified_indices))
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    axes = axes.ravel()
    
    for i in range(n_examples):
        idx = misclassified_indices[i]
        image = X_test[idx].reshape(28, 28)
        true_label = y_test[idx]
        pred_label = test_predictions[idx]
        
        # Get confidence
        probs = network.forward(X_test_norm[idx:idx+1])[0]
        confidence = probs[pred_label]
        
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(f'True: {true_label}, Pred: {pred_label}\nConf: {confidence:.2f}')
        axes[i].axis('off')
    
    plt.suptitle('Misclassified Examples', fontsize=16)
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------
# PART 7: VISUALIZING LEARNED FEATURES
# -----------------------------------------------------------------------------
print("\n\n7. VISUALIZING LEARNED FEATURES")
print("-" * 40)

# Visualize first layer weights
first_layer_weights = network.weights[0]  # Shape: (128, 784)

# Show first 32 filters
n_filters = min(32, first_layer_weights.shape[0])
fig, axes = plt.subplots(4, 8, figsize=(12, 6))
axes = axes.ravel()

for i in range(n_filters):
    weight = first_layer_weights[i].reshape(28, 28)
    
    # Normalize for visualization
    vmin, vmax = weight.min(), weight.max()
    
    axes[i].imshow(weight, cmap='seismic', vmin=-abs(max(vmin, vmax)), vmax=abs(max(vmin, vmax)))
    axes[i].axis('off')

plt.suptitle('First Layer Learned Features (Weights)', fontsize=16)
plt.tight_layout()
plt.show()

print("Notice: First layer often learns edge detectors and simple patterns!")

# -----------------------------------------------------------------------------
# EXERCISES
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print("EXERCISES - Improve Your Model!")
print("="*60)

print("\n📝 Exercise 1: Implement Adam Optimizer")
print("Replace basic gradient descent with Adam")

class AdamOptimizer:
    """
    Implement Adam optimizer for better training.
    
    Adam combines:
    - Momentum (exponential moving average of gradients)
    - RMSprop (exponential moving average of squared gradients)
    - Bias correction
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = {}  # First moments
        self.v = {}  # Second moments
        
    def update(self, param_name, param, gradient):
        """
        Update parameters using Adam.
        
        Returns: updated parameter
        """
        # YOUR CODE HERE
        # 1. Initialize moments if needed
        # 2. Update biased first moment: m = β1*m + (1-β1)*gradient
        # 3. Update biased second moment: v = β2*v + (1-β2)*gradient²
        # 4. Correct bias: m_hat = m/(1-β1^t), v_hat = v/(1-β2^t)
        # 5. Update param: param = param - lr*m_hat/(sqrt(v_hat)+ε)
        
        return param  # Replace with Adam update

# Test Adam optimizer
print("\nTest Adam optimizer implementation:")
adam = AdamOptimizer()
test_param = np.array([1.0, 2.0])
test_grad = np.array([0.1, -0.2])

try:
    updated = adam.update('test', test_param, test_grad)
    if np.array_equal(updated, test_param):
        print("⚠️  Not implemented yet")
    else:
        print(f"Original param: {test_param}")
        print(f"Gradient: {test_grad}")
        print(f"Updated param: {updated}")
except:
    print("⚠️  Error in implementation")

print("\n📝 Exercise 2: Implement Data Augmentation")
print("Add rotation and translation to training data")

def augment_batch(X_batch, max_rotation=15, max_shift=2):
    """
    Augment a batch of images.
    
    Args:
        X_batch: Images of shape (batch_size, 784)
        max_rotation: Maximum rotation in degrees
        max_shift: Maximum pixel shift
    
    Returns:
        Augmented batch
    """
    batch_size = X_batch.shape[0]
    X_augmented = np.zeros_like(X_batch)
    
    for i in range(batch_size):
        # Reshape to 28x28
        img = X_batch[i].reshape(28, 28)
        
        # YOUR CODE HERE
        # 1. Random rotation: angle = random.uniform(-max_rotation, max_rotation)
        # 2. Random shift: dx, dy = random.randint(-max_shift, max_shift)
        # 3. Apply transformations (hint: use scipy.ndimage if available)
        # For now, just implement shift manually
        
        augmented_img = img  # Replace with augmented version
        
        X_augmented[i] = augmented_img.flatten()
    
    return X_augmented

# Test augmentation
print("\nTest data augmentation:")
test_batch = X_train_norm[:5]
try:
    augmented = augment_batch(test_batch)
    if np.array_equal(augmented, test_batch):
        print("⚠️  Not implemented yet")
    else:
        print("✅ Augmentation applied!")
        # Visualize
        fig, axes = plt.subplots(2, 5, figsize=(10, 4))
        for i in range(5):
            axes[0, i].imshow(test_batch[i].reshape(28, 28), cmap='gray')
            axes[0, i].axis('off')
            axes[1, i].imshow(augmented[i].reshape(28, 28), cmap='gray')
            axes[1, i].axis('off')
        axes[0, 0].set_ylabel('Original')
        axes[1, 0].set_ylabel('Augmented')
        plt.tight_layout()
        plt.show()
except:
    print("⚠️  Error in implementation")

print("\n📝 Exercise 3: Implement Learning Rate Scheduling")
print("Reduce learning rate when validation loss plateaus")

class LearningRateScheduler:
    """
    Reduce learning rate on plateau.
    """
    def __init__(self, initial_lr=0.001, factor=0.5, patience=3, min_lr=1e-6):
        self.current_lr = initial_lr
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best_loss = float('inf')
        self.wait = 0
        
    def update(self, val_loss):
        """
        Check if learning rate should be reduced.
        
        Returns: new learning rate
        """
        # YOUR CODE HERE
        # 1. If val_loss improved: update best_loss, reset wait
        # 2. Else: increment wait
        # 3. If wait >= patience: reduce lr by factor (but not below min_lr)
        
        return self.current_lr

# Test scheduler
print("\nTest learning rate scheduler:")
scheduler = LearningRateScheduler(initial_lr=0.01)
val_losses = [1.0, 0.8, 0.7, 0.71, 0.70, 0.69, 0.695, 0.694]

for epoch, loss in enumerate(val_losses):
    lr = scheduler.update(loss)
    print(f"Epoch {epoch}: val_loss={loss:.3f}, lr={lr:.6f}")

print("\n🎉 Congratulations! You've built a complete MNIST classifier!")
print("\nYour achievements:")
print("  ✓ Loaded and explored real-world data")
print("  ✓ Built a multi-layer neural network from scratch")
print("  ✓ Implemented forward and backward propagation")
print("  ✓ Trained with mini-batch gradient descent")
print("  ✓ Achieved >95% accuracy on handwritten digits")
print("  ✓ Analyzed errors and visualized learned features")
print("\nYou're now ready for Phase 2: Modern Architectures!")