"""
Lesson 12: Convolutional Neural Networks
Teaching computers to see like humans do
"""

import numpy as np
import matplotlib.pyplot as plt

# Check PyTorch availability
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision
    import torchvision.transforms as transforms
    PYTORCH_AVAILABLE = True
    print("✅ PyTorch is installed!")
except ImportError:
    PYTORCH_AVAILABLE = False
    print("⚠️  PyTorch not found. We'll demonstrate concepts with NumPy")

print("="*60)
print("LESSON 12: CONVOLUTIONAL NEURAL NETWORKS")
print("="*60)

# -----------------------------------------------------------------------------
# PART 1: UNDERSTANDING CONVOLUTION
# -----------------------------------------------------------------------------
print("\n1. THE CONVOLUTION OPERATION")
print("-" * 40)

# Simple convolution example with NumPy
def simple_convolution(image, kernel):
    """Perform 2D convolution with valid padding"""
    i_height, i_width = image.shape
    k_height, k_width = kernel.shape
    
    # Output dimensions
    o_height = i_height - k_height + 1
    o_width = i_width - k_width + 1
    
    output = np.zeros((o_height, o_width))
    
    # Slide the kernel
    for i in range(o_height):
        for j in range(o_width):
            # Extract patch
            patch = image[i:i+k_height, j:j+k_width]
            # Element-wise multiply and sum
            output[i, j] = np.sum(patch * kernel)
    
    return output

# Create a simple image
image = np.array([
    [1, 2, 3, 0, 1],
    [4, 5, 6, 1, 2],
    [7, 8, 9, 2, 3],
    [1, 2, 3, 3, 4],
    [0, 1, 0, 4, 5]
])

# Edge detection kernel
kernel = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])

print("Original image (5x5):")
print(image)
print("\nKernel (3x3) - Vertical edge detector:")
print(kernel)

# Apply convolution
result = simple_convolution(image, kernel)
print("\nConvolution result (3x3):")
print(result)
print("\nNotice: High values where vertical edges exist!")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(kernel, cmap='RdBu')
axes[1].set_title('Kernel (Edge Detector)')
axes[1].axis('off')

axes[2].imshow(result, cmap='gray')
axes[2].set_title('Convolution Result')
axes[2].axis('off')

plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# PART 2: MULTIPLE FILTERS
# -----------------------------------------------------------------------------
print("\n\n2. MULTIPLE FILTERS - DETECTING DIFFERENT FEATURES")
print("-" * 40)

# Different types of kernels
kernels = {
    'Vertical edges': np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
    'Horizontal edges': np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
    'Blur': np.ones((3, 3)) / 9,
    'Sharpen': np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
}

# Create a more interesting test image
test_image = np.zeros((10, 10))
# Add vertical line
test_image[:, 4:6] = 1
# Add horizontal line
test_image[4:6, :] = 1
# Add some noise
test_image += np.random.normal(0, 0.1, (10, 10))

# Apply all kernels
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

axes[0].imshow(test_image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

for idx, (name, kernel) in enumerate(kernels.items(), 1):
    result = simple_convolution(test_image, kernel)
    axes[idx].imshow(result, cmap='gray')
    axes[idx].set_title(f'{name}')
    axes[idx].axis('off')

# Remove empty subplot
axes[-1].axis('off')

plt.suptitle('Different Filters Detect Different Features')
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# PART 3: PADDING AND STRIDE
# -----------------------------------------------------------------------------
print("\n\n3. PADDING AND STRIDE - CONTROLLING OUTPUT SIZE")
print("-" * 40)

def convolution_with_padding_stride(image, kernel, padding=0, stride=1):
    """Convolution with padding and stride support"""
    i_height, i_width = image.shape
    k_height, k_width = kernel.shape
    
    # Add padding
    if padding > 0:
        image = np.pad(image, padding, mode='constant', constant_values=0)
    
    # Calculate output dimensions
    o_height = (image.shape[0] - k_height) // stride + 1
    o_width = (image.shape[1] - k_width) // stride + 1
    
    output = np.zeros((o_height, o_width))
    
    # Convolution with stride
    for i in range(o_height):
        for j in range(o_width):
            # Extract patch with stride
            h_start = i * stride
            w_start = j * stride
            patch = image[h_start:h_start+k_height, w_start:w_start+k_width]
            output[i, j] = np.sum(patch * kernel)
    
    return output

# Test different padding and stride
test_cases = [
    (0, 1, 'No padding, stride 1'),
    (1, 1, 'Padding 1, stride 1 (same size)'),
    (0, 2, 'No padding, stride 2 (smaller)'),
    (2, 1, 'Padding 2, stride 1 (larger)')
]

print(f"Original image size: {test_image.shape}")
print(f"Kernel size: {kernels['Vertical edges'].shape}")

for padding, stride, description in test_cases:
    result = convolution_with_padding_stride(
        test_image, kernels['Vertical edges'], padding, stride
    )
    print(f"\n{description}:")
    print(f"  Output size: {result.shape}")
    print(f"  Formula: ({test_image.shape[0]} - 3 + 2×{padding}) / {stride} + 1 = {result.shape[0]}")

# -----------------------------------------------------------------------------
# PART 4: POOLING LAYERS
# -----------------------------------------------------------------------------
print("\n\n4. POOLING - REDUCING SPATIAL DIMENSIONS")
print("-" * 40)

def max_pooling(image, pool_size=2, stride=None):
    """Simple max pooling implementation"""
    if stride is None:
        stride = pool_size
    
    h, w = image.shape
    o_h = (h - pool_size) // stride + 1
    o_w = (w - pool_size) // stride + 1
    
    output = np.zeros((o_h, o_w))
    
    for i in range(o_h):
        for j in range(o_w):
            h_start = i * stride
            w_start = j * stride
            patch = image[h_start:h_start+pool_size, w_start:w_start+pool_size]
            output[i, j] = np.max(patch)
    
    return output

# Create example for pooling
pool_example = np.array([
    [1, 2, 5, 6],
    [3, 4, 7, 8],
    [9, 10, 13, 14],
    [11, 12, 15, 16]
])

print("Original 4x4 image:")
print(pool_example)

pooled = max_pooling(pool_example, pool_size=2)
print("\nAfter 2x2 max pooling:")
print(pooled)
print("\nNotice: Takes maximum value in each 2x2 region")

# Visualize pooling effect
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

im1 = ax1.imshow(pool_example, cmap='viridis')
ax1.set_title('Original 4x4')
ax1.set_xticks(np.arange(4))
ax1.set_yticks(np.arange(4))
ax1.grid(True, alpha=0.3)

# Draw pooling regions
for i in range(0, 4, 2):
    for j in range(0, 4, 2):
        rect = plt.Rectangle((j-0.5, i-0.5), 2, 2, 
                           fill=False, edgecolor='red', linewidth=2)
        ax1.add_patch(rect)

im2 = ax2.imshow(pooled, cmap='viridis')
ax2.set_title('After 2x2 Max Pooling')
ax2.set_xticks(np.arange(2))
ax2.set_yticks(np.arange(2))

plt.colorbar(im1, ax=ax1)
plt.colorbar(im2, ax=ax2)
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# PART 5: BUILDING A CNN IN PYTORCH
# -----------------------------------------------------------------------------
print("\n\n5. BUILDING A COMPLETE CNN")
print("-" * 40)

if PYTORCH_AVAILABLE:
    class SimpleCNN(nn.Module):
        """A simple CNN for digit classification"""
        def __init__(self):
            super(SimpleCNN, self).__init__()
            # First conv block
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            
            # Second conv block  
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            
            # Fully connected layers
            # After 2 pooling layers: 28x28 -> 14x14 -> 7x7
            self.fc1 = nn.Linear(64 * 7 * 7, 128)
            self.fc2 = nn.Linear(128, 10)
            
            self.dropout = nn.Dropout(0.25)
            
        def forward(self, x):
            # First block: conv -> relu -> conv -> relu -> pool
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            
            # Second block: conv -> relu -> pool
            x = F.relu(self.conv3(x))
            x = self.pool(x)
            
            # Flatten for fully connected layers
            x = x.view(-1, 64 * 7 * 7)
            
            # Fully connected layers
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            
            return x
    
    # Create the model
    model = SimpleCNN()
    print("CNN Architecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    # Show parameter distribution
    print("\nParameters per layer:")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.shape} = {param.numel():,} parameters")
    
    # Trace through the network
    print("\n\nTracing data flow through the network:")
    x = torch.randn(1, 1, 28, 28)  # Batch of 1, 1 channel, 28x28 image
    print(f"Input shape: {x.shape}")
    
    # Manually trace through each layer
    x1 = F.relu(model.conv1(x))
    print(f"After conv1 + ReLU: {x1.shape}")
    
    x2 = F.relu(model.conv2(x1))
    print(f"After conv2 + ReLU: {x2.shape}")
    
    x3 = model.pool(x2)
    print(f"After first pooling: {x3.shape}")
    
    x4 = F.relu(model.conv3(x3))
    print(f"After conv3 + ReLU: {x4.shape}")
    
    x5 = model.pool(x4)
    print(f"After second pooling: {x5.shape}")
    
    x6 = x5.view(-1, 64 * 7 * 7)
    print(f"After flattening: {x6.shape}")
    
    output = model(torch.randn(1, 1, 28, 28))
    print(f"Final output: {output.shape} (10 classes)")

# -----------------------------------------------------------------------------
# PART 6: VISUALIZING FEATURE MAPS
# -----------------------------------------------------------------------------
print("\n\n6. WHAT DOES A CNN SEE?")
print("-" * 40)

# Create a simple pattern
pattern = np.zeros((28, 28))
# Add some geometric shapes
pattern[5:10, 5:23] = 1  # Horizontal bar
pattern[5:23, 10:15] = 1  # Vertical bar
pattern[15:20, 5:23] = 1  # Another horizontal bar

if PYTORCH_AVAILABLE:
    # Convert to tensor
    input_tensor = torch.tensor(pattern, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # Get feature maps from first conv layer
    model.eval()
    with torch.no_grad():
        features = F.relu(model.conv1(input_tensor))
    
    # Plot original and first 8 feature maps
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.flatten()
    
    # Original image
    axes[0].imshow(pattern, cmap='gray')
    axes[0].set_title('Original Pattern')
    axes[0].axis('off')
    
    # Feature maps
    for i in range(1, 9):
        if i-1 < features.shape[1]:
            feature_map = features[0, i-1].numpy()
            axes[i].imshow(feature_map, cmap='gray')
            axes[i].set_title(f'Feature Map {i}')
            axes[i].axis('off')
    
    plt.suptitle('What the First Conv Layer Sees')
    plt.tight_layout()
    plt.show()
    
    print("Each feature map detects different patterns!")
    print("Some might detect edges, others detect corners or textures.")
else:
    # NumPy visualization of concept
    plt.figure(figsize=(8, 4))
    plt.subplot(121)
    plt.imshow(pattern, cmap='gray')
    plt.title('Input Pattern')
    plt.axis('off')
    
    # Simulate edge detection
    edges = simple_convolution(pattern, kernels['Vertical edges'])
    plt.subplot(122)
    plt.imshow(edges, cmap='gray')
    plt.title('Edge Detection Result')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------
# PART 7: RECEPTIVE FIELDS
# -----------------------------------------------------------------------------
print("\n\n7. RECEPTIVE FIELDS - HOW MUCH EACH NEURON SEES")
print("-" * 40)

def calculate_receptive_field(layers):
    """Calculate receptive field size through CNN layers"""
    rf = 1  # Start with 1 pixel
    
    for layer_type, kernel_size, stride in layers:
        if layer_type == 'conv':
            rf = rf + (kernel_size - 1)
        elif layer_type == 'pool':
            rf = rf + (kernel_size - 1) * stride
    
    return rf

# Example architecture
layers = [
    ('conv', 3, 1),  # 3x3 conv, stride 1
    ('conv', 3, 1),  # 3x3 conv, stride 1
    ('pool', 2, 2),  # 2x2 pool, stride 2
    ('conv', 3, 1),  # 3x3 conv, stride 1
]

print("Layer-by-layer receptive field growth:")
print("Starting with 1x1 receptive field\n")

current_rf = 1
for i, (layer_type, kernel_size, stride) in enumerate(layers):
    if layer_type == 'conv':
        current_rf = current_rf + (kernel_size - 1)
        print(f"After conv layer {i+1} (3x3): {current_rf}x{current_rf} receptive field")
    elif layer_type == 'pool':
        # Pooling increases receptive field more dramatically
        print(f"After pooling layer: receptive field doubles in effect")

print(f"\nFinal receptive field: Each neuron sees a {current_rf}x{current_rf} region")
print("This is how deep networks see larger contexts!")

# -----------------------------------------------------------------------------
# PART 8: FAMOUS CNN ARCHITECTURES
# -----------------------------------------------------------------------------
print("\n\n8. EVOLUTION OF CNN ARCHITECTURES")
print("-" * 40)

architectures = {
    'LeNet-5 (1998)': {
        'layers': 7,
        'parameters': '60K',
        'innovation': 'First successful CNN'
    },
    'AlexNet (2012)': {
        'layers': 8,
        'parameters': '60M',
        'innovation': 'ReLU, Dropout, GPU training'
    },
    'VGG-16 (2014)': {
        'layers': 16,
        'parameters': '138M',
        'innovation': 'Only 3x3 convolutions'
    },
    'ResNet-50 (2015)': {
        'layers': 50,
        'parameters': '25M',
        'innovation': 'Skip connections'
    },
    'EfficientNet (2019)': {
        'layers': 'Varies',
        'parameters': '5-66M',
        'innovation': 'Compound scaling'
    }
}

print("Famous CNN Architectures:\n")
for name, info in architectures.items():
    print(f"{name}:")
    print(f"  Layers: {info['layers']}")
    print(f"  Parameters: {info['parameters']}")
    print(f"  Key innovation: {info['innovation']}")
    print()

# -----------------------------------------------------------------------------
# EXERCISES
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print("EXERCISES - Master CNNs!")
print("="*60)

print("\n📝 Exercise 1: Implement Custom Convolution")
print("Complete the convolution with different padding modes")

def custom_convolution(image, kernel, padding='valid'):
    """
    Implement convolution with different padding modes:
    - 'valid': no padding
    - 'same': pad to keep same size
    - 'full': maximum padding
    """
    i_h, i_w = image.shape
    k_h, k_w = kernel.shape
    
    # YOUR CODE HERE
    if padding == 'valid':
        pad = 0
    elif padding == 'same':
        pad = None  # Calculate padding for same size
    elif padding == 'full':
        pad = None  # Calculate maximum padding
    
    # Apply padding and convolution
    # YOUR CODE HERE
    
    return image  # Replace with actual result

# Test your implementation
test_img = np.random.rand(5, 5)
test_kernel = np.ones((3, 3)) / 9

try:
    result = custom_convolution(test_img, test_kernel, 'same')
    if result.shape == test_img.shape:
        print("✅ 'same' padding working correctly!")
    else:
        print(f"❌ Expected shape {test_img.shape}, got {result.shape}")
except:
    print("⚠️  Not implemented yet")

print("\n📝 Exercise 2: Build a Mini CNN")
print("Create a CNN for 32x32 RGB images")

if PYTORCH_AVAILABLE:
    class MyCNN(nn.Module):
        """
        Build a CNN with:
        - Input: 32x32x3 (RGB images)
        - Conv layers: at least 3
        - Use batch normalization
        - Output: 10 classes
        """
        def __init__(self):
            super(MyCNN, self).__init__()
            # YOUR CODE HERE
            # Define your layers
            pass
            
        def forward(self, x):
            # YOUR CODE HERE
            # Define forward pass
            return x
    
    # Test your network
    try:
        my_cnn = MyCNN()
        test_input = torch.randn(1, 3, 32, 32)
        output = my_cnn(test_input)
        
        if output.shape == torch.Size([1, 10]):
            print("✅ Network output shape is correct!")
            params = sum(p.numel() for p in my_cnn.parameters())
            print(f"Total parameters: {params:,}")
        else:
            print(f"❌ Expected output shape [1, 10], got {output.shape}")
    except:
        print("⚠️  Network not implemented correctly")

print("\n📝 Exercise 3: Calculate Output Dimensions")
print("Given network architecture, calculate final size")

def calculate_output_size(input_size, architecture):
    """
    Calculate output size after series of conv/pool layers
    
    architecture: list of (type, kernel_size, stride, padding) tuples
    """
    h, w = input_size
    
    for layer_type, kernel_size, stride, padding in architecture:
        # YOUR CODE HERE
        # Update h and w based on layer type
        pass
    
    return (h, w)

# Test architecture
test_arch = [
    ('conv', 5, 1, 2),   # 5x5 kernel, stride 1, padding 2
    ('pool', 2, 2, 0),   # 2x2 pool, stride 2
    ('conv', 3, 1, 1),   # 3x3 kernel, stride 1, padding 1
    ('pool', 2, 2, 0),   # 2x2 pool, stride 2
]

try:
    final_size = calculate_output_size((28, 28), test_arch)
    print(f"Starting with 28x28 image")
    print(f"Final size: {final_size}")
    # Should be (7, 7) for this architecture
except:
    print("⚠️  Not implemented yet")

print("\n🎉 Congratulations! You understand how computers see!")
print("\nKey achievements:")
print("  ✓ Mastered the convolution operation")
print("  ✓ Understand filters, padding, and stride")
print("  ✓ Know how pooling reduces dimensions")
print("  ✓ Can build complete CNNs")
print("  ✓ Understand receptive fields")
print("\nNext: Attention mechanisms - teaching networks to focus!")