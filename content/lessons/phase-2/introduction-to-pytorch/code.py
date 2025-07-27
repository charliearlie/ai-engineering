"""
Lesson 11: Introduction to PyTorch
Your gateway to modern deep learning
"""

import numpy as np
import matplotlib.pyplot as plt

# Check if PyTorch is available
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    PYTORCH_AVAILABLE = True
    print("✅ PyTorch is installed!")
    print(f"PyTorch version: {torch.__version__}")
except ImportError:
    PYTORCH_AVAILABLE = False
    print("❌ PyTorch not found. Install with: pip install torch")
    print("This lesson will show comparisons with NumPy")

print("="*60)
print("LESSON 11: INTRODUCTION TO PYTORCH")
print("="*60)

# -----------------------------------------------------------------------------
# PART 1: FROM NUMPY TO PYTORCH
# -----------------------------------------------------------------------------
print("\n1. NUMPY vs PYTORCH - SPOT THE DIFFERENCE!")
print("-" * 40)

# NumPy way
np_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
print(f"NumPy array: {np_array}")
print(f"Type: {type(np_array)}")

if PYTORCH_AVAILABLE:
    # PyTorch way
    torch_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    print(f"\nPyTorch tensor: {torch_tensor}")
    print(f"Type: {type(torch_tensor)}")
    
    # They look similar, but PyTorch has superpowers!
    print("\nSuperpowers PyTorch tensors have:")
    print("  ✓ Can live on GPU")
    print("  ✓ Track computation history")
    print("  ✓ Compute gradients automatically")

# -----------------------------------------------------------------------------
# PART 2: TENSOR OPERATIONS
# -----------------------------------------------------------------------------
print("\n\n2. TENSOR OPERATIONS - JUST LIKE NUMPY!")
print("-" * 40)

if PYTORCH_AVAILABLE:
    # Creating tensors
    print("Creating tensors:")
    
    # Different ways to create
    x = torch.zeros(2, 3)
    print(f"Zeros:\n{x}")
    
    x = torch.ones(2, 3)
    print(f"\nOnes:\n{x}")
    
    x = torch.randn(2, 3)  # Random normal
    print(f"\nRandom normal:\n{x}")
    
    # From Python lists
    x = torch.tensor([[1, 2], [3, 4]])
    print(f"\nFrom list:\n{x}")
    
    # Operations
    print("\nBasic operations:")
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])
    
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"a + b = {a + b}")
    print(f"a * b = {a * b}")  # Element-wise
    print(f"a.dot(b) = {a.dot(b)}")  # Dot product
    
    # Reshaping
    x = torch.arange(12)
    print(f"\nOriginal shape: {x.shape}")
    x_reshaped = x.reshape(3, 4)
    print(f"Reshaped to 3x4:\n{x_reshaped}")

# -----------------------------------------------------------------------------
# PART 3: THE MAGIC OF AUTOGRAD
# -----------------------------------------------------------------------------
print("\n\n3. AUTOGRAD - AUTOMATIC DIFFERENTIATION!")
print("-" * 40)

if PYTORCH_AVAILABLE:
    print("Let's compute gradients automatically!")
    
    # Simple example: y = x^2
    x = torch.tensor(3.0, requires_grad=True)
    print(f"\nx = {x}")
    
    y = x ** 2
    print(f"y = x² = {y}")
    
    # Compute gradient
    y.backward()
    print(f"dy/dx = {x.grad} (should be 2*x = 6)")
    
    # More complex example
    print("\nMore complex: z = (x + 2)² + 3x")
    x = torch.tensor(1.0, requires_grad=True)
    z = (x + 2)**2 + 3*x
    print(f"x = {x}")
    print(f"z = {z}")
    
    z.backward()
    print(f"dz/dx = {x.grad}")
    print("(Manual calculation: 2(x+2) + 3 = 2(3) + 3 = 9)")
    
    # Gradient with multiple variables
    print("\nMultiple variables:")
    x = torch.tensor(2.0, requires_grad=True)
    y = torch.tensor(3.0, requires_grad=True)
    
    z = x**2 + y**2 + x*y
    print(f"x = {x}, y = {y}")
    print(f"z = x² + y² + xy = {z}")
    
    z.backward()
    print(f"∂z/∂x = {x.grad} (should be 2x + y = 7)")
    print(f"∂z/∂y = {y.grad} (should be 2y + x = 8)")

# -----------------------------------------------------------------------------
# PART 4: COMPUTATIONAL GRAPHS
# -----------------------------------------------------------------------------
print("\n\n4. COMPUTATIONAL GRAPHS - TRACKING OPERATIONS")
print("-" * 40)

if PYTORCH_AVAILABLE:
    # Build a computation graph
    x = torch.tensor(2.0, requires_grad=True)
    print(f"x = {x}")
    print(f"x.grad_fn = {x.grad_fn} (None - it's a leaf)")
    
    y = x + 3
    print(f"\ny = x + 3 = {y}")
    print(f"y.grad_fn = {y.grad_fn}")
    
    z = y * y
    print(f"\nz = y * y = {z}")
    print(f"z.grad_fn = {z.grad_fn}")
    
    # The graph tracks everything!
    print("\nPyTorch built this graph:")
    print("x (leaf) → [AddBackward] → y → [MulBackward] → z")
    
    # Compute gradients through the graph
    z.backward()
    print(f"\nAfter backward():")
    print(f"x.grad = {x.grad}")
    print("Chain rule: dz/dx = dz/dy * dy/dx = 2y * 1 = 2(5) = 10")

# -----------------------------------------------------------------------------
# PART 5: BUILDING NEURAL NETWORKS
# -----------------------------------------------------------------------------
print("\n\n5. NEURAL NETWORKS THE PYTORCH WAY")
print("-" * 40)

if PYTORCH_AVAILABLE:
    # Define a simple network
    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(2, 4)  # 2 inputs → 4 hidden
            self.fc2 = nn.Linear(4, 1)  # 4 hidden → 1 output
            
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    # Create the network
    model = TinyNet()
    print("Created a tiny neural network!")
    print(f"\nModel structure:\n{model}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params}")
    
    # Show parameters
    print("\nParameters:")
    for name, param in model.named_parameters():
        print(f"  {name}: shape {param.shape}")
    
    # Forward pass
    x = torch.randn(1, 2)  # 1 sample, 2 features
    output = model(x)
    print(f"\nInput: {x}")
    print(f"Output: {output}")

# -----------------------------------------------------------------------------
# PART 6: TRAINING A MODEL - XOR PROBLEM
# -----------------------------------------------------------------------------
print("\n\n6. TRAINING A REAL MODEL - SOLVING XOR")
print("-" * 40)

if PYTORCH_AVAILABLE:
    # XOR dataset
    X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
    
    print("XOR Truth Table:")
    print("Input  | Output")
    print("-------|-------")
    for i in range(len(X)):
        print(f"{X[i].numpy()} | {y[i].item()}")
    
    # Build model
    model = nn.Sequential(
        nn.Linear(2, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
        nn.Sigmoid()
    )
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    criterion = nn.BCELoss()
    
    print("\nTraining...")
    losses = []
    
    for epoch in range(500):
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch:3d}: Loss = {loss.item():.4f}")
    
    # Test the trained model
    print("\nTrained model predictions:")
    with torch.no_grad():
        predictions = model(X)
        for i in range(len(X)):
            print(f"Input: {X[i].numpy()} → Output: {predictions[i].item():.3f} (target: {y[i].item()})")
    
    # Plot training curve
    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('XOR Training Loss')
    plt.grid(True, alpha=0.3)
    plt.show()

# -----------------------------------------------------------------------------
# PART 7: MANUAL VS PYTORCH COMPARISON
# -----------------------------------------------------------------------------
print("\n\n7. MANUAL vs PYTORCH - SEE THE DIFFERENCE!")
print("-" * 40)

# Manual gradient calculation (NumPy)
print("Manual gradient calculation (NumPy):")
def manual_gradient():
    x = 3.0
    # y = x^2
    y = x ** 2
    # dy/dx = 2x
    grad = 2 * x
    return y, grad

y_manual, grad_manual = manual_gradient()
print(f"x = 3.0")
print(f"y = x² = {y_manual}")
print(f"dy/dx = 2x = {grad_manual}")

if PYTORCH_AVAILABLE:
    print("\nPyTorch automatic gradient:")
    x = torch.tensor(3.0, requires_grad=True)
    y = x ** 2
    y.backward()
    print(f"x = {x}")
    print(f"y = x² = {y}")
    print(f"dy/dx = {x.grad}")
    print("\nNo manual calculation needed! 🎉")

# -----------------------------------------------------------------------------
# PART 8: GPU ACCELERATION
# -----------------------------------------------------------------------------
print("\n\n8. GPU ACCELERATION")
print("-" * 40)

if PYTORCH_AVAILABLE:
    # Check GPU availability
    if torch.cuda.is_available():
        print("🚀 GPU is available!")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        
        # Moving to GPU
        x = torch.randn(1000, 1000)
        print(f"\nTensor on CPU: {x.device}")
        
        x_gpu = x.cuda()  # or x.to('cuda')
        print(f"Tensor on GPU: {x_gpu.device}")
        
        # Speed comparison
        import time
        
        # CPU timing
        start = time.time()
        for _ in range(100):
            _ = x @ x
        cpu_time = time.time() - start
        
        # GPU timing
        start = time.time()
        for _ in range(100):
            _ = x_gpu @ x_gpu
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        
        print(f"\nMatrix multiplication (1000x1000) 100 times:")
        print(f"CPU time: {cpu_time:.3f}s")
        print(f"GPU time: {gpu_time:.3f}s")
        print(f"Speedup: {cpu_time/gpu_time:.1f}x")
    else:
        print("No GPU available - using CPU")
        print("GPU acceleration available with CUDA-capable NVIDIA GPU")

# -----------------------------------------------------------------------------
# PART 9: COMMON PATTERNS AND TIPS
# -----------------------------------------------------------------------------
print("\n\n9. COMMON PATTERNS AND TIPS")
print("-" * 40)

if PYTORCH_AVAILABLE:
    print("1. Always zero gradients before backward():")
    print("   optimizer.zero_grad()  # Clear old gradients")
    print("   loss.backward()        # Compute new gradients")
    print("   optimizer.step()       # Update weights")
    
    print("\n2. Use no_grad() for evaluation:")
    model = nn.Linear(10, 1)
    with torch.no_grad():
        # No gradients computed here - saves memory
        output = model(torch.randn(5, 10))
    
    print("\n3. Check tensor shapes constantly:")
    x = torch.randn(32, 10)  # batch_size=32, features=10
    print(f"   x.shape = {x.shape}")
    print(f"   x.size() = {x.size()}  # Same thing")
    
    print("\n4. Move model and data to same device:")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = model.to(device)
    # data = data.to(device)
    
    print("\n5. Save and load models:")
    print("   torch.save(model.state_dict(), 'model.pth')")
    print("   model.load_state_dict(torch.load('model.pth'))")

# -----------------------------------------------------------------------------
# EXERCISES
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print("EXERCISES - Start Your PyTorch Journey!")
print("="*60)

print("\n📝 Exercise 1: Tensor Basics")
print("Create and manipulate tensors")

if PYTORCH_AVAILABLE:
    def tensor_exercise():
        """
        Create the following tensors:
        1. A 3x3 identity matrix
        2. A random tensor of shape (2, 4) with values between 0 and 1
        3. Compute the sum of all elements in tensor 2
        """
        # YOUR CODE HERE
        identity = None  # Create 3x3 identity matrix
        random_tensor = None  # Create 2x4 random tensor (0 to 1)
        tensor_sum = None  # Sum all elements
        
        return identity, random_tensor, tensor_sum
    
    # Test your implementation
    try:
        identity, random_tensor, tensor_sum = tensor_exercise()
        if identity is None:
            print("⚠️  Not implemented yet")
        else:
            print(f"Identity matrix:\n{identity}")
            print(f"Random tensor:\n{random_tensor}")
            print(f"Sum: {tensor_sum}")
    except:
        print("⚠️  Error in implementation")

print("\n📝 Exercise 2: Autograd Practice")
print("Use autograd to find derivatives")

if PYTORCH_AVAILABLE:
    def autograd_exercise():
        """
        Given f(x, y) = x²y + y³
        Find ∂f/∂x and ∂f/∂y at point (x=2, y=3)
        """
        # YOUR CODE HERE
        x = None  # Create x tensor with requires_grad=True
        y = None  # Create y tensor with requires_grad=True
        
        # Compute f = x²y + y³
        f = None
        
        # Compute gradients
        # YOUR CODE HERE
        
        return f, x.grad, y.grad if x is not None else (None, None, None)
    
    # Test
    try:
        f_val, dx, dy = autograd_exercise()
        if f_val is None:
            print("⚠️  Not implemented yet")
        else:
            print(f"f(2,3) = {f_val}")
            print(f"∂f/∂x = {dx} (should be 2xy = 12)")
            print(f"∂f/∂y = {dy} (should be x² + 3y² = 31)")
    except:
        print("⚠️  Error in implementation")

print("\n📝 Exercise 3: Build a Mini Neural Network")
print("Create a network for regression")

if PYTORCH_AVAILABLE:
    class MyFirstNet(nn.Module):
        """
        Build a network with:
        - Input size: 5
        - Hidden layer: 10 neurons with ReLU
        - Output size: 1
        """
        def __init__(self):
            super().__init__()
            # YOUR CODE HERE
            # Define layers
            pass
            
        def forward(self, x):
            # YOUR CODE HERE
            # Define forward pass
            return x
    
    # Test your network
    try:
        net = MyFirstNet()
        test_input = torch.randn(3, 5)  # 3 samples, 5 features
        output = net(test_input)
        print(f"Network created!")
        print(f"Input shape: {test_input.shape}")
        print(f"Output shape: {output.shape}")
        
        # Count parameters
        params = sum(p.numel() for p in net.parameters())
        print(f"Total parameters: {params} (should be 5*10 + 10 + 10*1 + 1 = 71)")
    except:
        print("⚠️  Network not implemented correctly")

print("\n🎉 Congratulations! You've taken your first steps with PyTorch!")
print("\nYou've learned:")
print("  ✓ Tensors are like NumPy arrays with superpowers")
print("  ✓ Autograd computes gradients automatically")
print("  ✓ nn.Module is the base for all neural networks")
print("  ✓ The training loop pattern: forward → loss → backward → step")
print("  ✓ GPU acceleration is just a .to() away")
print("\nNext: Convolutional Neural Networks with PyTorch!")