# Lesson 11: Introduction to PyTorch - Your Gateway to Modern Deep Learning

## Why PyTorch?

Remember building neural networks from scratch? All those NumPy arrays, manual gradient calculations, and careful bookkeeping? That was like building a house with hand tools. PyTorch is your power toolkit.

PyTorch gives you three superpowers:

1. **Tensors that live on GPUs** - Like NumPy arrays but turbocharged
2. **Automatic differentiation** - No more manual backprop calculations
3. **Pre-built neural network components** - Layers, optimizers, and more

The best part? PyTorch feels like Python. If you can write Python, you can write PyTorch. No weird syntax. No cryptic errors. Just clean, readable code that runs fast.

## From NumPy to PyTorch: A Familiar Friend

PyTorch tensors are just like NumPy arrays with extra abilities. Let's see the parallels:

**NumPy way:**

```python
import numpy as np
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
z = x + y
```

**PyTorch way:**

```python
import torch
x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])
z = x + y
```

See? Almost identical. But PyTorch tensors can:

- Move to GPU for 100x speedups
- Track their computation history
- Calculate gradients automatically

Think of it this way: NumPy arrays are bicycles. PyTorch tensors are motorcycles. Same basic idea, way more power.

## Tensors: The Building Blocks

Everything in PyTorch is a tensor. Scalars, vectors, matrices, images, entire neural networks - all tensors under the hood.

**Creating Tensors:**

```python
# From data
x = torch.tensor([1.0, 2.0, 3.0])

# Common patterns
zeros = torch.zeros(3, 4)      # 3x4 matrix of zeros
ones = torch.ones(2, 3)        # 2x3 matrix of ones
random = torch.randn(3, 3)     # 3x3 random normal values
```

**Key Difference from NumPy:**

PyTorch defaults to 32-bit floats (better for neural networks). NumPy defaults to 64-bit. This saves memory and speeds up training.

**Shape Matters:**

Just like NumPy, shapes are crucial:

```python
x = torch.randn(3, 4)
print(x.shape)  # torch.Size([3, 4])
print(x.size()) # Same thing
```

## The Magic of Autograd

Here's where PyTorch shines. Remember calculating gradients by hand? PyTorch does it automatically.

**The Simple Example:**

Let's compute the gradient of y = x² at x = 3:

```python
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)  # tensor(6.) because dy/dx = 2x = 6
```

That's it! No manual derivative calculation. PyTorch tracked that y came from squaring x and computed the gradient.

**How It Works:**

1. Set `requires_grad=True` on input tensors
2. Do your computation
3. Call `.backward()` on the output
4. Gradients appear in `.grad` attributes

It's like having a math tutor who watches your calculations and instantly provides all derivatives.

## Computational Graphs: PyTorch's Secret Weapon

Every operation creates a piece of a graph:

```python
x = torch.tensor(2.0, requires_grad=True)
y = x + 3
z = y * y
w = z - 4
```

PyTorch builds this graph:

```
x (2.0) → [+3] → y (5.0) → [square] → z (25.0) → [-4] → w (21.0)
```

When you call `w.backward()`, PyTorch walks backward through this graph, applying the chain rule automatically. Magic!

**Dynamic Graphs:**

Unlike some frameworks, PyTorch rebuilds the graph each forward pass. This means you can use Python control flow:

```python
if x > 0:
    y = x * 2
else:
    y = x * 3
```

The graph adapts to your logic. No special syntax needed.

## Building Neural Networks the PyTorch Way

Remember our manual neural network? Here's the PyTorch version:

```python
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

That's a complete neural network! PyTorch handles:

- Weight initialization
- Forward propagation
- Gradient computation
- Everything else

**The nn.Module Magic:**

Every PyTorch model inherits from `nn.Module`. This gives you:

- Automatic parameter tracking
- Easy device movement (CPU → GPU)
- Built-in training/evaluation modes
- Module composition

## Training Loop: Bringing It All Together

Here's the pattern you'll use thousands of times:

```python
model = SimpleNet()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(100):
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Backward pass
    optimizer.zero_grad()  # Clear old gradients
    loss.backward()        # Compute new gradients
    optimizer.step()       # Update weights
```

**The Key Steps:**

1. **Forward**: Push data through the model
2. **Loss**: Measure how wrong we are
3. **Zero**: Clear gradients (they accumulate by default!)
4. **Backward**: Compute gradients via autograd
5. **Step**: Update weights using gradients

This loop is the heartbeat of deep learning.

## GPU Acceleration: Feel the Speed

Moving to GPU is trivial:

```python
# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move model and data
model = model.to(device)
inputs = inputs.to(device)
targets = targets.to(device)
```

That's it! Your code now runs on GPU. Expect 10-100x speedups for large models.

**Pro tip:** Always move model and data to the same device. Mixing CPU and GPU tensors causes errors.

## Common Gotchas and How to Avoid Them

**1. Forgetting to zero gradients:**

```python
# Wrong - gradients accumulate!
for epoch in range(100):
    loss.backward()
    optimizer.step()

# Right
for epoch in range(100):
    optimizer.zero_grad()  # Always zero first!
    loss.backward()
    optimizer.step()
```

**2. Tracking gradients when you shouldn't:**

```python
# During evaluation
model.eval()
with torch.no_grad():  # Turn off gradient tracking
    predictions = model(test_data)
```

**3. Shape mismatches:**

```python
# PyTorch won't automatically broadcast everything
x = torch.randn(10, 5)
y = torch.randn(5)  # This might not work as expected!
y = torch.randn(1, 5)  # Better - explicit broadcasting
```

## PyTorch vs NumPy: When to Use What

**Use NumPy for:**

- Data preprocessing
- Simple numerical computations
- When you don't need gradients

**Use PyTorch for:**

- Neural networks
- Anything needing gradients
- GPU computation
- Deep learning models

You can convert between them easily:

```python
# NumPy to PyTorch
np_array = np.array([1, 2, 3])
torch_tensor = torch.from_numpy(np_array)

# PyTorch to NumPy
torch_tensor = torch.tensor([1, 2, 3])
np_array = torch_tensor.numpy()
```

## The PyTorch Ecosystem

PyTorch isn't just a library - it's an ecosystem:

- **TorchVision**: Computer vision datasets and models
- **TorchText**: NLP tools and datasets
- **TorchAudio**: Audio processing
- **PyTorch Lightning**: High-level training framework
- **Hugging Face**: Pre-trained transformer models

Each builds on PyTorch's foundation, adding domain-specific tools.

## Your First Real PyTorch Model

Let's build something real - a network that learns XOR:

```python
# Data
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# Model
model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
    nn.Sigmoid()
)

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
criterion = nn.BCELoss()

for epoch in range(1000):
    output = model(X)
    loss = criterion(output, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

In 20 lines, you've solved XOR - something a simple perceptron can't do!

## Debugging PyTorch Code

**Print shapes constantly:**

```python
x = torch.randn(10, 20)
print(f"x shape: {x.shape}")  # Debug every step
```

**Check gradient flow:**

```python
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad mean = {param.grad.mean()}")
```

**Use assertions:**

```python
assert x.shape == (10, 20), f"Expected (10, 20), got {x.shape}"
```

## The Path Forward

You've just scratched PyTorch's surface. Here's what's next:

1. **Master tensor operations** - They're the vocabulary of deep learning
2. **Understand autograd deeply** - It powers everything
3. **Learn the nn.Module patterns** - Every model uses them
4. **Practice the training loop** - It becomes second nature

PyTorch is your companion for the deep learning journey ahead. From CNNs to Transformers, from computer vision to NLP, everything builds on these foundations.

## Key Takeaways

1. **PyTorch = NumPy + Gradients + GPU**
2. **Tensors** are enhanced arrays that track computation
3. **Autograd** eliminates manual gradient calculation
4. **nn.Module** is your blueprint for models
5. **The training loop** is always: forward → loss → backward → step
6. **GPU acceleration** is one line of code
7. **Dynamic graphs** mean regular Python control flow works
8. **Start simple**, complexity comes later

## What's Next?

In the next lesson, we'll dive into Convolutional Neural Networks. You'll see how PyTorch makes building CNNs almost trivial - turning what would be thousands of lines of NumPy into elegant, readable code.

Remember: PyTorch isn't just a tool, it's a new way of thinking about deep learning. Instead of fighting with derivatives and matrix multiplies, you can focus on ideas and architectures. That's the real power here.
