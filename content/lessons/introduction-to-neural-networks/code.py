"""
Lesson 1: Introduction to Neural Networks
Let's explore the basic concepts with simple Python code!
"""

import numpy as np
import matplotlib.pyplot as plt

print("="*60)
print("LESSON 1: INTRODUCTION TO NEURAL NETWORKS")
print("="*60)

# -----------------------------------------------------------------------------
# PART 1: A SIMPLE NEURON
# -----------------------------------------------------------------------------
print("\n1. A SIMPLE NEURON")
print("-" * 40)

# Let's create a simple neuron that decides if you should go outside
# Inputs: temperature (0-100°F), is_raining (0 or 1), have_umbrella (0 or 1)

def simple_neuron(temperature, is_raining, have_umbrella):
    """A neuron that decides if you should go outside"""
    
    # These are the weights - how important is each input?
    weight_temperature = 0.5    # Temperature is pretty important
    weight_raining = -2.0       # Rain is bad for going outside
    weight_umbrella = 1.0       # Having an umbrella helps
    
    # The bias - our baseline tendency to go outside
    bias = -10  # We're slightly hesitant
    
    # Calculate the weighted sum
    total = (temperature * weight_temperature + 
             is_raining * weight_raining + 
             have_umbrella * weight_umbrella + 
             bias)
    
    # Decision: if total > 0, go outside!
    decision = "Go outside! 🌞" if total > 0 else "Stay inside 🏠"
    
    return total, decision

# Test different scenarios
scenarios = [
    (75, 0, 0),  # Nice day, no rain, no umbrella
    (75, 1, 0),  # Nice day, raining, no umbrella
    (75, 1, 1),  # Nice day, raining, have umbrella
    (30, 0, 0),  # Cold day, no rain, no umbrella
]

print("Testing our 'should I go outside?' neuron:\n")
for temp, rain, umbrella in scenarios:
    total, decision = simple_neuron(temp, rain, umbrella)
    rain_status = "Raining" if rain else "Not raining"
    umbrella_status = "Have umbrella" if umbrella else "No umbrella"
    print(f"  {temp}°F, {rain_status}, {umbrella_status}")
    print(f"  Score: {total:.1f} → {decision}\n")

# -----------------------------------------------------------------------------
# PART 2: PATTERN RECOGNITION EXAMPLE
# -----------------------------------------------------------------------------
print("\n2. PATTERN RECOGNITION - Is it a smiley face?")
print("-" * 40)

# Simple 3x3 pixel images (1 = black, 0 = white)
smiley_face = np.array([
    [0, 1, 0],
    [0, 0, 0],
    [1, 0, 1]
])

sad_face = np.array([
    [0, 1, 0],
    [1, 0, 1],
    [0, 0, 0]
])

random_pattern = np.array([
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 1]
])

# A simple "smiley detector" with learned weights
# These weights were "learned" to detect the smiley pattern
smiley_weights = np.array([
    [0, 1, 0],      # Eyes should be at the top
    [-1, -1, -1],   # Middle should be empty
    [1, 0, 1]       # Smile at the bottom
])

def detect_smiley(image, weights):
    """Detect if an image is a smiley face"""
    # Element-wise multiplication and sum (like a dot product for 2D)
    score = np.sum(image * weights)
    is_smiley = score > 1  # Threshold
    return score, is_smiley

# Test our detector
test_images = [
    ("Smiley face", smiley_face),
    ("Sad face", sad_face),
    ("Random pattern", random_pattern)
]

print("Testing our smiley face detector:\n")
for name, image in test_images:
    score, is_smiley = detect_smiley(image, smiley_weights)
    result = "😊 It's a smiley!" if is_smiley else "❌ Not a smiley"
    print(f"{name}: Score = {score} → {result}")

# Visualize the patterns
fig, axes = plt.subplots(1, 4, figsize=(12, 3))
images_to_show = [
    ("Smiley", smiley_face),
    ("Sad", sad_face),
    ("Random", random_pattern),
    ("Weights", smiley_weights)
]

for ax, (title, img) in zip(axes, images_to_show):
    ax.imshow(img, cmap='gray', vmin=-1, vmax=1)
    ax.set_title(title)
    ax.axis('off')

plt.suptitle("Pattern Recognition Example", fontsize=14)
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# PART 3: LEARNING FROM EXAMPLES
# -----------------------------------------------------------------------------
print("\n\n3. HOW NEURAL NETWORKS LEARN")
print("-" * 40)

# Let's simulate learning with a simple example
# We want to learn the pattern: output = 2 * input

# Training data
inputs = np.array([1, 2, 3, 4, 5])
correct_outputs = np.array([2, 4, 6, 8, 10])

# Start with a random weight
weight = 0.5
learning_rate = 0.1

print("Learning to multiply by 2:")
print(f"Starting weight: {weight}")
print("\nTraining...")

# Track the learning process
weights_history = [weight]
errors_history = []

# Simple learning loop
for epoch in range(10):
    total_error = 0
    
    for x, y_correct in zip(inputs, correct_outputs):
        # Forward pass: make prediction
        y_predicted = x * weight
        
        # Calculate error
        error = y_correct - y_predicted
        total_error += abs(error)
        
        # Backward pass: adjust weight
        weight_adjustment = learning_rate * error * x / len(inputs)
        weight += weight_adjustment
    
    weights_history.append(weight)
    errors_history.append(total_error / len(inputs))
    
    if epoch % 3 == 0:
        print(f"  Epoch {epoch}: weight = {weight:.3f}, avg error = {total_error/len(inputs):.3f}")

print(f"\nFinal weight: {weight:.3f} (should be close to 2.0)")

# Test our learned model
print("\nTesting on new data:")
test_inputs = [6, 7, 8]
for x in test_inputs:
    prediction = x * weight
    print(f"  Input: {x} → Prediction: {prediction:.1f} (correct: {x*2})")

# Visualize the learning process
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Plot 1: Weight evolution
ax1.plot(weights_history, 'b-o', linewidth=2, markersize=6)
ax1.axhline(y=2, color='r', linestyle='--', label='Target weight (2.0)')
ax1.set_xlabel('Training epoch')
ax1.set_ylabel('Weight value')
ax1.set_title('How the Weight Changes During Learning')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Error reduction
ax2.plot(errors_history, 'r-o', linewidth=2, markersize=6)
ax2.set_xlabel('Training epoch')
ax2.set_ylabel('Average error')
ax2.set_title('Error Decreases as Network Learns')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# INTERACTIVE EXERCISE
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print("EXERCISE: Build Your Own Simple Classifier")
print("="*60)

def create_classifier(weight1, weight2, bias):
    """
    Create a classifier that decides between two categories
    based on two input features.
    
    TODO: Complete this function
    - Calculate: score = input1 * weight1 + input2 * weight2 + bias
    - Return: "Category A" if score > 0, else "Category B"
    """
    def classifier(input1, input2):
        # YOUR CODE HERE
        # Calculate the weighted sum
        score = None  # Replace None with your calculation
        
        # Make decision
        if score is None:
            return "Not implemented"
        return "Category A" if score > 0 else "Category B"
    
    return classifier

# Test your classifier
print("\n📝 Exercise: Animal Classifier")
print("Let's classify animals based on size (0-10) and furriness (0-10)")
print("We want: small furry → Cat, large not-furry → Elephant")

# These weights should prefer small+furry for Cat
my_classifier = create_classifier(
    weight1=-0.5,  # Negative weight for size (smaller is better for cats)
    weight2=0.8,   # Positive weight for furriness
    bias=2         # Slight preference for cats
)

test_animals = [
    ("Small furry", 2, 8),      # Should be Cat
    ("Large not-furry", 9, 1),  # Should be Elephant
    ("Medium furry", 5, 6),     # Could be either
]

print("\nTesting your classifier:")
for description, size, furriness in test_animals:
    result = my_classifier(size, furriness)
    print(f"  {description} (size={size}, furriness={furriness}) → {result}")

print("\n🎉 Congratulations! You've completed Lesson 1!")
print("\nKey concepts you've learned:")
print("  ✓ Neurons make decisions based on weighted inputs")
print("  ✓ Pattern recognition happens through matching weights to patterns")
print("  ✓ Learning means adjusting weights to reduce errors")
print("  ✓ Simple rules can create complex behaviors")
print("\nNext lesson: The math behind neural networks!")