import torch

# Check if the GPU is turned on and recognized
if torch.cuda.is_available():
    print("✅ GPU is awake and ready!")
    print("Device name:", torch.cuda.get_device_name(0))
    
    # Create a random matrix and forcefully push it to the GPU memory
    tensor = torch.rand(3, 3).cuda()
    print("\nSuccessfully crunched this tensor directly on the GPU:")
    print(tensor)
else:
    print("❌ GPU not found. It's defaulting to the CPU.")

import torch.nn as nn
import matplotlib.pyplot as plt

# 1. Move everything to your RTX 3050
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 2. Create fake data (The hidden rule: y = 2.5x + 3)
x = torch.randn(100, 1) * 10 
# We add random noise so it's not a perfect line
y = 2.5 * x + 3 + torch.randn(100, 1) * 3 

x, y = x.to(device), y.to(device)

# 3. Build the simplest network possible (1 input feature -> 1 output prediction)
model = nn.Linear(in_features=1, out_features=1).to(device)

# 4. Setup the Optimizer and Loss function
criterion = nn.MSELoss() # Mean Squared Error: measures how far off the line is
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # Tweaks the line's slope and intercept

# 5. The Training Loop (Run through the data 200 times)
for epoch in range(200):
    # Forward pass: guess the answers
    predictions = model(x)
    
    # Calculate the error
    loss = criterion(predictions, y)
    
    # Backward pass: learn from the mistakes
    optimizer.zero_grad() # Clear old memory
    loss.backward()       # Calculate the gradients (the math part)
    optimizer.step()      # Actually update the line

# 6. Check the final math
print(f"Hidden Rule: y = 2.50x + 3.00")
print(f"AI Learned:  y = {model.weight.item():.2f}x + {model.bias.item():.2f}")

# 7. Plot it visually (pulling data back to the CPU so Matplotlib can read it)
plt.scatter(x.cpu().numpy(), y.cpu().numpy(), label='Messy Data', alpha=0.5, color='gray')
plt.plot(x.cpu().numpy(), model(x).cpu().detach().numpy(), color='red', label='AI Best Fit Line')
plt.legend()
plt.show()