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