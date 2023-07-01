import torch

# Check for cuda support
if not torch.cuda.is_available():
    print("No GPUs available.")
else:
    # Get the number of GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}")

    # Loop over each GPU and print its memory
    for i in range(num_gpus):
        gpu = torch.device(f'cuda:{i}')
        total_mem = torch.cuda.get_device_properties(gpu).total_memory
        print(f"GPU {i} Total Memory: {total_mem / 1024**3:.2f} GB")
