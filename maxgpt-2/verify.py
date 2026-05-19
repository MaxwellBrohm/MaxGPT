import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Device count: {torch.cuda.device_count()}")
print(f"Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

x = torch.randn(3, 3, device="cuda")
y = torch.randn(3, 3, device="cuda")
z = x @ y

print(f"Tensor on CUDA: {z.device}")
print(z)