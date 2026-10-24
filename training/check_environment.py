import torch, torchvision, transformers
print("torch",torch.__version__,"torchvision",torchvision.__version__,"transformers",transformers.__version__)
assert torch.cuda.is_available(), "CUDA GPU unavailable; do not accidentally train on CPU"
print("GPU:",torch.cuda.get_device_name(0))
print("VRAM GiB:",round(torch.cuda.get_device_properties(0).total_memory/2**30,1))
from model import EmotionModel
print("Imports OK. Weights download happens on first training run.")
