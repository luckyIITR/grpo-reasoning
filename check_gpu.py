import torch
print(f"torch:           {torch.__version__}")
print(f"compiled CUDA:   {torch.version.cuda}")
print(f"CUDA available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device count:    {torch.cuda.device_count()}")
    print(f"device 0:        {torch.cuda.get_device_name(0)}")