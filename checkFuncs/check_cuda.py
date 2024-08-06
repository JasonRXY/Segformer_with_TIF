import torch
import sys
import torch.distributed as dist

print("Current Python version:", sys.version)
print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("Is CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())



# 检查是否有 init_process_group 方法
if hasattr(dist, 'init_process_group'):
    print(f"PyTorch {torch.__version__} supports distributed training.")
else:
    print(f"PyTorch {torch.__version__} NOT supports distributed training.")

