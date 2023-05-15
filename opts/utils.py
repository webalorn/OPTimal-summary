import torch

try:
    from pynvml import *
    HAS_PYNVML = True
except:
    HAS_PYNVML = False

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print("Using device", device)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def print_gpu_utilization():
    if HAS_PYNVML:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(1)
        info = nvmlDeviceGetMemoryInfo(handle)
        print(f"GPU memory occupied: {info.used//1024**2} MB.")
    else:
        print("[WARNING] HAS_PYNVML=False")
