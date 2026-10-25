"""Explicit CPU/CUDA placement and reproducibility settings for both runners."""
import os

# Set before CUDA matrix operations/context creation in both parent and workers.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch


def resolve_device(requested='auto'):
    if requested == 'auto':
        requested = 'cuda' if torch.cuda.is_available() else 'cpu'
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable. Install the CUDA PyTorch build or use --device cpu.')
    if requested not in ['cpu','cuda']:
        raise ValueError(f'Unsupported training device: {requested}')
    return torch.device(requested)


def configure_device(requested='auto'):
    device = resolve_device(requested)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if device.type == 'cuda':
        torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # Preserve FP32 precision for the hyperbolic model and loss ablations.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return device


def device_metadata(device):
    device = torch.device(device)
    info = dict(device=str(device), torch=str(torch.__version__), cuda_build=torch.version.cuda,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                precision='float32, no AMP; TF32 disabled for CUDA',
                cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
    if device.type == 'cuda':
        properties = torch.cuda.get_device_properties(0)
        info.update(gpu=properties.name, gpu_memory_bytes=properties.total_memory,
                    compute_capability=f'{properties.major}.{properties.minor}',
                    cudnn_version=torch.backends.cudnn.version())
    return info


def model_device(model):
    return next(model.parameters()).device


def tensor_on(array, device):
    return torch.from_numpy(array).to(device)
