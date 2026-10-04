"""One explicit numeric policy for L1 training, resume and standalone evaluation."""
import os
import torch


def configure_runtime(device):
    device = torch.device(device)
    if device.type not in ('cpu', 'cuda') or (device.type == 'cpu' and device.index is not None):
        raise ValueError('L1 supports CPU or CUDA only')
    gpu = None
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA unavailable: install a CUDA PyTorch wheel and check nvidia-smi')
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8', ':16:8'):
            raise ValueError('export CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting Python')
        index = torch.cuda.current_device() if device.index is None else device.index
        properties = torch.cuda.get_device_properties(index)
        gpu = {'index': index, 'name': properties.name,
               'capability': [properties.major, properties.minor],
               'total_memory': properties.total_memory}
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    return {'device': str(device), 'gpu': gpu, 'cuda': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version(),
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG') if gpu else None,
            'precision': 'float32_no_amp_no_tf32', 'deterministic_algorithms': True,
            'cudnn_benchmark': False, 'cudnn_deterministic': True}
