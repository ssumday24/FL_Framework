"""Read-only environment checks; never downloads a model or calls an API."""
import importlib.metadata
import sys


def check_environment():
    issues = []
    result = {'python': sys.version.split()[0], 'executable': sys.executable,
              'packages': {}, 'gpu': None, 'issues': issues}
    if not (3, 10) <= sys.version_info[:2] < (3, 13):
        issues.append('Use Python 3.10-3.12 for the pinned inference dependencies; Python 3.13 is not supported by this setup.')
    for name in ('torch', 'transformers', 'accelerate', 'peft', 'bitsandbytes', 'sentencepiece'):
        try:
            result['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result['packages'][name] = None
            issues.append(f'Missing package: {name}')
    if result['packages']['torch']:
        try:
            import torch
            if not torch.cuda.is_available():
                issues.append('CUDA is unavailable in this Python environment; install a CUDA-enabled PyTorch build.')
            else:
                free, total = torch.cuda.mem_get_info(0)
                result['gpu'] = {'name': torch.cuda.get_device_name(0),
                                 'free_gib': round(free / 2**30, 2),
                                 'total_gib': round(total / 2**30, 2),
                                 'torch_cuda': torch.version.cuda}
        except (ImportError, OSError, RuntimeError) as error:
            issues.append(f'PyTorch/CUDA check failed: {error}')
    if result['packages']['bitsandbytes']:
        try:
            import bitsandbytes  # noqa: F401
        except (ImportError, OSError, RuntimeError) as error:
            issues.append(f'bitsandbytes import failed: {error}')
    result['ready_for_model_smoke_test'] = not issues
    return result
