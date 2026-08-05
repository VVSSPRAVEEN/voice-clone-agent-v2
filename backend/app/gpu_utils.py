"""GPU memory helpers for automatic CPU fallback when VRAM is constrained."""

from __future__ import annotations

import torch


def gpu_free_mb() -> int | None:
    """Return free GPU memory in MiB, or None when CUDA is unavailable."""
    if not torch.cuda.is_available():
        return None
    try:
        free_bytes, _ = torch.cuda.mem_get_info()
        return free_bytes // 2**20
    except Exception:
        return None


def should_use_cuda(min_free_mb: int) -> bool:
    """True when CUDA is available and at least min_free_mb MiB is free."""
    if not torch.cuda.is_available():
        return False
    from .config import SETTINGS
    if not SETTINGS.auto_cpu_fallback:
        return True
    free_mb = gpu_free_mb()
    return free_mb is not None and free_mb >= min_free_mb
