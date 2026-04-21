"""Lazy model exports to avoid importing heavy dependencies too early."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "ItemBasedCF",
    "MatrixFactorizationGPU",
    "BPRMatrixFactorization",
    "ImplicitBPRRanker",
    "ImplicitALSRanker",
    "WARPModel",
    "PopularityBaseline",
]

_MODEL_IMPORTS = {
    "ItemBasedCF": ("src.models.item_cf", "ItemBasedCF"),
    "MatrixFactorizationGPU": ("src.models.matrix_factorization", "MatrixFactorizationGPU"),
    "BPRMatrixFactorization": ("src.models.bpr", "BPRMatrixFactorization"),
    "ImplicitBPRRanker": ("src.models.implicit_bpr", "ImplicitBPRRanker"),
    "ImplicitALSRanker": ("src.models.implicit_als", "ImplicitALSRanker"),
    "WARPModel": ("src.models.warp", "WARPModel"),
    "PopularityBaseline": ("src.models.popularity", "PopularityBaseline"),
}


def __getattr__(name: str):
    """Load model classes on demand."""
    if name not in _MODEL_IMPORTS:
        raise AttributeError(f"module 'src.models' has no attribute {name!r}")
    module_name, attr_name = _MODEL_IMPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
