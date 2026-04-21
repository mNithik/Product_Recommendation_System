"""Verify whether implicit ALS/BPR CUDA kernels are usable."""

from __future__ import annotations

import importlib.util

import numpy as np
from scipy.sparse import csr_matrix


def _check_model(label: str, cls) -> None:
    x = csr_matrix(np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]], dtype=np.float32))
    try:
        model = cls(factors=8, iterations=1, use_gpu=True, random_state=42)
        model.fit(x, show_progress=False)
        print(f"{label}: GPU fit OK")
    except Exception as exc:  # noqa: BLE001
        print(f"{label}: GPU fit FAIL -> {type(exc).__name__}: {exc}")


def main() -> None:
    import implicit
    from implicit.als import AlternatingLeastSquares
    from implicit.bpr import BayesianPersonalizedRanking

    print(f"implicit version: {implicit.__version__}")
    print(f"implicit.gpu module present: {importlib.util.find_spec('implicit.gpu') is not None}")
    print(f"implicit.cuda module present: {importlib.util.find_spec('implicit.cuda') is not None}")

    _check_model("ALS", AlternatingLeastSquares)
    _check_model("BPR", BayesianPersonalizedRanking)


if __name__ == "__main__":
    main()
