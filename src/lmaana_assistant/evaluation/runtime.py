"""Record experiment provenance without importing optional model libraries."""

import os
import platform
import sys
from importlib.metadata import PackageNotFoundError, version


def runtime_metadata(embedder) -> dict:
    packages = {}
    for name in ("torch", "transformers", "sentence-transformers", "huggingface-hub", "numpy"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None

    model = getattr(embedder, "model", None)
    parameter = next(model.parameters(), None) if hasattr(model, "parameters") else None
    torch = sys.modules.get("torch")  # Never import a heavy runtime merely to describe it.
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "package_versions": packages,
        "model_device": str(parameter.device) if parameter is not None else None,
        "model_dtype": str(parameter.dtype) if parameter is not None else None,
        "model_max_seq_length": getattr(model, "max_seq_length", None),
        "torch_threads": torch.get_num_threads() if hasattr(torch, "get_num_threads") else None,
        "thread_environment": {
            name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "scope": "Runtime provenance, not model-quality evidence or a throughput benchmark.",
    }
