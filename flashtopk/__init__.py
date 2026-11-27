# SPDX-License-Identifier: Apache-2.0
"""FlashTopK: exact, deterministic fused FP8 top-k selection for DSA prefill.

Adapted from Tencent-Hunyuan/FlashTopK (090902abad86e5bb3324519124a680122fff91a1)
via SGLang PR #52822 (f6c4acab9220d6b08222a0d9eaec9c869e832419), with caller-owned
transient scratch, configurable adaptive pruning threshold, and a custom recovery/audit
finish callback.
"""

from __future__ import annotations

__version__ = "0.1.0+sii1"

__all__ = [
    "MAX_TOPK",
    "Selector",
    "__version__",
    "candidate_cap",
    "kernels",
    "plan",
    "recovery",
    "sample_count",
]


def __getattr__(name: str):
    if name in ("MAX_TOPK", "Selector", "candidate_cap", "sample_count"):
        from . import topk

        return getattr(topk, name)
    if name in ("kernels", "plan", "recovery"):
        import importlib

        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
