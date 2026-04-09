"""Runtime compatibility shims loaded automatically by Python startup.

This file is imported by Python's site machinery when its directory is on
`sys.path`. We keep it tiny and fail-safe so it never blocks normal startup.
"""

from __future__ import annotations

import sys
import types


def _install_torchvision_functional_tensor_alias() -> None:
    """Provide torchvision.transforms.functional_tensor for old BasicSR code.

    BasicSR<=1.4 imports:
      `from torchvision.transforms.functional_tensor import rgb_to_grayscale`
    while recent torchvision moved the symbol to `transforms.functional`.
    """
    try:
        # Already available in this environment.
        __import__("torchvision.transforms.functional_tensor")
        return
    except Exception:
        pass

    try:
        from torchvision.transforms import functional as _tv_f  # type: ignore
    except Exception:
        return

    rgb_to_grayscale = getattr(_tv_f, "rgb_to_grayscale", None)
    if rgb_to_grayscale is None:
        return

    shim = types.ModuleType("torchvision.transforms.functional_tensor")
    shim.rgb_to_grayscale = rgb_to_grayscale  # type: ignore[attr-defined]
    sys.modules["torchvision.transforms.functional_tensor"] = shim


_install_torchvision_functional_tensor_alias()

