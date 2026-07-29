"""Compatibility helpers for project pickles read by legacy model environments."""

from __future__ import annotations

import pickle
from typing import Any, BinaryIO


class NumpyCompatUnpickler(pickle.Unpickler):
    """Load NumPy 2 pickles in NumPy 1 environments.

    NumPy 2 serializes some arrays through the private ``numpy._core``
    namespace. NumPy 1 exposes the same implementation as ``numpy.core``.
    Remap only that namespace and leave every other pickle global untouched.
    """

    @staticmethod
    def compatible_module(module: str) -> str:
        """Return the module name available in both NumPy major versions."""

        if module == "numpy._core" or module.startswith("numpy._core."):
            return "numpy.core" + module[len("numpy._core") :]
        return module

    def find_class(self, module: str, name: str):
        return super().find_class(self.compatible_module(module), name)


def load_pickle_compat(handle: BinaryIO) -> Any:
    """Load a pickle with the narrow NumPy 2-to-1 namespace remapping."""

    return NumpyCompatUnpickler(handle).load()
