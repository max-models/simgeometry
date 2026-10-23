"""The absorb_polygons_2d kernel: Pyccel, NumPy and CUDA versions with the same arguments.

``kernel(...)`` runs the CUDA version when the arrays are CuPy arrays, and
otherwise the Pyccel version, or the NumPy version if Pyccel is unavailable
(see ``cunumpy.kernels.set_host_kernel_implementation`` to choose). The CUDA
version runs one thread per entry of ``rows`` (the third argument).
"""

import cunumpy as xp

from simgeometry.kernels._build import KERNEL_OPTIONS


def _row_count(args: tuple) -> int:
    return int(args[2].shape[0])


kernel = xp.kernels.Kernel.from_folder(
    __name__, **{**KERNEL_OPTIONS, "n_threads_from": _row_count}
)
