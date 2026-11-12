"""The mesh_inside kernel: Pyccel and NumPy versions with the same arguments.

``kernel(...)`` runs the Pyccel version, or the NumPy version if Pyccel is
unavailable (see ``cunumpy.kernels.set_host_kernel_implementation`` to choose).
There is no CUDA version yet: CuPy arguments are copied to the host and back.
"""

from simgeometry.kernels import KERNEL_PACKAGE

kernel = KERNEL_PACKAGE.from_folder(__name__, missing_cuda="fallback")
