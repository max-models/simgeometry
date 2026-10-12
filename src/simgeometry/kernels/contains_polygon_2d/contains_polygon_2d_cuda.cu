// The CUDA version (GPU) of the contains_polygon_2d kernel;
// contains_polygon_2d_pyccel.py next to this file is the Pyccel version (CPU),
// with the same arguments.  Loaded by picnic.kernels.catalog: one thread per
// marker (positions are (2, N)), even-odd ray casting over the (n, 2) vertices.

#include "cunumpy/array_view.cuh"

extern "C" __global__
void contains_polygon_2d(
    Array2D<double> positions,
    Array1D<char> mask,
    Array2D<double> vertices,
    int inverted
) {
    const long long marker = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (marker >= positions.shape[1]) return;
    const double x = positions(0, marker);
    const double y = positions(1, marker);
    const long long n = vertices.shape[0];
    bool inside = false;
    double xj = vertices(n - 1, 0);
    double yj = vertices(n - 1, 1);
    for (long long i = 0; i < n; ++i) {
        const double xi = vertices(i, 0);
        const double yi = vertices(i, 1);
        if ((yi > y) != (yj > y)
            && x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-16) + xi) {
            inside = !inside;
        }
        xj = xi;
        yj = yi;
    }
    if (inverted == 1) inside = !inside;
    mask(marker) = inside ? 1 : 0;
}
