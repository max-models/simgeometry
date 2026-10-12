// The CUDA version (GPU) of the contains_ball kernel; contains_ball_pyccel.py next
// to this file is the Pyccel version (CPU), with the same arguments.  Loaded by
// picnic.kernels.catalog: one thread per marker; positions are (ndim, N).
//
// mask[marker] = |position - midpoint| <= radius (negated if inverted).

#include "cunumpy/array_view.cuh"

extern "C" __global__
void contains_ball(
    Array2D<double> positions,
    Array1D<char> mask,
    Array1D<double> midpoint,
    double radius,
    int inverted
) {
    const long long marker = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (marker >= positions.shape[1]) return;
    double distance_squared = 0.0;
    for (long long axis = 0; axis < positions.shape[0]; ++axis) {
        const double delta = positions(axis, marker) - midpoint(axis);
        distance_squared += delta * delta;
    }
    bool inside = distance_squared <= radius * radius;
    if (inverted == 1) inside = !inside;
    mask(marker) = inside ? 1 : 0;
}
