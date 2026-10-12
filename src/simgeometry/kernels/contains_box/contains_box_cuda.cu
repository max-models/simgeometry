// The CUDA version (GPU) of the contains_box kernel; contains_box_pyccel.py next
// to this file is the Pyccel version (CPU), with the same arguments.  Loaded by
// picnic.kernels.catalog: one thread per marker; positions are (ndim, N).
//
// mask[marker] = marker inside the box [lower, upper] (negated if inverted).

#include "cunumpy/array_view.cuh"

extern "C" __global__
void contains_box(
    Array2D<double> positions,
    Array1D<char> mask,
    Array1D<double> lower,
    Array1D<double> upper,
    int inverted
) {
    const long long marker = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (marker >= positions.shape[1]) return;
    bool inside = true;
    for (long long axis = 0; axis < positions.shape[0]; ++axis) {
        const double x = positions(axis, marker);
        if (x < lower(axis) || x > upper(axis)) inside = false;
    }
    if (inverted == 1) inside = !inside;
    mask(marker) = inside ? 1 : 0;
}
