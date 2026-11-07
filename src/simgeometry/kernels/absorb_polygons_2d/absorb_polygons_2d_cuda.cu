// The CUDA version (GPU) of the absorb_polygons_2d kernel;
// absorb_polygons_2d_pyccel.py next to this file is the Pyccel version (CPU),
// with the same arguments: one thread per row. The alive markers rows[i] that end inside an object, or whose step from
// start crossed one of its edges, are absorbed; hit[i] is the object met first
// (-1: none). The objects are those of simgeometry_object_hit_time.

#include "cunumpy/array_view.cuh"
#include "simgeometry/kernels/polygon_hit.cuh"

extern "C" __global__
void absorb_polygons_2d(
    Array2D<double> positions,
    Array2D<double> start,
    Array1D<long long> rows,
    int use_start,
    double sweep_length,
    Array2D<double> vertices,
    Array1D<long long> polygon_offsets,
    Array1D<long long> object_first,
    Array1D<long long> object_last,
    Array1D<long long> object_inverted,
    Array2D<double> object_lower,
    Array2D<double> object_upper,
    Array1D<long long> object_boxed,
    Array1D<char> alive,
    Array1D<long long> hit
) {
    const long long index = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (index >= rows.shape[0]) return;
    const long long marker = rows(index);
    hit(index) = -1;
    if (!alive(marker)) return;
    const double x1 = positions(0, marker);
    const double y1 = positions(1, marker);
    double x0 = x1;
    double y0 = y1;
    int sweep = 0;
    if (use_start == 1) {
        x0 = start(0, marker);
        y0 = start(1, marker);
        const double length2 = (x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0);
        if (sweep_length > 0.0 && length2 <= sweep_length * sweep_length)
            sweep = 1;
    }
    double best = 2.0;
    long long best_object = -1;
    for (long long obj = 0; obj < object_first.shape[0]; ++obj) {
        if (object_boxed(obj) == 1
            && (fmax(x0, x1) < object_lower(obj, 0) || fmin(x0, x1) > object_upper(obj, 0)
                || fmax(y0, y1) < object_lower(obj, 1)
                || fmin(y0, y1) > object_upper(obj, 1)))
            continue;
        const double time = simgeometry_object_hit_time(
            vertices, polygon_offsets, object_first(obj), object_last(obj),
            object_inverted(obj), x0, y0, x1, y1, sweep);
        if (time < best) {
            best = time;
            best_object = obj;
        }
    }
    hit(index) = best_object;
    if (best_object >= 0) alive(marker) = 0;
}
