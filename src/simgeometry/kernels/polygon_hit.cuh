// The swept polygon-object hit test on the GPU: the same arithmetic as
// _object_hit_time of absorb_polygons_2d_pyccel.py. Included by
// absorb_polygons_2d_cuda.cu and by kernels of other packages that fuse the
// test into their own loops (#include "simgeometry/kernels/polygon_hit.cuh",
// with simgeometry's source root among the include directories). An object is the union of polygons first <= p < last of
// the (V, 2) vertices, polygon p the vertices polygon_offsets[p] <= v <
// polygon_offsets[p + 1] with the closing edge implied.
#pragma once
#include "cunumpy/array_view.cuh"

// Where along the step from (x0, y0) to (x1, y1) the marker enters the object:
// the parameter in (0, 1] of the first edge crossing (sweep == 1), else 0 if
// (x1, y1) lies in the solid, else 2 (no hit).
__device__ inline double simgeometry_object_hit_time(
    Array2D<double> vertices, Array1D<long long> polygon_offsets, long long first,
    long long last, long long inverted, double x0, double y0, double x1, double y1,
    int sweep) {
    bool inside = false;
    double crossing = 2.0;
    const double dx = x1 - x0;
    const double dy = y1 - y0;
    for (long long polygon = first; polygon < last; ++polygon) {
        const long long begin = polygon_offsets(polygon);
        const long long end = polygon_offsets(polygon + 1);
        bool in_polygon = false;
        double xj = vertices(end - 1, 0);
        double yj = vertices(end - 1, 1);
        for (long long vertex = begin; vertex < end; ++vertex) {
            const double xi = vertices(vertex, 0);
            const double yi = vertices(vertex, 1);
            if ((yi > y1) != (yj > y1)
                && x1 < (xj - xi) * (y1 - yi) / ((yj - yi) + 1e-16) + xi)
                in_polygon = !in_polygon;
            if (sweep == 1) {
                // the step x0 + t d meets the edge xj + u e
                const double ex = xi - xj;
                const double ey = yi - yj;
                const double denominator = dx * ey - dy * ex;
                if (denominator != 0.0) {
                    const double wx = xj - x0;
                    const double wy = yj - y0;
                    const double t = (wx * ey - wy * ex) / denominator;
                    const double u = (wx * dy - wy * dx) / denominator;
                    if (t > 0.0 && t <= 1.0 && u >= 0.0 && u <= 1.0 && t < crossing)
                        crossing = t;
                }
            }
            xj = xi;
            yj = yi;
        }
        if (in_polygon) inside = true;
    }
    if (inverted == 1) inside = !inside;
    if (crossing <= 1.0) return crossing;
    return inside ? 0.0 : 2.0;
}
