"""Mask of the markers inside a ball (disc in 2D, sphere in 3D, any dimension).

The Pyccel version (CPU) of the ``contains_ball`` kernel; ``contains_ball_cuda.cu``
next to this file is the CUDA version (GPU), with the same arguments.  ``positions`` is ``(ndim, N)`` (component-major).  A point
on the surface counts as inside.  With ``inverted == 1`` the mask is negated.
"""


def contains_ball(
    positions: "float[:,:]",
    mask: "bool[:]",
    midpoint: "float[:]",
    radius: float,
    inverted: int,
) -> None:
    """Set ``mask[marker]`` to whether the marker lies within ``radius`` of ``midpoint``."""
    num_markers = positions.shape[1]
    ndim = positions.shape[0]
    radius_squared = radius * radius
    # fmt: off
    #$ omp parallel for private(distance_squared, axis, delta, inside) schedule(dynamic, 4096)
    # fmt: on
    for marker in range(num_markers):
        distance_squared = 0.0
        for axis in range(ndim):
            delta = positions[axis, marker] - midpoint[axis]
            distance_squared += delta * delta
        inside = distance_squared <= radius_squared
        if inverted == 1:
            inside = not inside
        mask[marker] = inside
