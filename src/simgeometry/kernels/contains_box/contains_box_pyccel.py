"""Mask of the markers inside an axis-aligned box (any dimension).

The Pyccel version (CPU) of the ``contains_box`` kernel; ``contains_box_cuda.cu``
next to this file is the CUDA version (GPU), with the same arguments.  ``positions`` is ``(ndim, N)`` (component-major).  A point
on the box surface counts as inside.  With ``inverted == 1`` the mask is
negated: the solid is everything outside the box.
"""


def contains_box(
    positions: "float[:,:]",
    mask: "bool[:]",
    lower: "float[:]",
    upper: "float[:]",
    inverted: int,
) -> None:
    """Set ``mask[marker]`` to whether the marker lies in the box ``[lower, upper]``."""
    num_markers = positions.shape[1]
    ndim = positions.shape[0]
    # fmt: off
    #$ omp parallel for private(inside, axis) schedule(dynamic, 4096)
    # fmt: on
    for marker in range(num_markers):
        inside = True
        for axis in range(ndim):
            if positions[axis, marker] < lower[axis]:
                inside = False
            if positions[axis, marker] > upper[axis]:
                inside = False
        if inverted == 1:
            inside = not inside
        mask[marker] = inside
