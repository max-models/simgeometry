"""Three-dimensional aperture helpers for geometric object definitions."""

from __future__ import annotations

import cunumpy as xp

from .._types import Array, ConfigValue
from ..shapes import Shape3D, _component_major


class Aperture3D(Shape3D):
    """Represent a three-dimensional aperture embedded in a simulation domain."""

    #: Even when inverted, the aperture is solid only inside its bounding box
    #: (``contains`` is False outside it), so markers far from the box can be
    #: skipped before the shape test.
    solid_within_bounding_box = True

    def __init__(
        self,
        axis: str,
        center_coords: tuple,
        vertices: tuple[tuple[float, float], ...],
        lower_bound: tuple,
        upper_bound: tuple,
        bc: ConfigValue = None,
        inverted: bool = True,
    ) -> None:
        """
        Initialize a 3D aperture object.

        Parameters:
        axis (str): Axis along which the aperture extends ('x', 'y', or 'z').
        center_coords (list or tuple): Center coordinate pair for the aperture's center axis.
        vertices (list of lists): List of lists where each element is [coordinate_along_axis, radial_distance].
        bounds (list of numpy arrays): Bounds of the aperture in 3D space, defined as [b0, b1].
        bc (optional): Boundary condition information.
        inverted (bool, optional): Defaults to True so apertures model solid
            material around a central opening unless explicitly disabled.
        """
        # print(f"{axis = }")
        assert axis in ["x", "y", "z"], "Axis must be one of 'x', 'y', or 'z'."
        assert len(center_coords) == 2, (
            "Center coordinates must be a pair (2D coordinates)."
        )
        assert len(lower_bound) == 3
        assert len(upper_bound) == 3
        assert all(len(vertex) == 2 for vertex in vertices), (
            "Each vertex must be a list [coordinate_along_axis, radial_distance]."
        )

        self.axis = axis
        self.center_coords = xp.array(center_coords, dtype=float)
        self.vertices = xp.array(vertices)
        self._bc = bc
        self._inverted = bool(inverted)

        # Store the minimum and maximum radial distance for fast checks
        self._rmin_squared = float(xp.min(self.vertices[:, 1])) ** 2

        # Determine the axis index
        self.axis_index = {"x": 0, "y": 1, "z": 2}[axis]
        self._perpendicular_indices = [i for i in range(3) if i != self.axis_index]

        # Set the bounds directly
        super().__init__(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

    def mirror(self, axis: int, position: float) -> None:
        """
        Mirror the aperture across a specified axis at a given position.

        Parameters:
            axis (int): Axis index (0 for x-axis, 1 for y-axis, 2 for z-axis).
            position (float): Position along the axis to mirror.
        """
        if axis < 0 or axis > 2:
            raise ValueError("Axis must be 0 (x-axis), 1 (y-axis), or 2 (z-axis).")

        # Mirror the center_coords along the specified axis if applicable
        if axis in self._perpendicular_indices:
            index_in_center_coords = 0 if self._perpendicular_indices[0] == axis else 1
            self.center_coords[index_in_center_coords] = (
                2 * position - self.center_coords[index_in_center_coords]
            )

        # Mirror the vertices along the aperture's axis if applicable
        if axis == self.axis_index:
            self.vertices[:, 0] = 2 * position - self.vertices[:, 0]

        lower_bound = self.lower_bound
        upper_bound = self.upper_bound

        self.set_bounding_box(lower_bound, upper_bound)

    def copy(self) -> Aperture3D:
        """
        Create a copy of the aperture.

        Returns:
            Aperture3D: A new instance of Aperture3D with the same properties.
        """
        return Aperture3D(
            axis=self.axis,
            center_coords=self.center_coords.copy(),
            vertices=self.vertices.copy(),
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            bc=self._bc,
            inverted=self._inverted,
        )

    def translate(self, offset: tuple[float, float, float]) -> None:
        """
        Translate the aperture by the given offset.

        Parameters:
            offset (tuple): (dx, dy, dz) translation vector.
        """
        offset_vec = xp.asarray(offset)

        # Apply translation to center_coords
        for i, axis_idx in enumerate(self._perpendicular_indices):
            self.center_coords[i] += offset_vec[axis_idx]

        # Apply translation to vertices along the aperture's axis
        self.vertices[:, 0] += offset_vec[self.axis_index]

        lower_bound = self.lower_bound
        upper_bound = self.upper_bound

        self.set_bounding_box(lower_bound, upper_bound)

    def get_vertices(self, slice_axis: int, slice_position: float) -> ConfigValue:
        """
        Get the vertices of a slice of the aperture along a given axis.

        Parameters:
        slice_axis (int): Axis along which to slice (0 for x, 1 for y, 2 for z).
        slice_position (float): Position along the slice_axis where the slice is taken.

        Returns:
        vertices (ndarray): Array of vertices defining the slice shape.
                            If the slice does not intersect the aperture, returns an empty array.
        """
        assert 0 <= slice_axis < 3, "slice_axis must be 0, 1, or 2."
        inverted = self.inverted
        # Check if the slice_position is within the bounds along the slice_axis
        if not (
            self.lower_bound[slice_axis]
            <= slice_position
            <= self.upper_bound[slice_axis]
        ):
            # The slice is outside the aperture's bounds
            return xp.array([]), inverted

        # Determine the axes indices
        axis_aperture = self.axis_index
        axes_perp_to_aperture = self._perpendicular_indices  # list of two axes

        # Check if slice_axis is equal to aperture's axis
        if slice_axis == axis_aperture:
            # Slice is perpendicular to the aperture's axis
            # Interpolate the radial distance at the given position along the aperture's axis
            axis_positions = self.vertices[:, 0]
            radial_distances = self.vertices[:, 1]

            # Check if the slice_position is within the vertices' range
            if not (axis_positions[0] <= slice_position <= axis_positions[-1]):
                # The slice_position is outside the defined vertices
                return xp.array([]), inverted

            # Interpolate radial distance
            radial_distance = xp.interp(
                xp.asarray(slice_position, dtype=float),
                axis_positions,
                radial_distances,
            )

            # If radial distance is zero or negative, return empty array
            if radial_distance <= 0:
                return xp.array([]), inverted

            # Generate points along the circle
            N = 100  # Number of points to approximate the circle
            theta = xp.linspace(0, 2 * xp.pi, N)

            # Get the center coordinates in the slicing plane
            center1 = self.center_coords[0]
            center2 = self.center_coords[1]

            x = center1 + radial_distance * xp.cos(theta)
            y = center2 + radial_distance * xp.sin(theta)

            # Identify the perpendicular axes
            _axis1, _axis2 = axes_perp_to_aperture

            # Combine x and y into an array of vertices
            vertices = xp.zeros((N, 2))
            vertices[:, 0] = x
            vertices[:, 1] = y

            return xp.array(vertices), inverted

        elif slice_axis in axes_perp_to_aperture:
            # The slice axis is one of the axes perpendicular to the aperture's axis
            # Find the index in center_coords corresponding to slice_axis
            index_in_center_coords = axes_perp_to_aperture.index(slice_axis)

            D = abs(self.center_coords[index_in_center_coords] - slice_position)

            # The maximum radius along the aperture
            r_max = xp.max(self.vertices[:, 1])

            # If D > r_max, the plane does not intersect the aperture
            if D > r_max:
                return xp.array([]), inverted

            # Generate positions along axis_aperture
            s_values = xp.linspace(self.vertices[0, 0], self.vertices[-1, 0], num=100)

            # Interpolate radial distances at s_values
            r_values = xp.interp(s_values, self.vertices[:, 0], self.vertices[:, 1])

            # For each s, check if D <= r(s)
            valid_indices = xp.where(r_values >= D)[0]

            if len(valid_indices) == 0:
                # No intersection
                return xp.array([]), inverted

            # For valid s_values, compute r_perp = sqrt(r(s)^2 - D^2)
            s_valid = s_values[valid_indices]
            r_valid = r_values[valid_indices]
            r_perp = xp.sqrt(r_valid**2 - D**2)

            # The coordinate along the other axis (not slice_axis) in axes_perp_to_aperture
            # axis_other = axes_perp_to_aperture[1 - index_in_center_coords]
            index_other_in_center_coords = 1 - index_in_center_coords
            coord_other_center = self.center_coords[index_other_in_center_coords]

            # Create vertices along the perimeter of the shape
            # First half of vertices (coord_other increasing)
            vertices1 = xp.column_stack((s_valid, coord_other_center + r_perp))

            # Second half of vertices (coord_other decreasing)
            vertices2 = xp.column_stack(
                (s_valid[::-1], coord_other_center - r_perp[::-1]),
            )

            # Combine vertices
            vertices = xp.vstack((vertices1, vertices2))

            return vertices, inverted

        else:
            # The slice_axis is neither the aperture's axis nor one of the axes perpendicular to it
            # In 3D, this should not happen, so return an empty array
            return xp.array([]), inverted

    def contains(self, coords: Array) -> bool:
        """
        Check if the given coordinates are inside the aperture.

        Parameters:
        coords (list or tuple): 3D coordinates.

        Returns:
        bool: True if the coordinates are inside the aperture, False otherwise.
        """
        coords = xp.array(coords)

        # Early exit if the point is outside the bounding box
        if not self.bounding_box.contains(coords):
            return False

        # Calculate radial position squared to avoid taking square root
        perp_coords = coords[self._perpendicular_indices] - self.center_coords
        radial_position_squared = xp.dot(perp_coords, perp_coords)

        # Early exit if the point is guaranteed to be inside the aperture profile.
        if radial_position_squared < self._rmin_squared:
            return not self.inverted

        # Get the position along the primary axis
        axis_position = coords[self.axis_index]

        # Binary search to find the appropriate radial distance in vertices
        index = xp.searchsorted(self.vertices[:, 0], axis_position, side="left")
        # print(f"{index = }")
        # If out of the vertices range, the point is outside
        if index == 0:
            radial_distance_squared = self.vertices[0, 1] ** 2
        elif index >= len(self.vertices):
            radial_distance_squared = self.vertices[-1, 1] ** 2
        else:
            # Linear interpolation for radial distance
            t = (axis_position - self.vertices[index - 1, 0]) / (
                self.vertices[index, 0] - self.vertices[index - 1, 0]
            )
            radial_distance_squared = (
                (1 - t) * self.vertices[index - 1, 1] + t * self.vertices[index, 1]
            ) ** 2

        # Check if the point is within the interpolated radial distance
        inside = bool(radial_position_squared <= radial_distance_squared)
        return (not inside) if self.inverted else inside

    def contains_points(self, points: Array) -> Array:
        """Vectorized :meth:`contains`: a mask of the ``(3, n)`` points inside."""
        points = _component_major(points, 3)
        if points.shape[1] == 0:
            return xp.zeros(0, dtype=bool)
        lower = xp.asarray(self.bounding_box.lower_bound, dtype=float)[:, None]
        upper = xp.asarray(self.bounding_box.upper_bound, dtype=float)[:, None]
        # half-open, as BoundingBox.contains
        in_box = xp.all((points >= lower) & (points < upper), axis=0)
        perp = points[self._perpendicular_indices] - self.center_coords[:, None]
        radial_squared = xp.sum(perp * perp, axis=0)
        axis_position = points[self.axis_index]
        vertices = self.vertices
        if len(vertices) == 1:
            radius = xp.full(points.shape[1], float(vertices[0, 1]))
            inside = radial_squared <= radius**2
            return in_box & (~inside if self.inverted else inside)
        index = xp.searchsorted(vertices[:, 0], axis_position, side="left")
        # beyond the vertices the first or last radius holds, as in contains
        inner = xp.clip(index, 1, len(vertices) - 1)
        x0, r0 = vertices[inner - 1, 0], vertices[inner - 1, 1]
        x1, r1 = vertices[inner, 0], vertices[inner, 1]
        t = (axis_position - x0) / (x1 - x0)
        radius = xp.where(
            index == 0,
            vertices[0, 1],
            xp.where(index >= len(vertices), vertices[-1, 1], (1 - t) * r0 + t * r1),
        )
        inside = (radial_squared < self._rmin_squared) | (radial_squared <= radius**2)
        solid = ~inside if self.inverted else inside
        return in_box & solid

    def volume(self) -> None:
        """Declare the aperture volume interface."""
        raise NotImplementedError("Volume calculation not implemented for Aperture3D.")

    @property
    def min_radius(self) -> float:
        """Return the smallest aperture radius."""
        return xp.min(self.vertices[:, 1])

    @property
    def max_radius(self) -> float:
        """Return the largest aperture radius."""
        return xp.max(self.vertices[:, 1])

    def surface_area(self) -> None:
        """Declare the aperture surface-area interface."""
        raise NotImplementedError(
            "Surface area calculation not implemented for Aperture3D.",
        )
