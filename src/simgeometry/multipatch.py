"""Multi-patch structured grid support for complex geometries."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ._types import ConfigValue


@dataclass
class PatchDescriptor:
    """Metadata describing a single patch in a multi-patch grid."""

    name: str

    lower_bound: tuple[float, ...]

    upper_bound: tuple[float, ...]

    num_gridpoints: tuple[int, ...]

    resolution: float | None = None
    """Unique identifier for this patch (e.g., 'L_vertical', 'L_horizontal')."""
    """Lower corner coordinates of the patch."""
    """Upper corner coordinates of the patch."""
    """Number of grid points in each dimension."""
    """Grid spacing (computed from bounds and num_gridpoints if None)."""

    def __post_init__(self) -> None:
        """Validate patch descriptor."""
        if len(self.lower_bound) != len(self.upper_bound):
            raise ValueError(
                f"lower_bound and upper_bound must have same length. "
                f"Got {len(self.lower_bound)} and {len(self.upper_bound)}",
            )
        if len(self.lower_bound) != len(self.num_gridpoints):
            raise ValueError(
                f"lower_bound and num_gridpoints must have same length. "
                f"Got {len(self.lower_bound)} and {len(self.num_gridpoints)}",
            )
        for i, (lb, ub) in enumerate(zip(self.lower_bound, self.upper_bound)):
            if ub <= lb:
                raise ValueError(
                    f"Dimension {i}: upper_bound ({ub}) must be > lower_bound ({lb})",
                )

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self.lower_bound)

    @property
    def sidelengths(self) -> tuple[float, ...]:
        """Side lengths in each dimension."""
        return tuple(ub - lb for lb, ub in zip(self.lower_bound, self.upper_bound))


class PatchGeometry(Enum):
    """Predefined multi-patch geometries."""

    SINGLE = "single"
    L_SHAPE = "l_shape"
    T_SHAPE = "t_shape"
    CROSS = "cross"


class PatchLibrary:
    """Factory for creating predefined multi-patch geometries."""

    @staticmethod
    def single(
        lower_bound: tuple[float, ...],
        upper_bound: tuple[float, ...],
        num_gridpoints: tuple[int, ...],
    ) -> dict[str, PatchDescriptor]:
        """Create a single-patch domain (default, backward compatible)."""
        return {
            "default": PatchDescriptor(
                name="default",
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                num_gridpoints=num_gridpoints,
            ),
        }

    @staticmethod
    def l_shape_2d(
        vertical_length: float = 2.0,
        horizontal_length: float = 2.0,
        thickness: float = 0.5,
        num_points_per_unit: int = 10,
    ) -> dict[str, PatchDescriptor]:
        """
        Create an L-shaped 2D domain with two patches forming a proper L.

        The L is formed with:
        - Vertical leg: bottom-left, spans full height
        - Horizontal leg: bottom-right, extends from vertical leg

        Parameters:
            vertical_length: Height of the vertical leg (y extent)
            horizontal_length: Width of the horizontal leg (x extent)
            thickness: Thickness of the vertical leg (x extent)
            num_points_per_unit: Grid points per unit length

        Returns:
            Dict mapping patch names to PatchDescriptor objects.

        Example:
            patches = PatchLibrary.l_shape_2d(
                vertical_length=2.0,
                horizontal_length=2.0,
                thickness=0.5,
                num_points_per_unit=10
            )

        Visualization:
            |-----| = horizontal leg
            |
            |
            | = vertical leg
        """
        nx_vert = int(thickness * num_points_per_unit) + 1
        ny_vert = int(vertical_length * num_points_per_unit) + 1
        nx_horiz = int(horizontal_length * num_points_per_unit) + 1
        ny_horiz = int(thickness * num_points_per_unit) + 1

        # Vertical leg: x ∈ [0, thickness], y ∈ [0, vertical_length]
        # Horizontal leg: x ∈ [0, horizontal_length], y ∈ [0, thickness]
        # The overlap at x ∈ [0, thickness], y ∈ [0, thickness] ensures
        # the full side of the vertical leg connects to the horizontal leg.
        return {
            "L_vertical": PatchDescriptor(
                name="L_vertical",
                lower_bound=(0.0, 0.0),
                upper_bound=(thickness, vertical_length),
                num_gridpoints=(nx_vert, ny_vert),
            ),
            "L_horizontal": PatchDescriptor(
                name="L_horizontal",
                lower_bound=(0.0, 0.0),
                upper_bound=(horizontal_length, thickness),
                num_gridpoints=(nx_horiz, ny_horiz),
            ),
        }

    @staticmethod
    def t_shape_2d(
        stem_length: float = 2.0,
        stem_thickness: float = 0.5,
        head_width: float = 1.5,
        head_height: float = 0.5,
        num_points_per_unit: int = 10,
    ) -> dict[str, PatchDescriptor]:
        """
        Create a T-shaped 2D domain with three patches.

        Parameters:
            stem_length: Length of the vertical stem
            stem_thickness: Thickness of the stem
            head_width: Width of the horizontal head
            head_height: Height of the head
            num_points_per_unit: Grid points per unit length

        Returns:
            Dict mapping patch names to PatchDescriptor objects.
        """
        nx_stem = int(stem_thickness * num_points_per_unit) + 1
        ny_stem = int(stem_length * num_points_per_unit) + 1
        nx_head = int(head_width * num_points_per_unit) + 1
        ny_head = int(head_height * num_points_per_unit) + 1

        stem_x_min = (head_width - stem_thickness) / 2

        return {
            "T_stem": PatchDescriptor(
                name="T_stem",
                lower_bound=(stem_x_min, 0.0),
                upper_bound=(stem_x_min + stem_thickness, stem_length),
                num_gridpoints=(nx_stem, ny_stem),
            ),
            "T_head": PatchDescriptor(
                name="T_head",
                lower_bound=(0.0, stem_length),
                upper_bound=(head_width, stem_length + head_height),
                num_gridpoints=(nx_head, ny_head),
            ),
        }

    @staticmethod
    def cross_2d(
        arm_length: float = 2.0,
        arm_thickness: float = 0.5,
        num_points_per_unit: int = 10,
    ) -> dict[str, PatchDescriptor]:
        """
        Create a cross (+) shaped 2D domain with four patches.

        Parameters:
            arm_length: Length of each arm from center
            arm_thickness: Thickness of each arm
            num_points_per_unit: Grid points per unit length

        Returns:
            Dict mapping patch names to PatchDescriptor objects.
        """
        nx_h = int(arm_length * num_points_per_unit) + 1
        ny_h = int(arm_thickness * num_points_per_unit) + 1
        nx_v = int(arm_thickness * num_points_per_unit) + 1
        ny_v = int(arm_length * num_points_per_unit) + 1

        center_x = arm_length / 2
        center_y = arm_length / 2

        return {
            "Cross_horizontal": PatchDescriptor(
                name="Cross_horizontal",
                lower_bound=(0.0, center_y - arm_thickness / 2),
                upper_bound=(arm_length, center_y + arm_thickness / 2),
                num_gridpoints=(nx_h, ny_h),
            ),
            "Cross_vertical": PatchDescriptor(
                name="Cross_vertical",
                lower_bound=(center_x - arm_thickness / 2, 0.0),
                upper_bound=(center_x + arm_thickness / 2, arm_length),
                num_gridpoints=(nx_v, ny_v),
            ),
        }

    @classmethod
    def create(
        cls,
        geometry: PatchGeometry | str,
        **kwargs: ConfigValue,
    ) -> dict[str, PatchDescriptor]:
        """
        Factory method to create patch geometries by name.

        Parameters:
            geometry: PatchGeometry enum or string name
            **kwargs: Geometry-specific parameters

        Returns:
            Dict mapping patch names to PatchDescriptor objects.

        Example:
            patches = PatchLibrary.create(
                PatchGeometry.L_SHAPE,
                vertical_length=2.0,
                horizontal_length=2.0,
                thickness=0.5,
            )
        """
        if isinstance(geometry, str):
            geometry = PatchGeometry(geometry.lower())

        if geometry == PatchGeometry.SINGLE:
            return cls.single(**kwargs)
        elif geometry == PatchGeometry.L_SHAPE:
            return cls.l_shape_2d(**kwargs)
        elif geometry == PatchGeometry.T_SHAPE:
            return cls.t_shape_2d(**kwargs)
        elif geometry == PatchGeometry.CROSS:
            return cls.cross_2d(**kwargs)
        else:
            raise ValueError(f"Unknown geometry: {geometry}")
