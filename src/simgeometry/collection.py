"""Named collections with explicit overlap precedence and display styles."""

from __future__ import annotations

from collections.abc import Mapping

import cunumpy as xp
import numpy as np

from simgeometry.operations import mask
from simgeometry.shapes import Shape, _component_major


class GeometryCollection(Shape):
    """Named objects; last listed object wins labels in overlaps.

    Label 0 is background, labels 1..N follow insertion order. Styles map object
    names to plotting keyword dictionaries. Explicit bounds also permit an empty
    collection and define the default viewing/sampling region, without clipping
    object membership. Collection objects are references, not snapshots.
    """

    def __init__(self, objects, *, bounds=None, styles=None):
        if not isinstance(objects, Mapping):
            objects = {f"object_{i + 1}": obj for i, obj in enumerate(objects)}
        self.named_objects = dict(objects)
        if any(not isinstance(name, str) or not name for name in self.named_objects):
            raise ValueError("Object names must be nonempty strings")
        self.names = tuple(self.named_objects)
        self.objects = list(self.named_objects.values())
        self.styles = dict(styles or {})
        if set(self.styles) - set(self.names):
            raise ValueError("Style names must identify objects")
        if bounds is None:
            if not self.objects:
                raise ValueError("An empty collection needs bounds")
            bounds = (
                np.min([s.lower_bound for s in self.objects], axis=0),
                np.max([s.upper_bound for s in self.objects], axis=0),
            )
        ndim = len(bounds[0])
        if any(obj.ndim != ndim for obj in self.objects):
            raise ValueError("All collection objects must have the same dimension")
        self._inverted = any(obj.inverted for obj in self.objects)
        super().__init__(ndim, *bounds)

    def __getitem__(self, name):
        return self.named_objects[name]

    def labels(self, points):
        """Integer object identities for (ndim, n) points; last object wins."""
        points = np.asarray(xp.to_numpy(_component_major(points, self.ndim)), float)
        result = np.zeros(points.shape[1], dtype=np.int64)
        for label, obj in enumerate(self.objects, start=1):
            result[mask(obj, points)] = label
        return result

    def contains_points(self, points):
        return xp.asarray(self.labels(points) != 0)

    def contains(self, coords):
        return any(obj.contains(coords) for obj in self.objects)

    def evaluate(self, *args, quantity="labels", **kwargs):
        return super().evaluate(*args, quantity=quantity, **kwargs)
