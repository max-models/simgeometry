"""Named collections with explicit overlap precedence, display styles and voltages."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from numbers import Real

import cunumpy as xp
import numpy as np
import scope_profiler as sp

from simgeometry._profiling import _PROFILE_TAGS
from simgeometry.operations import mask
from simgeometry.shapes import Shape, _component_major

#: A conductor voltage: a number, or a function of the time ``t`` returning one.
Voltage = float | Callable[[float], float]


@dataclass(frozen=True)
class DirichletValues:
    """The object and the voltage at each point of :meth:`GeometryCollection.dirichlet_values`.

    ``labels`` are the collection's labels (0 outside every object, ``i`` inside
    ``names[i - 1]``, the last listed object winning in overlaps). ``voltages`` is the
    voltage of that object at the requested time, NaN outside every object and inside
    objects without a voltage. Both have the shape of the grid, or ``(n,)`` for points.
    """

    labels: np.ndarray
    voltages: np.ndarray
    names: tuple[str, ...]

    @property
    def conductor(self) -> np.ndarray:
        """Where a point is inside a conductor, so its voltage is fixed."""
        return np.isfinite(self.voltages)


class GeometryCollection(Shape):
    """Named objects; last listed object wins labels in overlaps.

    Label 0 is background, labels 1..N follow insertion order. Styles map object
    names to plotting keyword dictionaries. Explicit bounds also permit an empty
    collection and define the default viewing/sampling region, without clipping
    object membership. Collection objects are references, not snapshots.

    ``voltages`` makes named objects conductors: it maps names to a voltage, either a
    number or a function of the time ``t`` (a ramped or RF electrode). Objects without
    a voltage are not conductors. :meth:`with_voltages` returns the same geometry with
    other voltages, :meth:`voltages_at` evaluates them at a time and
    :meth:`dirichlet_values` gives the voltage at points or grid nodes.

    The level set, when every object has one, is the minimum of theirs (the union).
    """

    level_set_exact = False

    def _checked_voltages(self, voltages):
        """``voltages`` as a dict in object order, checking names and values."""
        unknown = set(voltages) - set(self.names)
        if unknown:
            raise ValueError(f"Voltages name unknown objects: {sorted(unknown)}")
        for name, value in voltages.items():
            if isinstance(value, bool) or not (
                callable(value) or isinstance(value, Real)
            ):
                raise TypeError(
                    f"The voltage of {name!r} must be a number or a function of t, "
                    f"got {type(value).__name__}"
                )
        return {
            name: value if callable(value) else float(value)
            for name in self.names
            if (value := voltages.get(name)) is not None
        }

    def __init__(self, objects, *, bounds=None, styles=None, voltages=None):
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
        self.voltages = self._checked_voltages(voltages or {})
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

    @property
    def conductors(self):
        """Names of the objects with a voltage, in insertion order."""
        return tuple(self.voltages)

    def with_voltages(self, voltages=None, /, **overrides):
        """The same objects, bounds and styles with some voltages replaced.

        ``voltages`` (a mapping, for any name) and keyword arguments set the voltage of
        those objects; ``None`` removes it. The geometry is shared, not rebuilt, so
        each run of a parameter scan can set its own voltages cheaply::

            lens = electrodes.with_voltages(focus=500.0)
        """
        merged = {**self.voltages, **dict(voltages or {}), **overrides}
        return GeometryCollection(
            self.named_objects,
            bounds=(self.lower_bound, self.upper_bound),
            styles=self.styles,
            voltages={name: v for name, v in merged.items() if v is not None},
        )

    def voltages_at(self, t=0.0):
        """The voltage of every conductor at time ``t``, as ``{name: float}``."""
        return {
            name: float(value(t)) if callable(value) else value
            for name, value in self.voltages.items()
        }

    @sp.profile("simgeometry.dirichlet_values", tags=_PROFILE_TAGS)
    def dirichlet_values(self, where, *, t=0.0):
        """The label and the voltage of the object at each point, for Dirichlet nodes.

        ``where`` is a :class:`~simgeometry.grid.Grid` (results have its shape) or
        ``(ndim, n)`` points. Returns :class:`DirichletValues`; a point outside every
        conductor has a NaN voltage. Overlaps follow :meth:`labels`: the last listed
        object wins, even if it is not a conductor.
        """
        if hasattr(where, "node_points") and hasattr(where, "num_gridpoints"):
            points = where.node_points()
            shape = tuple(int(n) for n in where.num_gridpoints)
        else:
            points = where
            shape = None
        labels = self.labels(points)
        table = np.full(len(self.names) + 1, np.nan)
        voltages = self.voltages_at(t)
        for label, name in enumerate(self.names, start=1):
            table[label] = voltages.get(name, np.nan)
        voltages = table[labels]
        if shape is not None:
            labels, voltages = labels.reshape(shape), voltages.reshape(shape)
        return DirichletValues(labels, voltages, self.names)

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

    @property
    def has_level_set(self):
        """Whether every object has a level set."""
        return bool(self.objects) and all(
            getattr(obj, "has_level_set", False) for obj in self.objects
        )

    def level_set(self, points):
        """The minimum of the objects' level sets: the level set of their union."""
        if not self.has_level_set:
            raise NotImplementedError("Every collection object needs a level set")
        points = _component_major(points, self.ndim)
        values = self.objects[0].level_set(points)
        for obj in self.objects[1:]:
            values = xp.minimum(values, obj.level_set(points))
        return values

    def evaluate(self, *args, quantity="labels", **kwargs):
        return super().evaluate(*args, quantity=quantity, **kwargs)

    def __getitem__(self, name):
        return self.named_objects[name]
