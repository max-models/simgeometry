"""The scope-profiler tag of simgeometry's own profiling regions.

``simgeometry/__init__.py`` registers the tag as opt-in, so these regions are off
unless an application enables ``"simgeometry"`` (see the README).
"""

#: Tags of every ``simgeometry.<operation>`` region.
_PROFILE_TAGS = ("simgeometry",)
