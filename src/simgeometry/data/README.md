# simgeometry.data

Reference geometry shipped with the package.

- `w7x_desc_boundary.json`: the plasma boundary (the last closed flux surface, rho = 1) of the
  W7-X standard-configuration example equilibrium of [DESC](https://github.com/PlasmaControl/DESC)
  (`desc.examples.get("W7-X")`, DESC 0.17.3, MIT license), in ConStellaration's
  `boundary.json` layout (`r_cos`, `z_sin`, `n_field_periods`; M = N = 12), converted with
  `plasma_plots.coils.boundary_from_desc`. Loaded by `simgeometry.data.w7x_boundary`.

It is not W7-X's first wall: that geometry (heat shield, baffles, divertor targets, NBI
armour) is IPP's CAD and not public. `simgeometry.data.w7x_like_wall` offsets the boundary
by a uniform gap as a stand-in.
