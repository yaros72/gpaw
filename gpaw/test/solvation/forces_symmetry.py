from ase import Atoms
from ase.data.vdw import vdw_radii
from gpaw.test import equal
from ase.units import Pascal, m
from gpaw.solvation import (
    SolvationGPAW,
    EffectivePotentialCavity,
    Power12Potential,
    LinearDielectric,
    KB51Volume,
    GradientSurface,
    VolumeInteraction,
    SurfaceInteraction,
    LeakedDensityInteraction
)
import numpy as np

h = 0.2
d = 2.5
min_vac = 4.0
u0 = .180
epsinf = 80.
T = 298.15
atomic_radii = lambda atoms: [vdw_radii[n] for n in atoms.numbers]

xy_cell = np.ceil((min_vac * 2.) / h / 8.) * 8. * h
z_cell = np.ceil((min_vac * 2. + d) / h / 8.) * 8. * h
atoms = Atoms(
    'NaCl', positions=(
        (xy_cell / 2., xy_cell / 2., z_cell / 2. - d / 2.),
        (xy_cell / 2., xy_cell / 2., z_cell / 2. + d / 2.)
    )
)
atoms.set_cell((xy_cell, xy_cell, z_cell))

atoms.calc = SolvationGPAW(
    xc='PBE', h=h, setups={'Na': '1'},
    cavity=EffectivePotentialCavity(
        effective_potential=Power12Potential(atomic_radii, u0),
        temperature=T,
        volume_calculator=KB51Volume(),
        surface_calculator=GradientSurface()
    ),
    dielectric=LinearDielectric(epsinf=epsinf),
    # parameters chosen to give ~ 1eV for each interaction
    interactions=[
        VolumeInteraction(pressure=-1e9 * Pascal),
        SurfaceInteraction(surface_tension=100. * 1e-3 * Pascal * m),
        LeakedDensityInteraction(voltage=10.)
    ]
)
F = atoms.calc.get_forces(atoms)

difference = F[0][2] + F[1][2]
print(difference)
equal(difference, .0, .02)  # gas phase is ~.007 eV / Ang
F[0][2] = F[1][2] = .0
print(np.abs(F))
equal(np.abs(F), .0, 1e-10)  # gas phase is ~1e-11 eV / Ang
