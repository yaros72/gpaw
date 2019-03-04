"""
tests solvation parameters as in

V. M. Sanchez, M. Sued, and D. A. Scherlis,
The Journal of Chemical Physics, vol. 131, no. 17, p. 174108, 2009
"""

from gpaw import GPAW
from gpaw.cluster import Cluster
from gpaw.test import equal
from ase import Atoms
from ase.units import mol, kcal, Pascal, m, Bohr
from gpaw.solvation import (
    SolvationGPAW,
    FG02SmoothStepCavity,
    LinearDielectric,
    GradientSurface,
    SurfaceInteraction,
    SSS09Density)

SKIP_VAC_CALC = True

h = 0.24
vac = 4.0

epsinf = 78.36
rho0 = 1.0 / Bohr ** 3
beta = 2.4
st = 72. * 1e-3 * Pascal * m
atomic_radii = lambda atoms: [2.059]
convergence = {
    'energy': 0.05 / 8.,
    'density': 10.,
    'eigenstates': 10.,
}

atoms = Cluster(Atoms('Cl'))
atoms.minimal_box(vac, h)

if not SKIP_VAC_CALC:
    atoms.calc = GPAW(xc='PBE', h=h, charge=-1, convergence=convergence)
    Evac = atoms.get_potential_energy()
    print(Evac)
else:
    # h=0.24, vac=4.0, setups: 0.9.11271, convergence: only energy 0.05 / 8
    Evac = -3.83245253419

atoms.calc = SolvationGPAW(
    xc='PBE', h=h, charge=-1, convergence=convergence,
    cavity=FG02SmoothStepCavity(
        rho0=rho0, beta=beta,
        density=SSS09Density(atomic_radii=atomic_radii),
        surface_calculator=GradientSurface()
    ),
    dielectric=LinearDielectric(epsinf=epsinf),
    interactions=[SurfaceInteraction(surface_tension=st)]
)
Ewater = atoms.get_potential_energy()
assert atoms.calc.get_number_of_iterations() < 40
atoms.get_forces()
DGSol = (Ewater - Evac) / (kcal / mol)
print('Delta Gsol: %s kcal / mol' % DGSol)

equal(DGSol, -75., 10.)
