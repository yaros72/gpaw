"""
tests solvation parameters as in

Damian A. Scherlis, Jean-Luc Fattebert, Francois Gygi,
Matteo Cococcioni and Nicola Marzari,
J. Chem. Phys. 124, 074103, 2006
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
    ElDensity
)

SKIP_VAC_CALC = True

h = 0.24
vac = 4.0

epsinf = 78.36
rho0 = 0.00078 / Bohr ** 3
beta = 1.3
st = 72. * 1e-3 * Pascal * m
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
        density=ElDensity(),
        surface_calculator=GradientSurface()),
    dielectric=LinearDielectric(epsinf=epsinf),
    interactions=[SurfaceInteraction(surface_tension=st)])
Ewater = atoms.get_potential_energy()
assert atoms.calc.get_number_of_iterations() < 40
atoms.get_forces()
DGSol = (Ewater - Evac) / (kcal / mol)
print('Delta Gsol: %s kcal / mol' % DGSol)

equal(DGSol, -75., 10.)
