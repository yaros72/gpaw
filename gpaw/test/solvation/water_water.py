from gpaw import GPAW
from gpaw.cluster import Cluster
from gpaw.test import equal
from ase.build import molecule
from ase.units import mol, kcal
from gpaw.solvation import (
    SolvationGPAW,
    get_HW14_water_kwargs,
)

SKIP_VAC_CALC = True

h = 0.24
vac = 4.0
convergence = {
    'energy': 0.05 / 8.,
    'density': 10.,
    'eigenstates': 10.,
}

atoms = Cluster(molecule('H2O'))
atoms.minimal_box(vac, h)

if not SKIP_VAC_CALC:
    atoms.calc = GPAW(xc='PBE', h=h, convergence=convergence)
    Evac = atoms.get_potential_energy()
    print(Evac)
else:
    # h=0.24, vac=4.0, setups: 0.9.11271, convergence: only energy 0.05 / 8
    Evac = -14.857414548

atoms.calc = SolvationGPAW(
    xc='PBE', h=h, convergence=convergence,
    **get_HW14_water_kwargs()
)
Ewater = atoms.get_potential_energy()
Eelwater = atoms.calc.get_electrostatic_energy()
Esurfwater = atoms.calc.get_solvation_interaction_energy('surf')
atoms.get_forces()
DGSol = (Ewater - Evac) / (kcal / mol)
print('Delta Gsol: %s kcal / mol' % DGSol)

equal(DGSol, -6.3, 2.)
equal(Ewater, Eelwater + Esurfwater, 1e-14)
