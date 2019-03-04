import numpy as np
from ase.build import molecule
from ase.data.vdw import vdw_radii
from gpaw import GPAW
from gpaw.cluster import Cluster
from gpaw.test import equal
from gpaw.solvation import (SolvationGPAW, EffectivePotentialCavity,
                            Power12Potential, LinearDielectric)

SKIP_REF_CALC = not not True

energy_eps = 0.0005 / 8.
forces_eps = 2e-5

h = 0.3
vac = 3.0
u0 = .180
T = 298.15
vdw_radii = vdw_radii.copy()
vdw_radii[1] = 1.09


def atomic_radii(atoms):
    return [vdw_radii[n] for n in atoms.numbers]

atoms = Cluster(molecule('H2O'))
atoms.minimal_box(vac, h)

convergence = {
    'energy': energy_eps,
    'forces': forces_eps ** 2,  # Force error is squared
    'density': 10.,
    'eigenstates': 10.,
}

if not SKIP_REF_CALC:
    atoms.calc = GPAW(xc='LDA', h=h, convergence=convergence)
    Eref = atoms.get_potential_energy()
    print(Eref)
    Fref = atoms.get_forces()
    print(Fref)
else:
    # setups: 0.9.11271, same settings as above
    Eref = -11.9879852185

    Fref = np.array(
        [[1.77087917e-12, -2.38046360e-12, -6.05015925e+00],
         [7.91317656e-14, 1.61479184e+00, 6.87595580e-02],
         [2.62581472e-13, -1.61479184e+00, 6.87595580e-02]])

atoms.calc = SolvationGPAW(
    xc='LDA', h=h, convergence=convergence,
    cavity=EffectivePotentialCavity(
        effective_potential=Power12Potential(atomic_radii=atomic_radii, u0=u0),
        temperature=T
    ),
    dielectric=LinearDielectric(epsinf=1.0),
)
Etest = atoms.get_potential_energy()
Eeltest = atoms.calc.get_electrostatic_energy()
Ftest = atoms.get_forces()
equal(Etest, Eref, energy_eps * atoms.calc.get_number_of_electrons())
equal(Ftest, Fref, forces_eps)
equal(Eeltest, Etest)
