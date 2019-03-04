from __future__ import division
from math import sqrt
import numpy as np
from ase import Atoms
from gpaw import GPAW, PW, restart
from gpaw.test import equal

d = 3.0
atoms = Atoms('Na3',
              positions=[(0, 0, 0),
                         (0, 0, d),
                         (0, d * sqrt(3 / 4), d / 2)],
              magmoms=[1.0, 1.0, 1.0],
              cell=(3.5, 3.5, 4 + 2 / 3),
              pbc=True)


def test(atoms):
    e0 = atoms.get_potential_energy()
    f0 = atoms.get_forces()
    m0 = atoms.get_magnetic_moments()
    eig00 = atoms.calc.get_eigenvalues(spin=0)
    eig01 = atoms.calc.get_eigenvalues(spin=1)
    wf0 = atoms.calc.get_pseudo_wave_function(band=1)
    # Write the restart file(s):
    atoms.calc.write('tmp1.gpw')
    atoms.calc.write('tmp2.gpw', 'all')

    # Try restarting:
    atoms, calc = restart('tmp2.gpw', txt=None)
    wf1 = calc.get_pseudo_wave_function(band=1)
    e1 = atoms.get_potential_energy()
    f1 = atoms.get_forces()
    m1 = atoms.get_magnetic_moments()
    eig10 = calc.get_eigenvalues(spin=0)
    eig11 = calc.get_eigenvalues(spin=1)
    print(e0, e1)
    equal(e0, e1, 1e-10)
    print(f0, f1)
    for ff0, ff1 in zip(f0, f1):
        err = np.linalg.norm(ff0 - ff1)
        assert err <= 1e-10
    print(m0, m1)
    for mm0, mm1 in zip(m0, m1):
        equal(mm0, mm1, 1e-10)
    print('A', eig00, eig10)
    for eig0, eig1 in zip(eig00, eig10):
        equal(eig0, eig1, 1e-10)
    print('B', eig01, eig11)
    for eig0, eig1 in zip(eig01, eig11):
        equal(eig0, eig1, 1e-10)
    equal(abs(wf1 - wf0).max(), 0, 1e-14)

    # Check that after restart everything is writable
    calc.write('tmp3.gpw')
    calc.write('tmp4.gpw', 'all')


# Only a short, non-converged calcuation
conv = {'eigenstates': 1.24, 'energy': 2e-1, 'density': 1e-1}

for kwargs in [{'mode': PW(200)},
               {'h': 0.30}]:
    atoms.calc = GPAW(nbands=3,
                      setups={'Na': '1'},
                      convergence=conv,
                      txt=None,
                      **kwargs)
    test(atoms)
