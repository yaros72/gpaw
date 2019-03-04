from __future__ import print_function

from ase import Atom, Atoms

from gpaw import GPAW
from gpaw.test import equal

try:
    calc = GPAW('NaCl.gpw')
    NaCl = calc.get_atoms()
    e = NaCl.get_potential_energy()
    niter = None
except IOError:
    h = 0.21  # gridspacing
    a = [6.5, 6.5, 7.7]  # unit cell
    d = 2.3608  # experimental bond length

    NaCl = Atoms([Atom('Na', [0, 0, 0]),
                  Atom('Cl', [0, 0, d])],
                 pbc=False, cell=a)
    NaCl.center()
    calc = GPAW(h=h, xc='LDA', nbands=5,
                setups={'Na': '1'},
                convergence={'eigenstates': 1e-6}, spinpol=1)

    NaCl.set_calculator(calc)
    e = NaCl.get_potential_energy()
    niter = calc.get_number_of_iterations()
    calc.write('NaCl.gpw')

equal(e, -4.9066211328768246, 0.001)

dv = NaCl.get_volume() / calc.get_number_of_grid_points().prod()
nt1 = calc.get_pseudo_density(gridrefinement=1)
Zt1 = nt1.sum() * dv
nt2 = calc.get_pseudo_density(gridrefinement=2)
Zt2 = nt2.sum() * dv / 8
print('Integral of pseudo density:', Zt1, Zt2)
equal(Zt1, Zt2, 1e-12)

for gridrefinement in [1, 2, 4]:
    n = calc.get_all_electron_density(gridrefinement=gridrefinement)
    Z = n.sum() * dv / gridrefinement**3
    print('Integral of all-electron density:', Z)
    equal(Z, 28, 1e-5)
