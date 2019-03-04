# http://listserv.fysik.dtu.dk/pipermail/gpaw-developers/2014-February/004374.html
from __future__ import print_function
from gpaw import GPAW, PoissonSolver
from gpaw.test import equal
from ase.build import molecule


for i in range(12):
    mol = molecule('H2')
    mol.center(vacuum=1.5)
    calc = GPAW(h=0.3, nbands=2, mode='lcao', txt=None, basis='sz(dzp)',
                poissonsolver=PoissonSolver(eps=17.), xc='oldLDA')

    def stop():
        calc.scf.converged = True

    calc.attach(stop, 1)
    mol.set_calculator(calc)
    e = mol.get_potential_energy()
    if i == 0:
        eref = e
    if calc.wfs.world.rank == 0:
        print(repr(e))
    equal(e - eref, 0, 1.e-12)
