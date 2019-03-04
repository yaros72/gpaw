from ase import Atoms
from gpaw import GPAW, Mixer
from gpaw.test import equal

"""The purpose of this test is to make sure that wfs-dependent
functionals do not produce wrong results due to incorrect saving of
the wfs object when calc.set() is used.

It works by converging a calculation with (1, 2, 1) kpts, then setting
to (4, 1, 1) kpts and comparing.  That should yield the same result as
converging (4, 1, 1) right form the start.

This is just one example of the many inconsistencies that can be
caused by calls to calc.set().  That method should therefore only be
used with the utmost care."""

atoms = Atoms('HF',positions=[(0.0,0.0,0.0),(1.0,0.0,0.0)])
atoms.set_pbc((True,True,True))
atoms.set_cell((2.0,2.0,2.0))

def MGGA_fail():
    calc = GPAW(xc='TPSS', mixer=Mixer(0.5, 5, 50.0), eigensolver='cg',kpts=(1,2,1), convergence={'density':1e-8})
    atoms.set_calculator(calc)
    atoms.get_potential_energy()
    calc.set(kpts=(4,1,1))
    return atoms.get_potential_energy()

def MGGA_work():
    calc = GPAW(xc='TPSS', mixer=Mixer(0.5, 5, 50.0), eigensolver='cg', kpts=(4,1,1), convergence={'density':1e-8})

    atoms.set_calculator(calc)
    return atoms.get_potential_energy()

def GLLBSC_fail():
    calc = GPAW(xc='GLLBSC', mixer=Mixer(0.5, 5, 50.0), eigensolver='cg', kpts=(1,2,1), convergence={'density':1e-8})

    atoms.set_calculator(calc)
    atoms.get_potential_energy()
    calc.set(kpts=(4,1,1))
    return atoms.get_potential_energy()

def GLLBSC_work():
    calc = GPAW(xc='GLLBSC', mixer=Mixer(0.5, 5, 50.0), eigensolver='cg', kpts=(4,1,1), convergence={'density':1e-8})

    atoms.set_calculator(calc)
    return atoms.get_potential_energy()

a = GLLBSC_fail()
b = GLLBSC_work()
equal(a,b, 1e-5)
c = MGGA_fail()
d = MGGA_work()
equal(c,d, 1e-5)

