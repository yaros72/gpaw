"""Test calculation for unoccupied states using IVOs."""
from __future__ import print_function

from ase.build import molecule
from gpaw.cluster import Cluster
from gpaw import GPAW, KohnShamConvergenceError, FermiDirac
from gpaw.eigensolvers import CG, RMMDIIS

calc_parms = [
    {'xc': 'PBE0:unocc=True',
     'eigensolver': RMMDIIS(niter=5),
     'convergence': {
         'energy': 0.005,
         'bands': -2,
         'eigenstates': 1e-4,
         'density': 1e-3}},
    {'xc': 'PBE0:excitation=singlet',
     'convergence': {
         'energy': 0.005,
         'bands': 'occupied',
         'eigenstates': 1e-4,
         'density': 1e-3}}]


def calc_me(atoms, nbands):
    """Do the calculation."""
    molecule_name = atoms.get_chemical_formula()
    atoms.set_initial_magnetic_moments([-1.0, 1.0])
    fname = '.'.join([molecule_name, 'PBE-SIN'])
    calc = GPAW(h=0.25,
                xc='PBE',
                eigensolver=CG(niter=5),
                nbands=nbands,
                txt=fname + '.log',
                occupations=FermiDirac(0.0, fixmagmom=True),
                convergence={
                    'energy': 0.005,
                    'bands': nbands,
                    'eigenstates': 1e-4,
                    'density': 1e-3})
    atoms.set_calculator(calc)
    try:
        atoms.get_potential_energy()
    except KohnShamConvergenceError:
        pass
    if calc.scf.converged:
        for calcp in calc_parms:
            calc.set(**calcp)
            try:
                calc.calculate(system_changes=[])
            except KohnShamConvergenceError:
                break
        if calc.scf.converged:
            calc.write(fname + '.gpw', mode='all')


loa = Cluster(molecule('NaCl'))
loa.minimal_box(border=6.0, h=0.25, multiple=16)
loa.center()
loa.translate([0.001, 0.002, 0.003])
nbands = 25
calc_me(loa, nbands)
