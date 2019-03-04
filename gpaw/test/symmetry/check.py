"""Example from the gpaw-dev ML.

https://listserv.fysik.dtu.dk/pipermail/gpaw-users/2016-January/003864.html
"""
from __future__ import division
from ase import Atoms
from ase.test import must_raise
from gpaw.symmetry import atoms2symmetry

a = 3.9
atoms = Atoms('Ge2Si4',
              cell=[[a, 0, 0], [0, a, 0], [-a / 2, -a / 2, a * 3 / 2**0.5]],
              scaled_positions=[[2 / 3, 2 / 3, 1 / 3],
                                [11 / 12, 5 / 12, 5 / 6],
                                [0, 0, 0],
                                [1 / 4, 3 / 4, 1 / 2],
                                [1 / 3, 1 / 3, 2 / 3],
                                [7 / 12, 1 / 12, 1 / 6]],
              pbc=True)
sym = atoms2symmetry(atoms)
with must_raise(ValueError):
    sym.check_grid((20, 20, 40))
