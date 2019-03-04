from __future__ import division
from ase import Atoms
from ase.lattice.hexagonal import Hexagonal
from gpaw import GPAW, PW, FermiDirac

calc = GPAW(mode=PW(600),
            xc='PBE',
            setups={'W': '6'},
            parallel={'band': 1, 'domain': 1},
            occupations=FermiDirac(width=0.01),
            kpts=(15, 15, 1),
            txt='WS2_gs.txt',
            )

a = 3.160
c = 12.0

cell = Hexagonal(symbol='W', latticeconstant={'a': a, 'c': c}).get_cell()
layer = Atoms(symbols='WS2', cell=cell, pbc=(1, 1, 1),
              scaled_positions=[(0, 0, 0),
                                (2 / 3, 1 / 3, 0.3),
                                (2 / 3, 1 / 3, -0.3)])

layer.center(axis=2)
layer.set_pbc(True)
pos = layer.get_positions()
pos[1][2] = pos[0][2] + 3.172 / 2
pos[2][2] = pos[0][2] - 3.172 / 2
layer.set_positions(pos)
layer.set_calculator(calc)
layer.get_potential_energy()
calc.write('WS2_gs.gpw')
