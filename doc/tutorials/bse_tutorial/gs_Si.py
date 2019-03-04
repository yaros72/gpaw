from ase import Atoms
from ase.build import bulk
from gpaw import GPAW
from gpaw.wavefunctions.pw import PW
from gpaw.occupations import FermiDirac

cell = bulk('Si', 'fcc', a=5.421).get_cell()
a = Atoms('Si2', cell=cell, pbc=True,
          scaled_positions=((0, 0, 0), (0.25, 0.25, 0.25)))

calc = GPAW(mode=PW(400),
            xc='PBE',
            occupations=FermiDirac(width=0.001),
            parallel={'domain': 1, 'band': 1},
            kpts={'size': (8, 8, 8), 'gamma': True},
            txt='gs_Si.txt')

a.set_calculator(calc)
a.get_potential_energy()

calc.diagonalize_full_hamiltonian(nbands=100)
calc.write('gs_Si.gpw', mode='all')
