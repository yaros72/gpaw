from ase import Atoms
from ase.build import bulk
from gpaw import GPAW, FermiDirac, PW

cell = bulk('Ga', 'fcc', a=5.68).cell
a = Atoms('GaAs', cell=cell, pbc=True,
          scaled_positions=((0, 0, 0), (0.25, 0.25, 0.25)))

calc = GPAW(mode=PW(600),
            xc='LDA',
            occupations=FermiDirac(width=0.01),
            convergence={'density': 1.e-6},
            symmetry='off',
            kpts={'size': (2, 2, 2), 'gamma': True},
            txt='gs_GaAs.txt')

a.set_calculator(calc)
a.get_potential_energy()
calc.write('GaAs.gpw', mode='all')
