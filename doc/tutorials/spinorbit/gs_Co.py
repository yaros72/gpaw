from ase.build import bulk
from gpaw import GPAW, PW, FermiDirac

calc = GPAW(mode=PW(600),
            xc='PBE',
            occupations=FermiDirac(0.01),
            kpts=(32, 32, 16),
            symmetry='off',
            parallel={'band': 1, 'domain': 1},
            txt='gs_Co.txt')

bulk = bulk('Co', 'hcp')
bulk.set_initial_magnetic_moments([1.0, 1.0])
bulk.set_calculator(calc)
bulk.get_potential_energy()

calc.write('gs_Co.gpw', mode='all')
