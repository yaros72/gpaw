from ase.build import bulk
from gpaw import GPAW, PW, FermiDirac

calc = GPAW(mode=PW(600),
            xc='PBE',
            occupations=FermiDirac(0.01),
            spinpol=True,
            kpts=[8, 8, 8],
            parallel={'band': 1, 'domain': 1},
            txt='Fe_gs.txt')

bulk = bulk('Fe', 'bcc', a=2.87)
bulk.set_initial_magnetic_moments([1.0])
bulk.set_calculator(calc)
bulk.get_potential_energy()

calc.write('Fe_gs.gpw', mode='all')
