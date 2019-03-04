from ase.build import bulk
from gpaw import GPAW, PW, FermiDirac

a = bulk('Pt', 'fcc', a=3.92)

calc = GPAW(mode=PW(600),
            xc='PBE',
            nbands=20,
            parallel={'domain': 1, 'band': 1},
            occupations=FermiDirac(width=0.01),
            convergence={'density': 1.e-6},
            kpts=[8, 8, 8],
            txt='Pt_gs.txt')

a.set_calculator(calc)
a.get_potential_energy()

calc.write('Pt_gs.gpw', mode='all')
