from ase.build import mx2
from gpaw import GPAW, PW, FermiDirac

calc = GPAW(mode=PW(400),
            xc='PBE',
            nbands=70,
            convergence={'bands': -20},
            setups={'Mo': '6'},
            parallel={'band': 1, 'domain': 1},
            occupations=FermiDirac(width=0.001),
            kpts={'size': (42, 42, 1), 'gamma': True},
            txt='gs_MoS2.txt')

slab = mx2(formula='MoS2', a=3.16, thickness=3.17, vacuum=5.0)
slab.set_calculator(calc)
slab.get_potential_energy()

calc.write('gs_MoS2.gpw', mode='all')
