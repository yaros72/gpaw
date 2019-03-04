from ase.build import mx2
from gpaw import GPAW, PW, FermiDirac

calc = GPAW(mode=PW(600),
            xc='PBE',
            occupations=FermiDirac(width=0.001),
            kpts={'size': (6, 6, 1), 'gamma': True},
            txt='gs_Sn.txt')

slab = mx2(formula='SnSn2', a=4.67, thickness=1.7, vacuum=5.0)
del slab[-1]
slab.set_calculator(calc)
slab.get_potential_energy()

calc.write('gs_Sn.gpw')
