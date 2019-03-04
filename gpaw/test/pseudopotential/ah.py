from ase.build import bulk
from gpaw import GPAW, PW
si = bulk('Si', 'diamond', a=5.5, cubic=not True)
si.calc = GPAW(mode=PW(200),
               setups='ah',
               kpts=(2, 2, 2))
si.get_forces()
si.get_stress()
si.calc.write('Si.gpw', 'all')
GPAW('Si.gpw')
