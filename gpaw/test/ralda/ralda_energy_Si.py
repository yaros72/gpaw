import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.dft.kpoints import monkhorst_pack

from gpaw import GPAW, FermiDirac
from gpaw.mpi import serial_comm
from gpaw.test import equal
from gpaw.xc.rpa import RPACorrelation
from gpaw.xc.fxc import FXCCorrelation

a0 = 5.43
cell = bulk('Si', 'fcc', a=a0).get_cell()
Si = Atoms('Si2', cell=cell, pbc=True,
           scaled_positions=((0,0,0), (0.25,0.25,0.25)))

kpts = monkhorst_pack((2,2,2))
kpts += np.array([1/4., 1/4., 1/4.])

calc = GPAW(mode='pw',
            kpts=kpts,
            occupations=FermiDirac(0.001),
            communicator=serial_comm)
Si.set_calculator(calc)
E = Si.get_potential_energy()
calc.diagonalize_full_hamiltonian(nbands=50)

rpa = RPACorrelation(calc)
E_rpa1 = rpa.calculate(ecut=[25, 50])

fxc = FXCCorrelation(calc, xc='RPA', nlambda=16)
E_rpa2 = fxc.calculate(ecut=[25, 50])

fxc = FXCCorrelation(calc, xc='rALDA', unit_cells=[1,1,2])
E_ralda = fxc.calculate(ecut=[25, 50])

fxc = FXCCorrelation(calc, xc='rAPBE', unit_cells=[1,1,2])
E_rapbe = fxc.calculate(ecut=[25, 50])

fxc = FXCCorrelation(calc, xc='rALDA', av_scheme='wavevector')
E_raldawave = fxc.calculate(ecut=[25, 50])

fxc = FXCCorrelation(calc, xc='rAPBE', av_scheme='wavevector')
E_rapbewave = fxc.calculate(ecut=[25, 50])

fxc = FXCCorrelation(calc, xc='JGMs', av_scheme='wavevector',Eg=3.1,nlambda=2)
E_JGMs = fxc.calculate(ecut=[25, 50])

fxc = FXCCorrelation(calc, xc='CP_dyn', av_scheme='wavevector',nfrequencies=2,nlambda=2)
E_CPdyn = fxc.calculate(ecut=[25, 50])

equal(E_rpa1[-1], E_rpa2[-1], 0.01)
equal(E_rpa2[-1], -12.6495, 0.001)
equal(E_ralda[-1], -11.3817, 0.001)
equal(E_rapbe[-1], -11.1640, 0.001)
equal(E_raldawave[-1], -11.0910, 0.001)
equal(E_rapbewave[-1],-10.8336, 0.001)
equal(E_JGMs[-1], -11.2561, 0.001)
equal(E_CPdyn[-1],-7.8640,0.001)
