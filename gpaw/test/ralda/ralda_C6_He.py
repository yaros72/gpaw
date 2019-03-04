from ase import Atoms
from gpaw import GPAW, PW
from gpaw.mpi import serial_comm
from gpaw.test import equal
from gpaw.xc.fxc_correlation_energy import FXCCorrelation

ecut = 50

He = Atoms('He')
He.center(vacuum=1.0)

calc = GPAW(mode=PW(force_complex_dtype=True),
            xc='PBE',
            communicator=serial_comm)
He.set_calculator(calc)
He.get_potential_energy()
calc.diagonalize_full_hamiltonian()

ralda = FXCCorrelation(calc, xc='rALDA')
C6_ralda, C6_0 = ralda.get_C6_coefficient(ecut=ecut,
                                          direction=2)

equal(C6_0, 1.772, 0.01)
equal(C6_ralda, 1.609, 0.01)
