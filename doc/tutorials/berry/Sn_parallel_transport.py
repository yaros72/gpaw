import os
from gpaw.berryphase import parallel_transport
from gpaw import GPAW
import gpaw.mpi as mpi

calc = GPAW('gs_Sn.gpw',
            kpts={'size': (7, 200, 1), 'gamma': True},
            fixdensity=True,
            symmetry='off',
            txt='Sn_berry.txt')
calc.get_potential_energy()
calc.write('gs_berry.gpw', mode='all')

parallel_transport('gs_berry.gpw', direction=0, name='7x200')

if mpi.world.rank == 0:
    os.system('rm gs_berry.gpw')
