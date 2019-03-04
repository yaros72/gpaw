from ase.parallel import parprint

from gpaw import GPAW
from gpaw.poisson import PoissonSolver
from gpaw.mpi import world, size

from gpaw.lrtddft2 import LrTDDFT2
from gpaw.lrtddft2.lr_communicators import LrCommunicators

# atoms, gs_calc = restart('r-methyl-oxirane.gpw')

dd_size = 2 * 2 * 2
eh_size = size // dd_size
assert eh_size * dd_size == size

lr_comms = LrCommunicators(world, dd_size, eh_size)

gs_calc = GPAW('r-methyl-oxirane.gpw',
               communicator=lr_comms.dd_comm,
               txt='lr_gs.txt',
               poissonsolver=PoissonSolver(eps=1e-20,
                                           remove_moment=(1 + 3 + 5)))

lr = LrTDDFT2('lrtddft2',
              gs_calc,
              fxc='LDA',
              min_occ=0,            # usually zero
              max_occ=15,           # a few above LUMO
              min_unocc=10,         # a few before HOMO
              max_unocc=20,         # number of converged states
              max_energy_diff=8.0,  # eV
              recalculate=None,
              lr_communicators=lr_comms,
              txt='-')

# This is the expensive part!
lr.calculate()

# transitions
(w, S, R, Sx, Sy, Sz) = lr.get_transitions('trans.dat')
# spectrum
lr.get_spectrum('spectrum.dat', 0, 10.)


parprint('')
parprint(' %5s => %5s  contribution' % ('occ', 'unocc'))
f2 = lr.lr_transitions.get_transition_contributions(0)
for (ip, val) in enumerate(f2):
    if (val > 1e-3):
        parprint(' %5d => %5d  %lf %%\n ' %
                 (lr.ks_singles.kss_list[ip].occ_ind, lr.ks_singles.kss_list[ip].unocc_ind, val / 2. * 100))
