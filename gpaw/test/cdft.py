from ase import Atoms
import numpy as np
from gpaw import GPAW, FermiDirac, Davidson, Mixer
from gpaw.cdft.cdft import CDFT
from gpaw.cdft.cdft_coupling import CouplingParameters
from gpaw.mpi import size

distance = 2.5
sys = Atoms('He2', positions = ([0.,0.,0.],[0.,0.,distance]))
sys.center(3)
sys.set_pbc(False)
sys.set_initial_magnetic_moments([0.5,0.5])

calc_b = GPAW(h = 0.2,
              mode='fd',
              basis='dzp',
              charge=1,
              xc='PBE',symmetry='off',
              occupations = FermiDirac(0., fixmagmom = True),
              eigensolver = Davidson(3),
              spinpol = True,
              nbands = 4,
              mixer = Mixer(beta=0.25, nmaxold=3, weight=100.0),
              txt='He2+_final_%3.2f.txt' % distance,
              convergence={'eigenstates':1.0e-4,'density':1.0e-1, 'energy':1e-1,'bands':4})


cdft_b = CDFT(calc = calc_b,
              atoms=sys,
              charge_regions = [[1]],
              charges = [1],
              charge_coefs = [2.7],
              method = 'L-BFGS-B',
              txt = 'He2+_final_%3.2f.cdft' % distance,
              minimizer_options={'gtol':0.01})
sys.set_calculator(cdft_b)
sys.get_potential_energy()

if size == 1:
    coupling = CouplingParameters(cdft_b, cdft_b, AE = False)
    overlaps = coupling.get_pair_density_matrix(cdft_b.calc, cdft_b.calc)[0]
    print(overlaps)

    for i in [0,1,2]:
        assert (np.isclose(np.real(overlaps[0,i,i]),1.))
