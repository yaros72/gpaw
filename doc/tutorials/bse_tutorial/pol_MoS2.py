import numpy as np
from gpaw.response.bse import BSE
from gpaw.response.df import DielectricFunction

ecut = 50
eshift = 0.8
eta = 0.05

df = DielectricFunction('gs_MoS2.gpw',
                        ecut=ecut,
                        frequencies=np.linspace(0, 5, 1001),
                        nbands=50,
                        intraband=False,
                        hilbert=False,
                        eta=eta,
                        eshift=eshift,
                        txt='rpa_MoS2.txt')

df.get_polarizability(filename='pol_rpa_MoS2.csv',
                      pbc=[True, True, False])

bse = BSE('gs_MoS2.gpw',
          spinors=True,
          ecut=ecut,
          valence_bands=[8],
          conduction_bands=[9],
          nbands=50,
          eshift=eshift,
          mode='BSE',
          txt='bse_MoS2.txt')

bse.get_polarizability(filename='pol_bse_MoS2.csv',
                       eta=eta,
                       pbc=[True, True, False],
                       write_eig='bse_MoS2_eig.dat',
                       w_w=np.linspace(0, 5, 5001))

bse = BSE('gs_MoS2.gpw',
          spinors=True,
          ecut=ecut,
          valence_bands=[8],
          conduction_bands=[9],
          truncation='2D',
          nbands=50,
          eshift=eshift,
          mode='BSE',
          txt='bse_MoS2_trun.txt')

bse.get_polarizability(filename='pol_bse_MoS2_trun.csv',
                       eta=eta,
                       pbc=[True, True, False],
                       write_eig='bse_MoS2_eig_trun.dat',
                       w_w=np.linspace(0, 5, 5001))
