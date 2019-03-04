import numpy as np
from pathlib import Path
from gpaw.mpi import world
from gpaw.response.bse import BSE
from gpaw.response.df import DielectricFunction

ecut = 50
eshift = 0.8
eta = 0.2

df = DielectricFunction('gs_Si.gpw',
                        ecut=ecut,
                        frequencies=np.linspace(0, 10, 1001),
                        nbands=8,
                        intraband=False,
                        hilbert=False,
                        eta=eta,
                        eshift=eshift,
                        txt='rpa_Si.txt')

df.get_dielectric_function(filename='eps_rpa_Si.csv')

bse = BSE('gs_Si.gpw',
          ecut=ecut,
          valence_bands=range(0, 4),
          conduction_bands=range(4, 8),
          nbands=50,
          eshift=eshift,
          mode='BSE',
          write_v=True,
          integrate_gamma=0,
          txt='bse_Si.txt')

bse.get_dielectric_function(filename='eps_bse_Si.csv',
                            eta=eta,
                            write_eig='bse_Si_eig.dat',
                            w_w=np.linspace(0.0, 10.0, 10001))

if world.rank == 0:
    Path('v_TS.ulm').unlink()
