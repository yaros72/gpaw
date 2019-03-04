from __future__ import print_function
import numpy as np
from ase.build import bulk
from gpaw import GPAW
from gpaw.response.df import DielectricFunction
from gpaw.response.bse import BSE
from gpaw.test import findpeak, equal

GS = 1
df = 1
bse = 1
check_spectrum = 1

if GS:
    a = 4.043
    atoms = bulk('Al', 'fcc', a=a)
    calc = GPAW(mode='pw',
                kpts={'size': (4, 4, 4), 'gamma': True},
                xc='LDA',
                nbands=4,
                convergence={'bands': 'all'})
    
    atoms.set_calculator(calc)
    atoms.get_potential_energy()
    calc.write('Al.gpw', 'all')

q_c = np.array([0.25, 0.0, 0.0])
w_w = np.linspace(0, 24, 241)
eta = 0.2
ecut = 50
if bse:
    bse = BSE('Al.gpw',
              valence_bands=range(4),
              conduction_bands=range(4),
              mode='RPA',
              nbands=4,
              ecut=ecut,
              write_h=False,
              write_v=False,
              )
    bse_w = bse.get_eels_spectrum(filename=None,
                                  q_c=q_c,
                                  w_w=w_w,
                                  eta=eta)[1]
    
if df:
    df = DielectricFunction(calc='Al.gpw',
                            frequencies=w_w,
                            eta=eta,
                            ecut=ecut,
                            hilbert=False)
    df_w = df.get_eels_spectrum(q_c=q_c, filename=None)[1]

if check_spectrum:
    w_ = 15.1423
    I_ = 25.4359
    wbse, Ibse = findpeak(w_w, bse_w)
    wdf, Idf = findpeak(w_w, df_w)
    equal(wbse, w_, 0.01)
    equal(wdf, w_, 0.01)
    equal(Ibse, I_, 0.1)
    equal(Idf, I_, 0.1)
