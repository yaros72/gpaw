"""
Calculate the magnetic response in iron using ALDA.
"""

# Workflow modules
import numpy as np

# Script modules
import time

from ase.build import bulk
from ase.dft.kpoints import monkhorst_pack
from ase.parallel import parprint

from gpaw import GPAW, PW
from gpaw.response.tms import TransverseMagneticSusceptibility
from gpaw.test import findpeak, equal
from gpaw.mpi import world

# ------------------- Inputs ------------------- #

# Part 1: ground state calculation
xc = 'LDA'
kpts = 4
nb = 6
pw = 300
a = 2.867
mm = 2.21

# Part 2: magnetic response calculation
q_qc = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5 / 2.]]  # Two q-points along G-N path
frq_qw = [np.linspace(0.150, 0.350, 26), np.linspace(0.300, 0.500, 26)]
Kxc = 'ALDA'
ecut = 300
eta = 0.01

# ------------------- Script ------------------- #

# Part 1: ground state calculation

t1 = time.time()

Febcc = bulk('Fe', 'bcc', a=a)
Febcc.set_initial_magnetic_moments([mm])

calc = GPAW(xc=xc,
            mode=PW(pw),
            kpts=monkhorst_pack((kpts, kpts, kpts)),
            nbands=nb,
            txt=None)

Febcc.set_calculator(calc)
Febcc.get_potential_energy()
calc.write('Fe', 'all')
t2 = time.time()

# Part 2: magnetic response calculation
for q in range(2):
    tms = TransverseMagneticSusceptibility(calc='Fe',
                                           frequencies=frq_qw[q],
                                           eta=eta,
                                           ecut=ecut,
                                           txt='iron_dsus_%d.out' % (q + 1))

    chiM0_w, chiM_w = tms.get_dynamic_susceptibility(q_c=q_qc[q], xc=Kxc,
                                                     filename='iron_dsus'
                                                     + '_%d.csv' % (q + 1))

t3 = time.time()

parprint('Ground state calculation took', (t2 - t1) / 60, 'minutes')
parprint('Excited state calculation took', (t3 - t2) / 60, 'minutes')

world.barrier()

# Part 3: identify peaks in scattering function and compare to test values
d1 = np.loadtxt('iron_dsus_1.csv', delimiter=', ')
d2 = np.loadtxt('iron_dsus_2.csv', delimiter=', ')

wpeak1, Ipeak1 = findpeak(d1[:, 0], - d1[:, 4])
wpeak2, Ipeak2 = findpeak(d2[:, 0], - d2[:, 4])

mw1 = (wpeak1 + d1[0, 0]) * 1000
mw2 = (wpeak2 + d2[0, 0]) * 1000

test_mw1 = 253.8  # meV
test_mw2 = 399.3  # meV
test_Ipeak1 = 66.5595189306  # a.u.
test_Ipeak2 = 54.9375540086  # a.u.

# Magnon peak:
equal(test_mw1, mw1, eta * 700)
equal(test_mw2, mw2, eta * 800)

# Scattering function intensity:
equal(test_Ipeak1, Ipeak1, 5)
equal(test_Ipeak2, Ipeak2, 5)
