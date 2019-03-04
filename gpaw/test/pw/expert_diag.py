from __future__ import print_function

import numpy as np

from ase.build import bulk
from gpaw import GPAW, PW
from gpaw.test import equal
from gpaw.mpi import world

# This test is asserting that the expert diagonalization
# routine gives the same result as the non-expert version
# in terms of eigenvalues and wavefunctions

wfs_e = []
for i, expert in enumerate([True, False]):
    si = bulk('Si')
    name = 'si_{0:d}'.format(i)
    si.center()
    calc = GPAW(mode=PW(120), kpts=(1, 1, 2),
                experimental={'niter_fixdensity': 2},
                eigensolver='rmm-diis',
                parallel={'domain': 1},
                symmetry='off', txt=name + '.txt')
    si.set_calculator(calc)
    si.get_potential_energy()
    calc.diagonalize_full_hamiltonian(expert=expert, nbands=48)
    string = name + '.gpw'
    calc.write(string, 'all')
    wfs_e.append(calc.wfs)

ref_revision = 133324
# Test against values from reference revision
epsn_n = np.array([-0.2108572, 0.05657162, 0.17660414, 0.1766701, 0.2644755])
wfsold_G = np.array([3.85175220 + 6.63095137e-14j,
                     27.98536984 - 4.29697033e-14j,
                     -1.95105925 + 1.38780689e-14j,
                     -0.07363307 - 2.90717943e-16j,
                     -0.32384296 + 3.57917155e-15j])


kpt_u = wfs_e[0].kpt_u
for kpt in kpt_u:
    if kpt.k == 0:
        psit = kpt.psit_nG[1, 0:5].copy()
        if wfsold_G[0] * psit[0] < 0:
            psit *= -1.
        if world.rank == 0:
            print('eps', repr(kpt.eps_n[0:5]))
            print('psit', repr(psit))
            assert np.allclose(epsn_n, kpt.eps_n[0:5], 1e-5), \
                'Eigenvalues have changed since rev. #%d' % ref_revision
            assert np.allclose(wfsold_G, psit, 1e-5), \
                'Wavefunctions have changed rev. #%d' % ref_revision

# Check that expert={True, False} give
# same result
while len(wfs_e) > 1:
    wfs = wfs_e.pop()
    for wfstmp in wfs_e:
        for kpt, kpttmp in zip(wfs.kpt_u, wfstmp.kpt_u):
            for m, (psi_G, eps) in enumerate(zip(kpt.psit_nG, kpt.eps_n)):
                # Have to do like this if bands are degenerate
                booleanarray = np.abs(kpttmp.eps_n - eps) < 1e-10
                inds = np.argwhere(booleanarray)
                count = len(inds)
                assert count > 0, 'Difference between eigenvalues!'

                psitmp_nG = kpttmp.psit_nG[inds][:, 0, :]
                fidelity = 0
                for psitmp_G in psitmp_nG:
                    fidelity += (np.abs(np.dot(psitmp_G.conj(), psi_G))**2 /
                                 np.dot(psitmp_G.conj(), psitmp_G) /
                                 np.dot(psi_G.conj(), psi_G))

                equal(fidelity, 1, 1e-10, 'Difference between wfs!')
