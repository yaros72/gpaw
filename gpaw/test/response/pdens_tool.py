"""
Calculate optical transition strengths.
"""

# General modules
import numpy as np

# Script modules
from ase.build import bulk

from gpaw import GPAW, PW, FermiDirac
from gpaw.response.tool import (get_bz_transitions,
                                get_chi0_integrand,
                                get_degeneracy_matrix,
                                get_individual_transition_strengths)
from gpaw.test import equal

# ------------------- Inputs ------------------- #

# Part 1: ground state calculation
pw = 200
kpts = 3
nbands = 8

# Part 2: optical transitions calculation
response = 'density'
spins = 'all'
q_c = [0., 0., 0.]
bzk_kc = np.array([[0., 0., 0.]])

# ------------------- Script ------------------- #

# Part 1: ground state calculation
a = 5.431
atoms = bulk('Si', 'diamond', a=a)

calc = GPAW(mode=PW(pw),
            kpts=(kpts, kpts, kpts),
            nbands=nbands,
            convergence={'bands': -1},
            xc='LDA',
            occupations=FermiDirac(0.001))  # use small FD smearing

atoms.set_calculator(calc)
atoms.get_potential_energy()  # get ground state density

calc.write('si.gpw', 'all')  # write wavefunctions

# Part 2: optical transition calculation

pair, pd, domainarg_td = get_bz_transitions('si.gpw', q_c, bzk_kc,
                                            response=response, spins=spins,
                                            ecut=10)

# non-empty bands
n_n = np.arange(0, pair.nocc2)
# not completely filled bands
m_m = np.arange(pair.nocc1, pair.calc.wfs.bd.nbands)

nt = len(domainarg_td)
nn = len(n_n)
nm = len(m_m)
nG = pd.ngmax
optical_limit = np.allclose(q_c, 0.) and response == 'density'

n_tnmG = np.zeros((nt, nn, nm, nG + 2 * optical_limit), dtype=complex)
df_tnm = np.zeros((nt, nn, nm), dtype=float)
eps_tn = np.zeros((nt, nn), dtype=float)
eps_tm = np.zeros((nt, nm), dtype=float)
for t, domainarg_d in enumerate(domainarg_td):
    (n_tnmG[t], df_tnm[t],
     eps_tn[t], eps_tm[t]) = get_chi0_integrand(pair, pd,
                                                n_n, m_m,
                                                *domainarg_d)

testNM_ibN = np.array([[[0], [4, 5, 6]], [[0], [7]],
                       [[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [7]]])
testts_iG = np.array([[0.07, 0.07, 0.07], [0.00, 0.00, 0.00],
                      [51.34, 51.34, 51.34], [22.69, 22.69, 22.69]])

for t, (domainarg_d, n_nmG,
        df_nm, eps_n, eps_m) in enumerate(zip(domainarg_td, n_tnmG, df_tnm,
                                              eps_tn, eps_tm)):

    # Find degeneracies
    degmat_Nn, eps_N = get_degeneracy_matrix(eps_n, tol=1.e-2)
    degmat_Mm, eps_M = get_degeneracy_matrix(eps_m, tol=1.e-2)

    # Find diagonal transition strengths
    its_nmG = np.zeros((nn, nm, 1 + 2 * optical_limit))
    for G in range(1 + 2 * optical_limit):
        its_nmG[:, :, G] = get_individual_transition_strengths(n_nmG, df_nm,
                                                               G, G)

    # Find unique transition strengths
    its_NmG = np.tensordot(degmat_Nn, its_nmG, axes=(1, 0))
    ts_MNG = np.tensordot(degmat_Mm, its_NmG, axes=(1, 1))
    ts_NMG = np.transpose(ts_MNG, (1, 0, 2))

    i = 0
    for N, ts_MG in enumerate(ts_NMG):
        for M, ts_G in enumerate(ts_MG):
            degN_n = n_n[np.where(degmat_Nn[N])]
            degM_m = m_m[np.where(degmat_Mm[M])]

            for testn, n in zip(testNM_ibN[i, 0], degN_n):
                equal(testn, n, 0.5)
            for testm, m in zip(testNM_ibN[i, 1], degM_m):
                equal(testm, m, 0.5)

            for testts, ts in zip(testts_iG[i], ts_G):
                equal(testts, ts, 0.01)

            i += 1
