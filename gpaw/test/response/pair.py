import numpy as np

from ase import Atoms

from gpaw import GPAW, PW
from gpaw.test import equal
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.wavefunctions.pw import PWDescriptor
from gpaw.response.pair import PairDensity
from gpaw.response.math_func import two_phi_nabla_planewave_integrals

np.set_printoptions(precision=1)

nb = 6

a = Atoms('H', cell=(3 * np.eye(3)), pbc=True)

calc = GPAW(mode=PW(600), kpts=[[0, 0, 0], [0.25, 0, 0]])
a.calc = calc
a.get_potential_energy()
calc.diagonalize_full_hamiltonian(nbands=nb, expert=True)
calc.write('a.gpw', 'all')

pair = PairDensity('a.gpw', ecut=100)

# Check continuity eq.
for q_c in [[0, 0, 0], [1. / 4, 0, 0]]:
    ol = np.allclose(q_c, 0.0)
    qd = KPointDescriptor([q_c])
    pd = PWDescriptor(pair.ecut, calc.wfs.gd, complex, qd)
    kptpair = pair.get_kpoint_pair(pd, 0, [0, 0, 0],
                                   0, nb, 0, nb)
    deps_nm = kptpair.get_transition_energies(np.arange(0, nb),
                                              np.arange(0, nb))

    n_nmG = pair.get_pair_density(pd, kptpair, np.arange(0, nb),
                                  np.arange(0, nb), optical_limit=ol)

    n_nmvG = pair.get_pair_momentum(pd, kptpair, np.arange(0, nb),
                                    np.arange(0, nb))

    if ol:
        n2_nmv = np.zeros((nb, nb, 3), complex)
        for n in range(0, nb):
            n2_nmv[n] = pair.optical_pair_velocity(n, np.arange(0, nb),
                                                   kptpair.kpt1,
                                                   kptpair.kpt2)

    # Check for nan's
    assert not np.isnan(n_nmG).any()
    assert not np.isnan(n_nmvG).any()
    if ol:
        assert not np.isnan(n2_nmv).any()

    # PAW correction test
    if ol:
        print('Checking PAW corrections')
        # Check that PAW-corrections are
        # are equal to nabla-PAW corrections
        G_Gv = pd.get_reciprocal_vectors()
    
        for id, atomdata in pair.calc.wfs.setups.setups.items():
            nabla_vii = atomdata.nabla_iiv.transpose((2, 0, 1))
            Q_vGii = two_phi_nabla_planewave_integrals(G_Gv, atomdata)
            ni = atomdata.ni
            Q_vGii.shape = (3, -1, ni, ni)
            equal(nabla_vii.astype(complex), Q_vGii[:, 0], tolerance=1e-10,
                  msg='Planewave-nabla PAW corrections not equal ' +
                  'to nabla PAW corrections when q + G = 0!')

        # Check optical limit nabla matrix elements
        err = np.abs(n_nmvG[..., 0] - n2_nmv)
        maxerr = np.max(err)
        arg = np.unravel_index(np.argmax(err), err.shape)
        equal(maxerr, 0.0, tolerance=1e-10,
              msg='G=0 pair densities wrong! ' + str(arg) + ' ')

    # Check longitudinal part of matrix elements
    print('Checking continuity eq.')
    G_Gv = pd.get_reciprocal_vectors()
    G2_G = np.sum(G_Gv**2.0, axis=1)

    n2_nmG = np.diagonal(np.dot(G_Gv, n_nmvG), axis1=0, axis2=3).copy()
    if ol:
        n2_nmG[..., 0] = n_nmvG[..., 2, 0]
    
    # Left hand side and right hand side of
    # continuity eq.: d/dr x J + d/dt n = 0
    lhs_nmG = n2_nmG - G2_G[np.newaxis, np.newaxis] / 2. * n_nmG[..., 2 * ol:]
    rhs_nmG = - deps_nm[..., np.newaxis] * n_nmG[..., 2 * ol:]

    err = np.abs(lhs_nmG - rhs_nmG)
    maxerr = np.max(err)
    arg = np.unravel_index(np.argmax(err), rhs_nmG.shape)
    equal(maxerr, 0.0, tolerance=1e-3, msg='Calculated current does ' +
          'not fulfill the continuity equation! ' + str(arg) + ' ')
