from gpaw.symmetry import map_k_points_fast, map_k_points
from ase.dft.kpoints import monkhorst_pack
from gpaw import GPAW
from ase.build import bulk
import numpy as np
from itertools import product


def test_mapping(bz2bz_ks, U_scc, bzk_kc, time_reversal=False):
    eps = 1e-7

    if time_reversal:
        U_scc = np.concatenate([U_scc, -U_scc])

    for k_c, bz2bz_s in zip(bzk_kc, bz2bz_ks):
        delta_sc = (np.dot(U_scc, k_c) -
                    bzk_kc[bz2bz_s, :])[bz2bz_s >= 0]
        delta_sc = np.abs(delta_sc - delta_sc.round())
        assert delta_sc.max() < eps


# Test kpoint mapping functionality of the gpaw.symmetry module
atoms = bulk('C')
calc = GPAW(mode='pw',
            kpts={'size': (5, 5, 5)},
            txt=None)
atoms.calc = calc
atoms.get_potential_energy()

U_scc = calc.wfs.kd.symmetry.op_scc
time_reversal = False

for gamma in [True, False]:
    for time_reversal in [True, False]:
        for i, j, k in product(*([range(1, 7)] * 3)):

            bzk_kc = monkhorst_pack((i, j, k))

            if gamma:
                offset = (((i + 1) % 2) / (2 * i),
                          ((j + 1) % 2) / (2 * j),
                          ((k + 1) % 2) / (2 * k))
                bzk_kc += offset

            bz2bz_ks = map_k_points(bzk_kc, U_scc,
                                    time_reversal,
                                    None, calc.wfs.kd.symmetry.tol)
            bz2bzfast_ks = map_k_points_fast(bzk_kc, U_scc,
                                             time_reversal,
                                             None,
                                             calc.wfs.kd.symmetry.tol)

            assert ((bz2bz_ks - bz2bzfast_ks)**2 < 1e-9).all()

            test_mapping(bz2bz_ks, U_scc, bzk_kc, time_reversal)
            test_mapping(bz2bzfast_ks, U_scc, bzk_kc, time_reversal)
