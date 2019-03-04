import numpy as np
import ase.io.ulm as ulm

from gpaw import GPAW
from gpaw.mpi import world


def ibz2bz(input_gpw, output_gpw=None):
    """Unfold IBZ to full BZ and write new gpw-file.

    Example::

        ibz2bz('abc.gpw')

    This will create a 'abc-bz.gpw' file.  Works also from the command line::

        $ python3 -m gpaw.utilities.ibz2bz abc.gpw

    """

    if world.rank > 0:
        return

    calc = GPAW(input_gpw, txt=None)
    spos_ac = calc.atoms.get_scaled_positions()
    kd = calc.wfs.kd

    nK = kd.N_c.prod()
    nbands = calc.wfs.bd.nbands
    ns = calc.wfs.nspins

    I_a = None

    for K, k in enumerate(kd.bz2ibz_k):
        a_a, U_aii, time_rev = construct_symmetry_operators(calc.wfs,
                                                            spos_ac, K)

        if I_a is None:
            # First time:
            I1 = 0
            I_a = [0]
            for U_ii in U_aii:
                I2 = I1 + len(U_ii)
                I_a.append(I2)
                I1 = I2

            # Allocate arrays:
            P_sKnI = np.empty((ns, nK, nbands, I2), calc.wfs.dtype)
            e_sKn = np.empty((ns, nK, nbands))
            f_sKn = np.empty((ns, nK, nbands))

        for s in range(ns):
            e_sKn[s, K] = calc.get_eigenvalues(k, s)
            f_sKn[s, K] = calc.get_occupation_numbers(k, s)
            P_nI = calc.wfs.collect_projections(k, s)
            P2_nI = P_sKnI[s, K]
            a = 0
            for b, U_ii in zip(a_a, U_aii):
                P_ni = np.dot(P_nI[:, I_a[b]:I_a[b + 1]], U_ii)
                if time_rev:
                    P_ni = P_ni.conj()
                P2_nI[:, I_a[a]:I_a[a + 1]] = P_ni
                a += 1

    # Write new gpw-file:
    parameters = calc.todict()
    parameters['symmetry'] = 'off'

    if output_gpw is None:
        assert input_gpw.endswith('.gpw')
        output_gpw = input_gpw[:-4] + '-bz.gpw'
    out = ulm.open(output_gpw, 'w')

    ulm.copy(input_gpw, out, exclude={'.wave_functions', '.parameters'})

    out.child('parameters').write(**parameters)

    wfs = out.child('wave_functions')
    wfs.write(eigenvalues=e_sKn,
              occupations=f_sKn,
              projections=P_sKnI)

    wfs.child('kpts').write(bz2ibz=np.arange(nK),
                            bzkpts=kd.bzk_kc,
                            ibzkpts=kd.bzk_kc,
                            weights=np.ones(nK) / nK)
    out.close()


def construct_symmetry_operators(wfs, spos_ac, K):
    """Construct symmetry operators for PAW projections.

    We want to transform a k-point in the irreducible part of the BZ to
    the corresponding k-point with index K.

    Returns a_a, U_aii, and time_reversal, where:

    * a_a is a list of symmetry related atom indices
    * U_aii is a list of rotation matrices for the PAW projections
    * time_reversal is a flag - if True, projections should be complex
      conjugated.
    """

    kd = wfs.kd

    s = kd.sym_k[K]
    U_cc = kd.symmetry.op_scc[s]
    time_reversal = kd.time_reversal_k[K]
    ik = kd.bz2ibz_k[K]
    ik_c = kd.ibzk_kc[ik]

    a_a = []
    U_aii = []
    for a, id in enumerate(wfs.setups.id_a):
        b = kd.symmetry.a_sa[s, a]
        S_c = np.dot(spos_ac[a], U_cc) - spos_ac[b]
        x = np.exp(2j * np.pi * np.dot(ik_c, S_c))
        U_ii = wfs.setups[a].R_sii[s].T * x
        a_a.append(b)
        U_aii.append(U_ii)

    return a_a, U_aii, time_reversal


if __name__ == '__main__':
    import sys
    ibz2bz(sys.argv[1])


