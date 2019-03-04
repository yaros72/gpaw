from __future__ import print_function
import itertools
import numpy as np
from gpaw.grid_descriptor import GridDescriptor
from gpaw.xc import XC
from gpaw.xc.libvdwxc import vdw_df_cx
from gpaw.mpi import world

def test(xc, tol=5e-10):
    N_c = np.array([10, 6, 4])
    # This test is totally serial.
    gd = GridDescriptor(N_c, N_c * 0.2, pbc_c=(1, 0, 1),
                        comm=world.new_communicator([world.rank]))
    actual_n = N_c - (1 - gd.pbc_c)

    gen = np.random.RandomState(17)
    n_sg = gd.zeros(2)
    n_sg[:] = gen.rand(*n_sg.shape)
    #sigma_xg = gd.zeros(3)
    #sigma_xg[:] = gen.random.rand(*sigma_xg.shape)

    if hasattr(xc, 'libvdwxc'):
        xc._nspins = 2
        xc.initialize_backend(gd)

    v_sg = gd.zeros(2)
    E = xc.calculate(gd, n_sg, v_sg)
    print('E', E)

    dn = 1e-6

    all_indices = itertools.product(range(2),
                                    range(1, actual_n[0], 2),
                                    range(0, actual_n[1], 2),
                                    range(0, actual_n[2], 2))

    for testindex in all_indices:
        n1_sg = n_sg.copy()
        n2_sg = n_sg.copy()
        v = v_sg[testindex]  * gd.dv
        n1_sg[testindex] -= dn
        n2_sg[testindex] += dn

        E1 = xc.calculate(gd, n1_sg, v_sg.copy())
        E2 = xc.calculate(gd, n2_sg, v_sg.copy())

        dedn = 0.5 * (E2 - E1) / dn
        err = abs(dedn - v)
        print('{}{} v={} fd={} err={}'.format(xc.name, list(testindex),
                                              v, dedn, err))
        assert err < tol, err

test(XC('PBE'))
test(vdw_df_cx())
test(vdw_df_cx(vdwcoef=0.0))
test(vdw_df_cx(vdwcoef=1e5), tol=2e-6)
