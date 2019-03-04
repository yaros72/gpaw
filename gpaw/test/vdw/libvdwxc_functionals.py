from __future__ import print_function
import numpy as np
from gpaw.grid_descriptor import GridDescriptor
from gpaw.xc.libvdwxc import vdw_df, vdw_df2, vdw_df_cx, \
    vdw_optPBE, vdw_optB88, vdw_C09, vdw_beef, \
    libvdwxc_has_mpi, libvdwxc_has_pfft

# This test verifies that the results returned by the van der Waals
# functionals implemented in libvdwxc do not change.

N_c = np.array([23, 10, 6])
gd = GridDescriptor(N_c, N_c * 0.2, pbc_c=(1, 0, 1))

n_sg = gd.zeros(1)
nG_sg = gd.collect(n_sg)
if gd.comm.rank == 0:
    gen = np.random.RandomState(0)
    nG_sg[:] = gen.rand(*nG_sg.shape)
gd.distribute(nG_sg, n_sg)

for mode in ['serial', 'mpi', 'pfft']:
    if mode == 'serial' and gd.comm.size > 1:
        continue
    if mode == 'mpi' and not libvdwxc_has_mpi():
        continue
    if mode == 'pfft' and not libvdwxc_has_pfft():
        continue

    errs = []

    def test(vdwxcclass, Eref=np.nan, nvref=np.nan):
        print('')
        xc = vdwxcclass(mode=mode)
        xc.initialize_backend(gd)
        if gd.comm.rank == 0:
            print(xc.libvdwxc.tostring())
        v_sg = gd.zeros(1)
        E = xc.calculate(gd, n_sg, v_sg)
        nv = gd.integrate(n_sg * v_sg, global_integral=True)
        nv = float(nv)  # Comes out as an array due to spin axis

        Eerr = abs(E - Eref)
        nverr = abs(nv - nvref)
        errs.append((vdwxcclass.__name__, Eerr, nverr))

        if gd.comm.rank == 0:
            name = xc.name
            print(name)
            print('=' * len(name))
            print('E  = %19.16f vs ref = %19.16f :: err = %10.6e'
                  % (E, Eref, Eerr))
            print('nv = %19.16f vs ref = %19.16f :: err = %10.6e'
                  % (nv, nvref, nverr))
            print()
        gd.comm.barrier()

        print('Update:')
        print('    test({}, {!r}, {!r})'.format(vdwxcclass.__name__,
                                                E, nv))
    test(vdw_df, -3.8730338590248383, -4.905182929615311)
    test(vdw_df2, -3.9017518972499317, -4.933889152385742)
    test(vdw_df_cx, -3.7262108992876644, -4.760433536500078)
    test(vdw_optPBE, -3.7954301587466506, -4.834460613766266)
    test(vdw_optB88, -3.8486341203104613, -4.879005564922708)
    test(vdw_C09, -3.7071083039260464, -4.746114441237086)
    test(vdw_beef, -3.8926148228224444, -4.961101745896925)

    if any(err[1] > 1e-14 or err[2] > 1e-14 for err in errs):
        # Try old values (compatibility)
        del errs[:]

        test(vdw_df, -3.8730338473027368, -4.905182296422721)
        test(vdw_df2, -3.9017516508476211, -4.933888350723616)
        test(vdw_df_cx, -3.7262108875655624, -4.760432903307487)
        test(vdw_optPBE, -3.7954301470245491, -4.834459980573675)
        test(vdw_optB88, -3.8486341085883597, -4.879004931730118)
        test(vdw_C09, -3.7071082922039449, -4.746113808044496)
        test(vdw_beef, -3.8926145764201334, -4.9611009442348015)

        for name, Eerr, nverr in errs:
            assert Eerr < 1e-14 and nverr < 1e-14, (name, Eerr, nverr)
