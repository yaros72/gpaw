# import functools
import numpy as np
# from gpaw.mpi import world
# from gpaw.fd_operators import Laplace
from gpaw.grid_descriptor import GridDescriptor
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.matrix import Matrix
from gpaw.atom_centered_functions import AtomCenteredFunctions
from gpaw.wavefunctions.arrays import UniformGridWaveFunctions
from gpaw.spline import Spline


def test(desc, kd, spositions, proj, basis, dS_aii,
         bcomm=None,
         spinpolarized=False, collinear=True, kpt=-1, dtype=None):
    phi_M = AtomCenteredFunctions(desc, basis, kd)  # , [kpt])
    phi_M.set_positions(spositions)
    nbands = len(phi_M)
    if desc.mode == 'fd':
        create = UniformGridWaveFunctions
    else:
        pass  # create = PlaneWaveExpansionWaveFunctions
    psi_n = create(nbands, desc, dtype=dtype, kpt=kpt,
                   collinear=collinear, dist=bcomm)
    phi_M.eval(psi_n)
    print(psi_n.array.real[:, 10, 10])
    print(psi_n.array.max())
    # psi_n.plot()
    S_nn = psi_n.matrix_elements(psi_n, hermitian=True)
    print(S_nn.array)
    pt_I = AtomCenteredFunctions(desc, proj, kd)
    pt_I.set_positions(spositions)
    P_In = Matrix(pt_I.mynfuncs, len(psi_n),
                  dtype=dtype, dist=(bcomm, 1, -1))
    pt_I.matrix_elements(psi_n, out=P_In)
    dSP_In = P_In.new()
    dS_II = None  # AtomBlockMatrix(dS_aii)
    dSP_In[:] = dS_II * P_In
    S_nn += P_In.H * dSP_In
    print(S_nn.array)
    S_nn.cholesky()
    S_nn.inv()
    psi2_n = psi_n.new()
    psi2_n[:] = S_nn.T * psi_n

    psi2_n.matrix_elements(psi2_n, out=S_nn)
    pt_I.matrix_elements(psi2_n, out=P_In)
    dSP_In[:] = dS_II * P_In
    norm = S_nn.array.trace()
    S_nn += P_In.H * dSP_In
    print(S_nn.array, norm)

    """
    nt = UniformGridDensity(desc, spinpolarized, collinear)
    f_n = np.ones(len(psi_n))  # ???
    nt.from_wave_functions(psi2_n, f_n)
    nt.integrate()
    """

    # kin(psit2_n, psit_n)
    # H_nn = Matrix((nbands, nbands), dtype, dist=?????)
    # H_nn = psit2_n.C *


spos = [(0.5, 0.5, 0.5), (0.5, 0.6, 0.75)]
size = np.array([20, 18, 22])
cell = 0.4 * size
gd = GridDescriptor(size, cell)
gd.mode = 'fd'

p = Spline(0, 1.2, [1, 0.6, 0.1, 0.0])
b = Spline(0, 1.7, [1, 0.6, 0.1, 0.0])
b2 = Spline(0, 1.7, [-1, -0.5, -0.2, 0.1, 0.0])

proj = [[p], [p]]
basis = [[b, b2], [b, b2]]
dS_aii = {0: np.array([[0.3]]), 1: np.array([[0.3]])}

test(gd, None, spos, proj, basis, dS_aii)

kd = KPointDescriptor([[0, 0, 0.25]])
# kd.gamma = False
test(gd, kd, spos, proj, basis, dS_aii, kpt=0, dtype=complex)
