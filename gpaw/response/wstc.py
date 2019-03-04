"""Wigner-Seitz truncated coulomb interaction.

See:

    Ravishankar Sundararaman and T. A. Arias:
    Phys. Rev. B 87, 165122 (2013)

    Regularization of the Coulomb singularity in exact exchange by
    Wigner-Seitz truncated interactions: Towards chemical accuracy
    in nontrivial systems
"""
from __future__ import print_function
import sys
from math import pi

import numpy as np
from ase.units import Bohr
from ase.utils import seterr

import gpaw.mpi as mpi
from gpaw.utilities import erf
from gpaw.fftw import get_efficient_fft_size
from gpaw.grid_descriptor import GridDescriptor


class WignerSeitzTruncatedCoulomb:
    def __init__(self, cell_cv, nk_c, txt=None):
        txt = txt or sys.stdout
        self.nk_c = nk_c
        bigcell_cv = cell_cv * nk_c[:, np.newaxis]
        L_c = (np.linalg.inv(bigcell_cv)**2).sum(0)**-0.5

        rc = 0.5 * L_c.min()
        print('Inner radius for %dx%dx%d Wigner-Seitz cell: %.3f Ang' %
              (tuple(nk_c) + (rc * Bohr,)), file=txt)

        self.a = 5 / rc
        print('Range-separation parameter: %.3f Ang^-1' % (self.a / Bohr),
              file=txt)

        nr_c = [get_efficient_fft_size(2 * int(L * self.a * 3.0))
                for L in L_c]
        print('FFT size for calculating truncated Coulomb: %dx%dx%d' %
              tuple(nr_c), file=txt)

        self.gd = GridDescriptor(nr_c, bigcell_cv, comm=mpi.serial_comm)
        v_ijk = self.gd.empty()

        pos_ijkv = self.gd.get_grid_point_coordinates().transpose((1, 2, 3, 0))
        corner_xv = np.dot(np.indices((2, 2, 2)).reshape((3, 8)).T, bigcell_cv)

        # Ignore division by zero (in 0,0,0 corner):
        with seterr(invalid='ignore'):
            # Loop over first dimension to avoid too large ndarrays.
            for pos_jkv, v_jk in zip(pos_ijkv, v_ijk):
                # Distances to the 8 corners:
                d_jkxv = pos_jkv[:, :, np.newaxis] - corner_xv
                r_jk = (d_jkxv**2).sum(axis=3).min(2)**0.5
                v_jk[:] = erf(self.a * r_jk) / r_jk

        # Fix 0/0 corner value:
        v_ijk[0, 0, 0] = 2 * self.a / pi**0.5

        self.K_Q = np.fft.fftn(v_ijk) * self.gd.dv

    def get_potential(self, pd, q_v=None):
        q_c = pd.kd.bzk_kc[0]
        shift_c = (q_c * self.nk_c).round().astype(int)
        max_c = self.gd.N_c // 2
        K_G = pd.zeros()
        N_c = pd.gd.N_c
        for G, Q in enumerate(pd.Q_qG[0]):
            Q_c = (np.unravel_index(Q, N_c) + N_c // 2) % N_c - N_c // 2
            Q_c = Q_c * self.nk_c + shift_c
            if (abs(Q_c) < max_c).all():
                K_G[G] = self.K_Q[tuple(Q_c)]

        qG_Gv = pd.get_reciprocal_vectors(add_q=True)
        if q_v is not None:
            qG_Gv += q_v
        G2_G = np.sum(qG_Gv**2, axis=1)
        # G2_G = pd.G2_qG[0]
        a = self.a
        if pd.kd.gamma:
            K_G[0] += pi / a**2
        else:
            K_G[0] += 4 * pi * (1 - np.exp(-G2_G[0] / (4 * a**2))) / G2_G[0]
        K_G[1:] += 4 * pi * (1 - np.exp(-G2_G[1:] / (4 * a**2))) / G2_G[1:]
        assert pd.dtype == complex
        return K_G
