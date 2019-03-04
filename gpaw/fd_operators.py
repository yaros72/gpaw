# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""Finite difference operators.

This file defines a series of finite difference operators used in grid mode.
"""

from __future__ import division
from math import pi, factorial as fact

import numpy as np
from numpy.fft import fftn, ifftn

import _gpaw
from gpaw import debug


# Expansion coefficients for finite difference Laplacian.  The numbers are
# from J. R. Chelikowsky et al., Phys. Rev. B 50, 11355 (1994):
laplace = [[0],
           [-2, 1],
           [-5 / 2, 4 / 3, -1 / 12],
           [-49 / 18, 3 / 2, -3 / 20, 1 / 90],
           [-205 / 72, 8 / 5, -1 / 5, 8 / 315, -1 / 560],
           [-5269 / 1800, 5 / 3, -5 / 21, 5 / 126, -5 / 1008, 1 / 3150],
           [-5369 / 1800, 12 / 7, -15 / 56, 10 / 189, -1 / 112, 2 / 1925,
            -1 / 16632]]

derivatives = [[1 / 2],
               [2 / 3, -1 / 12],
               [3 / 4, -3 / 20, 1 / 60],
               [4 / 5, -1 / 5, 4 / 105, -1 / 280]]


class FDOperator:
    def __init__(self, coef_p, offset_pc, gd, dtype=float,
                 description=None):
        """FDOperator(coefs, offsets, gd, dtype) -> FDOperator object.
        """

        # Is this a central finite-difference type of stencil?
        cfd = True
        for offset_c in offset_pc:
            if sum([offset != 0 for offset in offset_c]) >= 2:
                cfd = False

        maxoffset_c = [max([offset_c[c] for offset_c in offset_pc])
                       for c in range(3)]

        mp = maxoffset_c[0]
        if maxoffset_c[1] != mp or maxoffset_c[2] != mp:
            mp = max(maxoffset_c)
        n_c = gd.n_c
        M_c = n_c + 2 * mp
        stride_c = np.array([M_c[1] * M_c[2], M_c[2], 1])
        offset_p = np.dot(offset_pc, stride_c)
        coef_p = np.ascontiguousarray(coef_p, float)
        neighbor_cd = gd.neighbor_cd
        assert coef_p.ndim == 1
        assert coef_p.shape == offset_p.shape
        assert dtype in [float, complex]
        self.dtype = dtype
        self.shape = tuple(n_c)

        if gd.comm.size > 1:
            comm = gd.comm.get_c_object()
        else:
            comm = None

        assert neighbor_cd.flags.c_contiguous and offset_p.flags.c_contiguous
        self.mp = mp  # padding
        self.gd = gd
        self.npoints = len(coef_p)
        self.coef_p = coef_p
        self.offset_p = offset_p
        self.offset_pc = offset_pc
        self.comm = comm
        self.cfd = cfd

        self.operator = _gpaw.Operator(coef_p, offset_p, n_c, mp,
                                       neighbor_cd, dtype == float,
                                       comm, cfd)

        if description is None:
            description = '%d point finite-difference stencil' % self.npoints
        self.description = description

    def __str__(self):
        return '<' + self.description + '>'

    def apply(self, in_xg, out_xg, phase_cd=None):
        self.operator.apply(in_xg, out_xg, phase_cd)

    def relax(self, relax_method, f_g, s_g, n, w=None):
        self.operator.relax(relax_method, f_g, s_g, n, w)

    def get_diagonal_element(self):
        return self.operator.get_diagonal_element()

    def get_async_sizes(self):
        return self.operator.get_async_sizes()


if debug:
    _FDOperator = FDOperator

    class FDOperator(_FDOperator):
        def apply(self, in_xg, out_xg, phase_cd=None):
            assert in_xg.shape == out_xg.shape
            assert in_xg.shape[-3:] == self.shape
            assert in_xg.flags.contiguous
            assert in_xg.dtype == self.dtype
            assert out_xg.flags.contiguous
            assert out_xg.dtype == self.dtype
            assert (self.dtype == float or
                    (phase_cd.dtype == complex and
                     phase_cd.shape == (3, 2)))
            _FDOperator.apply(self, in_xg, out_xg, phase_cd)

        def relax(self, relax_method, f_g, s_g, n, w=None):
            assert f_g.shape == self.shape
            assert s_g.shape == self.shape
            assert f_g.flags.contiguous
            assert f_g.dtype == float
            assert s_g.flags.contiguous
            assert s_g.dtype == float
            assert self.dtype == float
            _FDOperator.relax(self, relax_method, f_g, s_g, n, w)


def Laplace(gd, scale=1.0, n=1, dtype=float):
    if n == 9:
        return FTLaplace(gd, scale, dtype)
    else:
        return GUCLaplace(gd, scale, n, dtype)


class GUCLaplace(FDOperator):
    def __init__(self, gd, scale=1.0, n=1, dtype=float):
        """Laplacian for general non orthorhombic grid.

        gd: GridDescriptor
            Descriptor for grid.
        scale: float
            Scaling factor.  Use scale=-0.5 for a kinetic energy operator.
        n: int
            Range of stencil.  Stencil has O(h^(2n)) error.
        dtype: float or complex
            Datatype to work on.
        """

        # Order the 26 neighbor grid points after length
        # (reduced to 13 inequivalent):
        M_ic = np.indices((3, 3, 3)).reshape((3, -3)).T[-13:] - 1
        u_cv = gd.h_cv / (gd.h_cv**2).sum(1)[:, np.newaxis]**0.5
        u2_i = (np.dot(M_ic, u_cv)**2).sum(1)
        i_d = u2_i.argsort()

        # x^2, y^2, z^2, yz, xz, xy:
        m_mv = np.array([(2, 0, 0), (0, 2, 0), (0, 0, 2),
                         (0, 1, 1), (1, 0, 1), (1, 1, 0)])
        # Try 3, 4, 5 and 6 of the shortest directions:
        for D in range(3, 7):
            h_dv = np.dot(M_ic[i_d[:D]], gd.h_cv)
            A_md = (h_dv**m_mv[:, np.newaxis, :]).prod(2)
            # Find best stencil coefficients:
            a_d, residual, rank, s = np.linalg.lstsq(A_md, [1, 1, 1, 0, 0, 0],
                                                     rcond=-1)
            if residual.sum() < 1e-14:
                if rank != D:
                    raise ValueError(
                        'You have a weird unit cell!  '
                        'Try to use the maximally reduced Niggli cell.  '
                        'See the ase.build.niggli_reduce() function.')
                # D directions was OK
                break

        a_d *= scale
        offsets = [(0, 0, 0)]
        coefs = [laplace[n][0] * a_d.sum()]
        for d in range(D):
            M_c = M_ic[i_d[d]]
            offsets.extend(np.arange(1, n + 1)[:, np.newaxis] * M_c)
            coefs.extend(a_d[d] * np.array(laplace[n][1:]))
            offsets.extend(np.arange(-1, -n - 1, -1)[:, np.newaxis] * M_c)
            coefs.extend(a_d[d] * np.array(laplace[n][1:]))

        FDOperator.__init__(self, coefs, offsets, gd, dtype)

        self.description = (
            '%d*%d+1=%d point O(h^%d) finite-difference Laplacian' %
            ((self.npoints - 1) // n, n, self.npoints, 2 * n))


class Gradient(FDOperator):
    def __init__(self, gd, v, scale=1.0, n=1, dtype=float):
        """Symmetric gradient for general non orthorhombic grid.

        gd: GridDescriptor
            Descriptor for grid.
        v: int
            Direction of gradient: 0, 1, or 2 for x, y and z.
        scale: float
            Scaling factor.
        n: int
            Range of stencil.  Stencil has O(h^(2n)) error.
        dtype: float or complex
            Datatype to work on.
        """

        from scipy.spatial import Voronoi

        # Find nearest neighbors.  If h is a vector pointing at a
        # neighbor grid-points then we don't also include -h in the list:
        M_ic = np.indices((3, 3, 3)).reshape((3, -3)).T - 1
        h_iv = M_ic.dot(gd.h_cv)
        voro = Voronoi(h_iv)
        i_d = []  # List[int]
        for i1, i2 in voro.ridge_points:
            if i1 == 13 and i2 > 13:
                i_d.append(i2)
            elif i2 == 13 and i1 > 13:
                i_d.append(i1)

        D = len(i_d)  # number of neighbors (3, 4, 5, 6 or 7)

        h_dv = h_iv[i_d]  # vectors pointing at neighbor grid-points

        # Find gradient along 3 directions (n_cv):
        invh_vc = np.linalg.inv(gd.h_cv)
        n_cv = (invh_vc / (invh_vc**2).sum(axis=0)**0.5).T

        coef_cd = np.zeros((3, D))  # unknown weights
        for c, n_v in enumerate(n_cv):
            # Point on the plane perpendicular to the directions get a
            # weight of zero
            ok_d = abs(h_dv.dot(n_v)) > 1e-10
            h_mv = h_dv[ok_d]

            # The theee equations: A_jm.dot(coef_m) = [1, 0, 0]
            A_jm = np.array([h_mv.dot(n_v),
                             h_mv.dot(gd.h_cv[c - 1]),
                             h_mv.dot(gd.h_cv[c - 2])])

            U_jm, S_m, V_mm = np.linalg.svd(A_jm, full_matrices=False)
            coef_m = V_mm.T.dot(np.diag(S_m**-1).dot(U_jm.T.dot([1, 0, 0])))
            coef_cd[c, ok_d] = coef_m

        # Now get the v-gradient:
        coef_d = np.linalg.inv(n_cv)[v].dot(coef_cd) * scale

        # Create stencil:
        offsets = []
        coefs = []
        stencil = np.array(derivatives[n - 1])
        for d, c in enumerate(coef_d):
            if abs(c) < 1e-10:
                continue
            M_c = M_ic[i_d[d]]
            offsets.extend(np.arange(1, n + 1)[:, np.newaxis] * M_c)
            coefs.extend(c * stencil)
            offsets.extend(np.arange(-1, -n - 1, -1)[:, np.newaxis] * M_c)
            coefs.extend(-c * stencil)

        FDOperator.__init__(self, coefs, offsets, gd, dtype)

        self.description = (
            'Finite-difference {}-derivative with O(h^{}) error ({} points)'
            .format('xyz'[v], 2 * n, self.npoints))


class LaplaceA(FDOperator):
    def __init__(self, gd, scale, dtype=float):
        assert gd.orthogonal
        c = np.divide(-1 / 12, gd.h_cv.diagonal()**2) * scale  # Why divide?
        c0 = c[1] + c[2]
        c1 = c[0] + c[2]
        c2 = c[1] + c[0]
        a = -16.0 * np.sum(c)
        b = 10.0 * c + 0.125 * a
        FDOperator.__init__(self,
                            [a,
                             b[0], b[0],
                             b[1], b[1],
                             b[2], b[2],
                             c0, c0, c0, c0,
                             c1, c1, c1, c1,
                             c2, c2, c2, c2],
                            [(0, 0, 0),
                             (-1, 0, 0), (1, 0, 0),
                             (0, -1, 0), (0, 1, 0),
                             (0, 0, -1), (0, 0, 1),
                             (0, -1, -1), (0, -1, 1), (0, 1, -1), (0, 1, 1),
                             (-1, 0, -1), (-1, 0, 1), (1, 0, -1), (1, 0, 1),
                             (-1, -1, 0), (-1, 1, 0), (1, -1, 0), (1, 1, 0)],
                            gd, dtype,
                            'O(h^4) Mehrstellen Laplacian (A)')


class LaplaceB(FDOperator):
    def __init__(self, gd, dtype=float):
        a = 0.5
        b = 1.0 / 12.0
        FDOperator.__init__(self,
                            [a,
                             b, b, b, b, b, b],
                            [(0, 0, 0),
                             (-1, 0, 0), (1, 0, 0),
                             (0, -1, 0), (0, 1, 0),
                             (0, 0, -1), (0, 0, 1)],
                            gd, dtype,
                            'O(h^4) Mehrstellen Laplacian (B)')


class FTLaplace:
    def __init__(self, gd, scale, dtype):
        assert gd.comm.size == 1 and gd.pbc_c.all()

        N_c1 = gd.N_c[:, np.newaxis]
        i_cq = np.indices(gd.N_c).reshape((3, -1))
        i_cq += N_c1 // 2
        i_cq %= N_c1
        i_cq -= N_c1 // 2
        B_vc = 2.0 * pi * gd.icell_cv.T
        k_vq = np.dot(B_vc, i_cq)
        k_vq *= k_vq
        self.k2_Q = k_vq.sum(axis=0).reshape(gd.N_c)
        self.k2_Q *= -scale
        self.d = 6.0 / gd.h_cv[0, 0]**2
        self.npoints = 1000

    def apply(self, in_xg, out_xg, phase_cd=None):
        if in_xg.ndim > 3:
            for in_g, out_g in zip(in_xg, out_xg):
                out_g[:] = ifftn(fftn(in_g) * self.k2_Q).real
        else:
            out_xg[:] = ifftn(fftn(in_xg) * self.k2_Q).real

    def get_diagonal_element(self):
        return self.d


class OldGradient(FDOperator):
    def __init__(self, gd, v, scale=1.0, n=1, dtype=float):
        h = (gd.h_cv**2).sum(1)**0.5
        d = gd.xxxiucell_cv[:, v]
        A = np.zeros((2 * n + 1, 2 * n + 1))
        for i, io in enumerate(range(-n, n + 1)):
            for j in range(2 * n + 1):
                A[i, j] = io**j / float(fact(j))
        A[n, 0] = 1.0
        coefs = np.linalg.inv(A)[1]
        coefs = np.delete(coefs, len(coefs) // 2)
        offs = np.delete(np.arange(-n, n + 1), n)
        coef_p = []
        offset_pc = []
        for i in range(3):
            if abs(d[i]) > 1e-11:
                coef_p.extend(list(coefs * d[i] / h[i] * scale))
                offset = np.zeros((2 * n, 3), int)
                offset[:, i] = offs
                offset_pc.extend(offset)

        FDOperator.__init__(self, coef_p, offset_pc, gd, dtype,
                            'O(h^%d) %s-gradient stencil' % (2 * n, 'xyz'[v]))
