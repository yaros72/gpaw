from __future__ import print_function
from math import pi, acos, sqrt

import numpy as np
from ase.atoms import string2vector
from ase.units import Bohr, Hartree

import gpaw.mpi as mpi
from gpaw.spherical_harmonics import Y
from gpaw.utilities.tools import coordinates


class AngularIntegral:

    """Perform an angular integral on the grid.

    center:
      the center (Ang)
    gd:
      grid_descriptor of the grids to expand
    Rmax:
      maximal radius of the expansion (Ang)
    dR:
      grid spacing in the radius (Ang)
    """

    def __init__(self, center, gd, Rmax=None, dR=None):
        assert gd.orthogonal
        center = Vector3d(center) / Bohr

        self.center = center
        self.gd = gd

        # set Rmax to the maximal radius possible
        # i.e. the corner distance
        if not Rmax:
            Rmax = 0
            extreme = gd.h_cv.diagonal() * gd.N_c
            for corner in ([0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                           [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]):
                Rmax = max(Rmax,
                           self.center.distance(np.array(corner) * extreme))
        else:
            Rmax /= Bohr
        self.Rmax = Rmax

        if not dR:
            dR = min(gd.h_cv.diagonal())
        else:
            dR /= Bohr
        self.dR = dR

        self.initialize()

    def initialize(self):
        """Initialize grids."""

        Rmax = self.Rmax
        dR = self.dR
        gd = self.gd

        # initialize the ylm and Radial grids

        # self.V_R will contain the volume of the R shell
        # self.R_g will contain the radial indicees corresponding to
        #     each grid point
        # self.ball_g will contain the mask of the ball of radius Rmax
        # self.y_Lg will contain the YL values corresponding to
        #     each grid point
        V_R = np.empty((int(Rmax / dR + 1),))
        R_R = np.empty((int(Rmax / dR + 1),))

        r_cg, r2_g = coordinates(gd, self.center, tiny=1.e-78)
        r_g = np.sqrt(r2_g)
        rhat_cg = r_cg / r_g

        ball_g = np.where(r_g < Rmax, 1, 0)
        self.R_g = np.where(r_g < Rmax, r_g / dR, -1).astype(int)

        if hasattr(self, 'L_l'):  # ExpandYl
            npY = np.vectorize(Y, (float,), 'spherical harmonic')
            nL = len(self.L_l)
            y_Lg = []
            for L in range(nL):
                y_Lg.append(npY(L, rhat_cg[0], rhat_cg[1], rhat_cg[2]))
            self.y_Lg = y_Lg

        for i, v in enumerate(V_R):
            R_g = np.where(self.R_g == i, 1., 0.)
            V_R[i] = gd.integrate(R_g)
            R_R[i] = gd.integrate(R_g * r_g)

        self.ball_g = ball_g
        self.V_R = V_R
        self.nominalR_R = self.dR * (np.arange(len(self.V_R)) + .5)
        V_R = np.where(V_R > 0, V_R, -1)
        self.R_R = np.where(V_R > 0, R_R / V_R, self.nominalR_R)

    def integrate(self, f_g):
        """Integrate a function on the grid over the angles.

        Contains the weight 4*pi*R^2 with R in Angstrom."""
        int_R = []
        for i, dV in enumerate(self.V_R):
            # get the R shell
            R_g = np.where(self.R_g == i, 1, 0)
            int_R.append(self.gd.integrate(f_g * R_g) / self.dR)
        return np.array(int_R) * Bohr ** 2

    def average(self, f_g):
        """Give the angular average of a function on the grid."""
        V_R = np.where(self.V_R > 0, self.V_R, 1.e32)
        return self.integrate(f_g) * self.dR / V_R / Bohr ** 2

    def radii(self, model='nominal'):
        """Return the radii of the radial shells in Angstrom"""
        if model == 'nominal':
            return self.nominalR_R * Bohr
        elif model == 'mean':
            return self.R_R * Bohr
        else:
            raise NotImplementedError


class ExpandYl(AngularIntegral):

    """Expand the smooth wave functions in spherical harmonics
    relative to a given center.

    center:
      the center for the expansion (Ang)
    gd:
      grid_descriptor of the grids to expand
    lmax:
      maximal angular momentum in the expansion (lmax<7)
    Rmax:
      maximal radius of the expansion (Ang)
    dR:
      grid spacing in the radius (Ang)
    """

    def __init__(self, center, gd, lmax=6, Rmax=None, dR=None):

        self.lmax = lmax
        self.L_l = []
        for l in range(lmax + 1):
            for m in range(2 * l + 1):
                self.L_l.append(l)

        AngularIntegral.__init__(self, center, gd, Rmax, dR)

    def expand(self, psit_g):
        """Expand a wave function"""

        gamma_l = np.zeros((self.lmax + 1))
        nL = len(self.L_l)
        L_l = self.L_l

        def abs2(z):
            return z.real**2 + z.imag**2

        for i, dV in enumerate(self.V_R):
            # get the R shell and it's Volume
            R_g = np.where(self.R_g == i, 1, 0)
            if dV > 0:
                for L in range(nL):
                    psit_LR = self.gd.integrate(psit_g * R_g * self.y_Lg[L])
                    gamma_l[L_l[L]] += 4 * pi / dV * abs2(psit_LR)
        # weight of the wave function inside the ball
        weight = self.gd.integrate((psit_g * psit_g.conj()).real * self.ball_g)

        return gamma_l, weight

    def to_file(self, calculator,
                filename='expandyl.dat',
                spins=None,
                kpoints=None,
                bands=None
                ):
        """Expand a range of wave functions and write the result
        to a file"""
        if mpi.rank == 0:
            f = open(filename, 'w')
        else:
            f = open('/dev/null', 'w')

        if not spins:
            srange = range(calculator.wfs.nspins)
        else:
            srange = spins
        if not kpoints:
            krange = range(len(calculator.wfs.kd.ibzk_kc))
        else:
            krange = kpoints
        if not bands:
            nrange = range(calculator.wfs.bd.nbands)
        else:
            nrange = bands

        print('# Yl expansion', 'of smooth wave functions', file=f)
        lu = 'Angstrom'
        print('# center =', self.center * Bohr, lu, file=f)
        print('# Rmax =', self.Rmax * Bohr, lu, file=f)
        print('# dR =', self.dR * Bohr, lu, file=f)
        print('# lmax =', self.lmax, file=f)
        print('# s    k     n', end=' ', file=f)
        print('kpt-wght    e[eV]      occ', end=' ', file=f)
        print('    norm      sum   weight', end=' ', file=f)
        spdfghi = 's p d f g h i'.split()
        for l in range(self.lmax + 1):
            print('      %' + spdfghi[l], end=' ', file=f)
        print(file=f)

        for s in srange:
            for k in krange:
                u = k * calculator.wfs.nspins + s
                for n in nrange:
                    kpt = calculator.wfs.kpt_u[u]
                    psit_G = kpt.psit_nG[n]
                    norm = self.gd.integrate((psit_G * psit_G.conj()).real)

                    gl, weight = self.expand(psit_G)
                    gsum = np.sum(gl)
                    gl = 100 * gl / gsum

                    print('%2d %5d %5d' % (s, k, n), end=' ', file=f)
                    print('%6.4f %10.4f %8.4f' % (kpt.weight,
                                                  kpt.eps_n[n] * Hartree,
                                                  kpt.f_n[n]),
                          end=' ', file=f)
                    print('%8.4f %8.4f %8.4f' %
                          (norm, gsum, weight), end=' ', file=f)

                    for g in gl:
                        print('%8.2f' % g, end=' ', file=f)
                    print(file=f)
                    f.flush()
        f.close()


class Vector3d(list):
    def __init__(self,vector=None):
        if vector is None:
            vector = [0,0,0]
        vector = string2vector(vector)
        list.__init__(self)
        for c in range(3):
            self.append(float(vector[c]))
        self.l = False

    def __add__(self, other):
        result = self.copy()
        for c in range(3):
            result[c] += other[c]
        return result

    def __truediv__(self,other):
        return Vector3d(np.array(self) / other)

    __div__ = __truediv__
    
    def __mul__(self, x):
        if isinstance(x, type(self)):
            return np.dot( self, x )
        else:
            return Vector3d(x * np.array(self))
        
    def __rmul__(self, x):
        return self.__mul__(x)
        
    def __lmul__(self, x):
        return self.__mul__(x)

    def __neg__(self):
        return -1 * self
        
    def __str__(self):
        return "(%g,%g,%g)" % tuple(self)

    def __sub__(self, other):
        result = self.copy()
        for c in range(3):
            result[c] -= other[c]
        return result

    def angle(self, other):
        """Return the angle between the directions of yourself and the
        other vector in radians."""
        other = Vector3d(other)
        ll = self.length() * other.length()
        if not ll > 0:
            return None
        return acos((self * other) / ll)
        
    def copy(self):
        return Vector3d(self)

    def distance(self,vector):
        if not isinstance(vector, type(self)):
            vector=Vector3d(vector)
        return (self - vector).length()

    def length(self,value=None):
        if value:
            fac = value / self.length()
            for c in range(3):
                self[c] *= fac
            self.l = False
        if not self.l:
            self.l = sqrt(self.norm())
        return self.l

    def norm(self):
        #return np.sum( self*self )
        return self*self  #  XXX drop this class and use numpy arrays ...
                         
    def x(self):
        return self[0]

    def y(self):
        return self[1]

    def z(self):
        return self[2]
