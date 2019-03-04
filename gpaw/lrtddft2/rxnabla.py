from math import sqrt, pi

import numpy as np


def get_magnetic_integrals_new(self, rgd, phi_jg, phit_jg):
    """Calculate PAW-correction matrix elements of r x nabla.

    ::

      /  _       _          _     ~   _      ~   _
      | dr [phi (r) O  phi (r) - phi (r) O  phi (r)]
      /        1     x    2         1     x    2

                   d      d
      where O  = y -- - z --
             x     dz     dy

    and similar for y and z."""

    # utility functions

    # from Y_L to Y_lm where Y_lm is a spherical harmonic and m= -l, ..., +l
    def YL_to_Ylm(L):
        # (c,l,m)
        if L == 0:
            return [(1.0, 0, 0)]
        if L == 1: # y
            return [ ( 1j/sqrt(2.), 1, -1),
                     ( 1j/sqrt(2.), 1,  1) ]
        if L == 2: # z
            return [(1.0, 1, 0)]
        if L == 3: # x
            return [ (  1/np.sqrt(2.), 1, -1),
                     ( -1/np.sqrt(2.), 1,  1) ]
        if L == 4: # xy
            return [ ( 1j/np.sqrt(2.), 2, -2),
                     (-1j/np.sqrt(2.), 2,  2) ]
        if L == 5: # yz
            return [ ( 1j/np.sqrt(2.), 2, -1),
                     ( 1j/np.sqrt(2.), 2,  1) ]
        if L == 6: # 3z2-r2
            return [(1.0, 2, 0)]
        if L == 7: # zx
            return [ ( 1/np.sqrt(2.), 2, -1),
                     (-1/np.sqrt(2.), 2,  1) ]
        if L == 8: # x2-y2
            return [ ( 1/np.sqrt(2.), 2, -2),
                     ( 1/np.sqrt(2.), 2,  2) ]

        raise NotImplementedError('Error in get_magnetic_integrals_new: '
                                  'YL_to_Ylm not implemented for l>2 yet.')

    # <YL1| Lz |YL2>
    # with help of YL_to_Ylm
    # Lz |lm> = hbar m |lm>
    def YL1_Lz_YL2(L1,L2):
        Yl1m1 = YL_to_Ylm(L1)
        Yl2m2 = YL_to_Ylm(L2)

        sum = 0.j
        for (c1,l1,m1) in Yl1m1:
            for (c2,l2,m2) in Yl2m2:
                # print '--------', c1, l1, m1, c2, l2, m2
                lz = m2
                if l1 == l2 and m1 == m2:
                    sum += lz * np.conjugate(c1) * c2

        return sum

    # <YL1| L+ |YL2>
    # with help of YL_to_Ylm
    # and using L+ |lm> = hbar sqrt( l(l+1) - m(m+1) ) |lm+1>
    def YL1_Lp_YL2(L1,L2):
        Yl1m1 = YL_to_Ylm(L1)
        Yl2m2 = YL_to_Ylm(L2)

        sum = 0.j
        for (c1,l1,m1) in Yl1m1:
            for (c2,l2,m2) in Yl2m2:
                # print '--------', c1, l1, m1, c2, l2, m2
                lp = sqrt(l2*(l2+1) - m2*(m2+1))
                if abs(lp) < 1e-5: continue
                if l1 == l2 and m1 == m2+1:
                    sum += lp * np.conjugate(c1) * c2

        return sum

    # <YL1| L- |YL2>
    # with help of YL_to_Ylm
    # and using L- |lm> = hbar sqrt( l(l+1) - m(m-1) ) |lm-1>
    def YL1_Lm_YL2(L1,L2):
        Yl1m1 = YL_to_Ylm(L1)
        Yl2m2 = YL_to_Ylm(L2)

        sum = 0.j
        for (c1,l1,m1) in Yl1m1:
            for (c2,l2,m2) in Yl2m2:
                # print '--------', c1, l1, m1, c2, l2, m2
                lp = sqrt(l2*(l2+1) - m2*(m2-1))
                if abs(lp) < 1e-5: continue
                if l1 == l2 and m1 == m2-1:
                    sum += lp * np.conjugate(c1) * c2

        return sum

    # <YL1| Lx |YL2>
    # using Lx = (L+ + L-)/2
    def YL1_Lx_YL2(L1,L2):
        return .5 * ( YL1_Lp_YL2(L1,L2) + YL1_Lm_YL2(L1,L2) )

    # <YL1| Lx |YL2>
    # using Ly = -i(L+ - L-)/2
    def YL1_Ly_YL2(L1,L2):
        return -.5j * ( YL1_Lp_YL2(L1,L2) - YL1_Lm_YL2(L1,L2) )


    # r x nabla for [i-index 1, i-index 2, (x,y,z)]
    rxnabla_iiv = np.zeros((self.ni, self.ni, 3))

    # loops over all j1=(l1,m1) values
    i1 = 0
    for j1, l1 in enumerate(self.l_j):
        for m1 in range(2 * l1 + 1):
            L1 = l1**2 + m1
            # loops over all j2=(l2,m2) values
            i2 = 0
            for j2, l2 in enumerate(self.l_j):
                # radial part, which is common for same j values
                # int_0^infty phi_l1,m1,g(r) phi_l2,m2,g(r) * 4*pi*r**2 dr
                # 4 pi here?????
                radial_part = rgd.integrate(phi_jg[j1] * phi_jg[j2] -
                                            phit_jg[j1] * phit_jg[j2]) / (4*pi)

                # <l1m1|r x nabla|l2m2> = i/hbar <l1m1|rxp|l2m2>
                for m2 in range(2 * l2 + 1):
                    L2 = l2**2 + m2
                    # Lx
                    Lx = (1j * YL1_Lx_YL2(L1,L2))
                    #print '%8.3lf %8.3lf | ' % (Lx.real, Lx.imag),
                    rxnabla_iiv[i1,i2,0] = Lx.real * radial_part

                    # Ly
                    Ly = (1j * YL1_Ly_YL2(L1,L2))
                    #print '%8.3lf %8.3lf | ' % (Ly.real, Ly.imag),
                    rxnabla_iiv[i1,i2,1] = Ly.real * radial_part

                    # Lz
                    Lz = (1j * YL1_Lz_YL2(L1,L2))
                    #print '%8.3lf %8.3lf | ' % (Lz.real, Lz.imag),
                    rxnabla_iiv[i1,i2,2] = Lz.real * radial_part

                    #print

                    # increase index 2
                    i2 += 1

            # increase index 1
            i1 += 1

    return rxnabla_iiv

