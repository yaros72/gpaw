# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import numpy as np

_gaunt = {}
_nabla = {}


def gaunt(lmax=2):
    if lmax in _gaunt:
        return _gaunt[lmax]

    Lmax = (lmax + 1)**2
    L2max = (2 * lmax + 1)**2

    from gpaw.spherical_harmonics import YL, gam
    G_LLL = np.zeros((Lmax, L2max, L2max))
    for L1 in range(Lmax):
        for L2 in range(L2max):
            for L in range(L2max):
                r = 0.0
                for c1, n1 in YL[L1]:
                    for c2, n2 in YL[L2]:
                        for c, n in YL[L]:
                            nx = n1[0] + n2[0] + n[0]
                            ny = n1[1] + n2[1] + n[1]
                            nz = n1[2] + n2[2] + n[2]
                            r += c * c1 * c2 * gam(nx, ny, nz)
                G_LLL[L1, L2, L] = r
    _gaunt[lmax] = G_LLL
    return G_LLL

    
def nabla(lmax=2):
    """Create the array of derivative intergrals.

    ::

      /  ^     1-l' d   l'
      | dr Y  r    ---(r Y  )
      /     L      dr     L'
                     v
    """

    if lmax in _nabla:
        return _nabla[lmax]
        
    Lmax = (lmax + 1)**2
    from gpaw.spherical_harmonics import YL, gam
    Y_LLv = np.zeros((Lmax, Lmax, 3))
    # Insert new values
    for L1 in range(Lmax):
        for L2 in range(Lmax):
            for v in range(3):
                r = 0.0
                for c1, n1 in YL[L1]:
                    for c2, n2 in YL[L2]:
                        n = [0, 0, 0]
                        n[0] = n1[0] + n2[0]
                        n[1] = n1[1] + n2[1]
                        n[2] = n1[2] + n2[2]
                        if n2[v] > 0:
                            # apply derivative
                            n[v] -= 1
                            # add integral
                            r += n2[v] * c1 * c2 * gam(n[0], n[1], n[2])
                Y_LLv[L1, L2, v] = r
    _nabla[lmax] = Y_LLv
    return Y_LLv
