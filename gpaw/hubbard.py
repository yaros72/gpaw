import numpy as np

from gpaw.utilities import pack2, unpack2


def hubbard(setup, D_sp):
    nspins = len(D_sp)

    l_j = setup.l_j
    l = setup.Hubl
    scale = setup.Hubs
    nl = np.where(np.equal(l_j, l))[0]
    nn = (2 * np.array(l_j) + 1)[0:nl[0]].sum()

    e_xc = 0.0
    dH_sp = []

    s = 0
    for D_p in D_sp:
        N_mm, V = aoom(setup, unpack2(D_p), l, scale)
        N_mm = N_mm / 2 * nspins

        if nspins == 4:
            N_mm = N_mm / 2.0
            if s == 0:
                Eorb = setup.HubU / 2. * (N_mm - 
                                    0.5 * np.dot(N_mm, N_mm)).trace()

                Vorb = setup.HubU / 2. * (np.eye(2 * l +1) -N_mm)

            else:
                Eorb = setup.HubU / 2. * (-0.5*np.dot(N_mm , N_mm)).trace()

                Vorb = -setup.HubU / 2. * N_mm
        else:        
            Eorb = setup.HubU / 2. * (N_mm -
                                np.dot(N_mm, N_mm)).trace()

            Vorb = setup.HubU * (0.5 * np.eye(2 * l + 1) - N_mm)

        e_xc += Eorb
        if nspins == 1:
            # add contribution of other spin manyfold
            e_xc += Eorb

        if len(nl) == 2:
            mm = (2 * np.array(l_j) + 1)[0:nl[1]].sum()

            V[nn:nn + 2 * l + 1, nn:nn + 2 * l + 1] *= Vorb
            V[mm:mm + 2 * l + 1, nn:nn + 2 * l + 1] *= Vorb
            V[nn:nn + 2 * l + 1, mm:mm + 2 * l + 1] *= Vorb
            V[mm:mm + 2 * l + 1, mm:mm + 2 * l + 1] *= Vorb
        else:
            V[nn:nn + 2 * l + 1, nn:nn + 2 * l + 1] *= Vorb

        dH_sp.append(pack2(V))
        s += 1
        
    return e_xc, dH_sp


def aoom(setup, DM, l, scale=1):
    """Atomic Orbital Occupation Matrix.

    Determine the Atomic Orbital Occupation Matrix (aoom) for a
    given l-quantum number.

    This operation, takes the density matrix (DM), which for
    example is given by unpack2(D_asq[i][spin]), and corrects for
    the overlap between the selected orbitals (l) upon which the
    the density is expanded (ex <p|p*>,<p|p>,<p*|p*> ).

    Returned is only the "corrected" part of the density matrix,
    which represents the orbital occupation matrix for l=2 this is
    a 5x5 matrix.
    """
    S = setup
    l_j = S.l_j
    lq = S.lq
    nl = np.where(np.equal(l_j, l))[0]
    V = np.zeros(np.shape(DM))
    if len(nl) == 2:
        aa = nl[0] * len(l_j) - (nl[0] - 1) * nl[0] // 2
        bb = nl[1] * len(l_j) - (nl[1] - 1) * nl[1] // 2
        ab = aa + nl[1] - nl[0]

        if not scale:
            lq_a = lq[aa]
            lq_ab = lq[ab]
            lq_b = lq[bb]
        else:
            lq_a = 1
            lq_ab = lq[ab] / lq[aa]
            lq_b = lq[bb] / lq[aa]

        # and the correct entrances in the DM
        nn = (2 * np.array(l_j) + 1)[0:nl[0]].sum()
        mm = (2 * np.array(l_j) + 1)[0:nl[1]].sum()

        # finally correct and add the four submatrices of NC_DM
        A = DM[nn:nn + 2 * l + 1, nn:nn + 2 * l + 1] * lq_a
        B = DM[nn:nn + 2 * l + 1, mm:mm + 2 * l + 1] * lq_ab
        C = DM[mm:mm + 2 * l + 1, nn:nn + 2 * l + 1] * lq_ab
        D = DM[mm:mm + 2 * l + 1, mm:mm + 2 * l + 1] * lq_b

        V[nn:nn + 2 * l + 1, nn:nn + 2 * l + 1] = lq_a
        V[nn:nn + 2 * l + 1, mm:mm + 2 * l + 1] = lq_ab
        V[mm:mm + 2 * l + 1, nn:nn + 2 * l + 1] = lq_ab
        V[mm:mm + 2 * l + 1, mm:mm + 2 * l + 1] = lq_b

        return A + B + C + D, V
    else:
        assert len(nl) == 1
        assert l_j[-1] == l
        nn = (2 * np.array(l_j) + 1)[0:nl[0]].sum()
        A = DM[nn:nn + 2 * l + 1, nn:nn + 2 * l + 1] * lq[-1]
        V[nn:nn + 2 * l + 1, nn:nn + 2 * l + 1] = lq[-1]
        return A, V
