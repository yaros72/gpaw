from math import pi
import numpy as np
from ase.units import Ha


def ekin(dataset):
    """Calculate PAW kinetic energy contribution as a function of G."""
    ds = dataset
    rgd = dataset.rgd
    de_j = ds.data.e_kin_jj.diagonal()
    e0 = -ds.Kc
    e_k = 0.0
    for f, l, de, phit in zip(ds.f_j, ds.l_j, de_j, ds.phit_j):
        if f == 0.0:
            continue
        phit_r = np.array([phit(r) for r in rgd.r_g])
        G_k, phit_k = rgd.fft(phit_r * rgd.r_g**(l + 1), l)
        e_k += f * 0.5 * phit_k**2 * G_k**4 / (2 * pi)**3
        e0 -= f * de

    return G_k, e_k, e0


def dekindecut(G, de, ecut):
    dG = G[1]
    G0 = (2 * ecut)**0.5
    g = int(G0 / dG)
    # linear interpolation:
    dedG = np.polyval(np.polyfit(G[g:g + 2], de[g:g + 2], 1), G0)
    dedecut = dedG / G0
    return dedecut


if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt
    from gpaw.setup import create_setup

    parser = argparse.ArgumentParser(
        description='Calculate approximation to the energy variation with '
        'plane-wave cutoff energy.  The approximation is to use the kinetic '
        'energy from a PAW atom, which can be calculated efficiently on '
        'a radial grid.')
    parser.add_argument('-d', '--derivative', type=float, metavar='ECUT',
                        help='Calculate derivative of energy correction with '
                        'respect to plane-wave cutoff energy.')
    parser.add_argument('name', help='Name of PAW dataset.')
    args = parser.parse_args()

    ds = create_setup(args.name)

    G, de, e0 = ekin(ds)
    dG = G[1]

    if args.derivative:
        dedecut = -dekindecut(G, de, args.derivative / Ha)
        print('de/decut({}, {} eV) = {:.6f}'
              .format(args.name, args.derivative, dedecut))
    else:
        de = (np.add.accumulate(de) - 0.5 * de[0] - 0.5 * de) * dG

        ecut = 0.5 * G**2 * Ha
        y = (de[-1] - de) * Ha
        plt.plot(ecut, y)
        plt.xlim(300, 1000)
        plt.ylim(0, y[ecut > 300][0])
        plt.show()
