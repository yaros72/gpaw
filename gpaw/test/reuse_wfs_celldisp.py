import numpy as np
from gpaw import GPAW, Mixer
from gpaw.mpi import world
from ase.build import molecule

# Place one atom next to cell boundary, then check that reuse_wfs
# works correctly when atom is subsequently displaced across the
# boundary, i.e., that the kpoint phases of the PAW correction
# are handled correctly when unprojecting/reprojecting the wavefunctions.


def check(reuse):
    atoms = molecule('H2')
    atoms.pbc = 1
    atoms.center(vacuum=1.5)
    atoms.positions -= atoms.positions[1]
    dz = 1e-2
    atoms.positions[:, 2] += dz

    calc = GPAW(mode='pw',
                txt='gpaw.txt',
                nbands=1,
                experimental=dict(reuse_wfs_method='paw' if reuse else None),
                kpts=[[-0.3, 0.4, 0.2]],
                symmetry='off',
                mixer=Mixer(0.7, 5, 50.0))
    atoms.calc = calc

    first_iter_err = []

    def monitor():
        logerr = np.log10(calc.wfs.eigensolver.error)
        n = calc.scf.niter
        if n == 1:
            first_iter_err.append(logerr)

        if world.rank == 0:
            print('iter', n, 'err', logerr)
    calc.attach(monitor, 1)

    atoms.get_potential_energy()
    logerr1 = first_iter_err.pop()

    atoms.positions[:, 2] -= 2 * dz
    atoms.get_potential_energy()

    logerr2 = first_iter_err.pop()

    if world.rank == 0:
        print('reuse={}'.format(bool(reuse)))
        print('logerr1', logerr1)
        print('logerr2', logerr2)
        gain = logerr2 - logerr1
        print('gain', gain)
    return logerr2


noreuse_logerr = check(0)
reuse_logerr = check(1)
# Ref values: logerr=-3.6 without reuse_wfs and -5.0 with reuse_wfs
assert reuse_logerr < -4.8, reuse_logerr
assert reuse_logerr < noreuse_logerr - 1.2, (reuse_logerr, noreuse_logerr)
