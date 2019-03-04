# Creates: convergence.db
import numpy as np
from ase import Atoms
from ase.build import hcp0001
from ase.constraints import FixAtoms
from ase.optimize import BFGSLineSearch
from gpaw import GPAW, PW, Davidson

a = 2.72
c = 1.58 * a
vacuum = 4.0


def adsorb(db, height=1.2, nlayers=3, nkpts=7, ecut=400):
    """Adsorb nitrogen in hcp-site on Ru(0001) surface.

    Do calculations for N/Ru(0001), Ru(0001) and a nitrogen atom
    if they have not already been done.

    db: Database
        Database for collecting results.
    height: float
        Height of N-atom above top Ru-layer.
    nlayers: int
        Number of Ru-layers.
    nkpts: int
        Use a (nkpts * nkpts) Monkhorst-Pack grid that includes the
        Gamma point.
    ecut: float
        Cutoff energy for plane waves.

    Returns height.
    """

    name = f'Ru{nlayers}-{nkpts}x{nkpts}-{ecut:.0f}'

    parameters = dict(mode=PW(ecut),
                      eigensolver=Davidson(niter=2),
                      poissonsolver={'dipolelayer': 'xy'},
                      kpts={'size': (nkpts, nkpts, 1), 'gamma': True},
                      xc='PBE')

    # N/Ru(0001):
    slab = hcp0001('Ru', a=a, c=c, size=(1, 1, nlayers))
    z = slab.positions[:, 2].max() + height
    x, y = np.dot([2 / 3, 2 / 3], slab.cell[:2, :2])
    slab.append('N')
    slab.positions[-1] = [x, y, z]
    slab.center(vacuum=vacuum, axis=2)  # 2: z-axis

    # Fix first nlayer atoms:
    slab.constraints = FixAtoms(indices=list(range(nlayers)))

    id = db.reserve(name=f'N/{nlayers}Ru(0001)', nkpts=nkpts, ecut=ecut)
    if id is not None:  # skip calculation if already done
        slab.calc = GPAW(txt='N' + name + '.txt',
                         **parameters)
        optimizer = BFGSLineSearch(slab, logfile='N' + name + '.opt')
        optimizer.run(fmax=0.01)
        height = slab.positions[-1, 2] - slab.positions[:-1, 2].max()
        del db[id]
        db.write(slab,
                 name=f'N/{nlayers}Ru(0001)', nkpts=nkpts, ecut=ecut,
                 height=height)

    # Clean surface (single point calculation):
    id = db.reserve(name=f'{nlayers}Ru(0001)', nkpts=nkpts, ecut=ecut)
    if id is not None:
        del slab[-1]  # remove nitrogen atom
        slab.calc = GPAW(txt=name + '.txt',
                         **parameters)
        slab.get_forces()
        del db[id]
        db.write(slab,
                 name=f'{nlayers}Ru(0001)', nkpts=nkpts, ecut=ecut)

    # Nitrogen atom:
    id = db.reserve(name='N-atom', ecut=ecut)
    if id is not None:
        # Create spin-polarized nitrogen atom:
        molecule = Atoms('N', magmoms=[3])
        molecule.center(vacuum=4.0)
        # Remove parameters that make no sense for an isolated atom:
        del parameters['kpts']
        del parameters['poissonsolver']
        # Calculate energy:
        molecule.calc = GPAW(txt=name + '.txt', **parameters)
        molecule.get_potential_energy()
        del db[id]
        db.write(molecule, name='N-atom', ecut=ecut)

    return height


def run():
    h = 1.2
    from ase.db import connect
    db = connect('convergence.db')
    for n in range(1, 10):
        h = adsorb(db, h, n, 7, 400)
    for k in range(4, 18):
        h = adsorb(db, h, 2, k, 400)
    for ecut in range(350, 801, 50):
        h = adsorb(db, h, 2, 7, ecut)


if __name__ == '__main__':
    run()
