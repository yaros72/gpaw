from __future__ import print_function
import numpy as np
from gpaw import GPAW
from ase.build import molecule
from gpaw.mpi import world

system = molecule('H2O')
system.cell = (4, 4, 4)
system.pbc = 1


for mode in ['fd',
             'pw',
             'lcao'
             ]:
    energy = []
    force = []
    stress = []

    if mode != 'lcao':
        eigensolver = 'rmm-diis'
    else:
        eigensolver = None

    domain = 1 + (world.size >= 2)
    band = 1 + (world.size >= 4)

    for augment_grids in 0, 1:
        if world.rank == 0:
            print(mode, augment_grids)
        calc = GPAW(mode=mode,
                    gpts=(16, 16, 16),
                    txt='gpaw.%s.%d.txt' % (mode, int(augment_grids)),
                    eigensolver=eigensolver,
                    parallel=dict(augment_grids=augment_grids,
                                  band=band, domain=domain),
                    basis='szp(dzp)',
                    kpts=[1, 1, 4],
                    nbands=8)

        def stopcalc():
            calc.scf.converged = True
        # Iterate enough for density to update so it depends on potential
        calc.attach(stopcalc, 3 if mode == 'lcao' else 5)
        system.set_calculator(calc)
        energy.append(system.get_potential_energy())
        force.append(system.get_forces())
        if mode == 'pw':
            stress.append(system.get_stress())
    ferr = np.abs(force[1] - force[0]).max()
    eerr = abs(energy[1] - energy[0])
    if mode == 'pw':
        serr = np.abs(stress[1] - stress[0]).max()

    assert eerr < 1e-10, eerr
    assert ferr < 1e-10, ferr
