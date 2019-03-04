from ase.build import molecule
from gpaw.mpi import world
from gpaw import GPAW, Mixer

size = (world.size // 2, 2) if world.size > 1 else (1, 1)

energies = []
for use_elpa in [1, 0]:
    atoms = molecule('CH3CH2OH', vacuum=2.5)
    #atoms = molecule('H2', vacuum=3.0)
    calc = GPAW(mode='lcao', basis='dzp',
                h=0.25,
                parallel=dict(sl_default=(size[0], size[1], 3),
                              use_elpa=use_elpa),
                mixer=Mixer(0.5, 5, 50.0))
    atoms.calc = calc
    E = atoms.get_potential_energy()
    energies.append(E)
err = abs(energies[1] - energies[0])
assert err < 1e-10, err
