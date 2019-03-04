from gpaw import GPAW, PW, MethfesselPaxton
from ase.spacegroup import crystal
from ase.io import write

a = 4.55643
mnsi = crystal(['Mn', 'Si'],
               [(0.1380, 0.1380, 0.1380), (0.84620, 0.84620, 0.84620)],
               spacegroup=198,
               cellpar=[a, a, a, 90, 90, 90])


for atom in mnsi:
    if atom.symbol == 'Mn':
        atom.magmom = 0.5

mnsi.calc = GPAW(xc='PBE',
                 kpts=(2, 2, 2),
                 mode=PW(800),
                 occupations=MethfesselPaxton(width=0.005),
                 txt='mnsi.txt')

mnsi.get_potential_energy()
mnsi.calc.write('mnsi.gpw')
v = mnsi.calc.get_electrostatic_potential()
write('mnsi.cube', mnsi, data=v)

assert abs(v.max() - 13.43) < 0.01
