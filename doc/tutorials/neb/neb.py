from ase.io import read
from ase.constraints import FixAtoms
from ase.neb import NEB
from ase.optimize import BFGS
from ase.parallel import rank, size

from gpaw import GPAW

initial = read('initial.traj')
final = read('final.traj')

constraint = FixAtoms(mask=[atom.tag > 1 for atom in initial])

n = size // 3      # number of cpu's per image
j = 1 + rank // n  # my image number
assert 3 * n == size

images = [initial]

for i in range(3):
    ranks = range(i * n, (i + 1) * n)
    image = initial.copy()

    if rank in ranks:

        calc = GPAW(h=0.3,
                    kpts=(2, 2, 1),
                    txt='neb{}.txt'.format(j),
                    communicator=ranks)

        image.set_calculator(calc)

    image.set_constraint(constraint)
    images.append(image)

images.append(final)

neb = NEB(images, parallel=True, climb=True)
neb.interpolate()

qn = BFGS(neb, logfile='qn.log', trajectory='neb.traj')
qn.run(fmax=0.05)
