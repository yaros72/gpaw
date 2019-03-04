import traceback
from time import time
from glob import glob

import ase.db

import gpaw.mpi
from gpaw import GPAW

exec(open('params.py').read())  # get the calc() function
setup = globals()['calc']

c = ase.db.connect('results.db')

names = [name for name in glob('*.py') if name not in
         ['run.py', 'params.py', 'submit.agts.py', 'analyse.py']]

for name in names:
    namespace = {}
    exec(open(name).read(), namespace)
    atoms = namespace['atoms']
    ncpus = namespace.get('ncpus', 8)

    if gpaw.mpi.size != ncpus:
        continue

    name = name[:-3]

    id = c.reserve(name=name)
    if not id:
        continue

    if atoms.calc is None:
        atoms.calc = GPAW()

    setup(atoms)

    atoms.calc.set(txt=name + '.txt')

    t = time()
    try:
        e1 = atoms.get_potential_energy()
        ok = True
    except:
        ok = False
        if gpaw.mpi.rank == 0:
            traceback.print_exc(file=open(name + '.error', 'w'))
    t = time() - t

    c.write(atoms, name=name, ok=ok,
            time=t, iters=atoms.calc.get_number_of_iterations(), ncpus=ncpus)

    del c[id]
