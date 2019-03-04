# Creates: N2Ru_hollow.png, 2NadsRu.png, TS.xyz
from myqueue.task import task


def create_tasks():
    nbrun = 'gpaw.utilities.nbrun'
    return [
        task('check_convergence.py', tmax='5h', cores=8),
        task(nbrun, args=['convergence.ipynb'], deps='check_convergence.py'),
        task(nbrun, args=['n2_on_metal.master.ipynb'], tmax='6h'),
        task(nbrun, args=['neb.master.ipynb'], tmax='3h', cores=8,
             deps=nbrun + '+n2_on_metal.master.ipynb'),
        task(nbrun, args=['vibrations.master.ipynb'], tmax='9h', cores=8,
             deps=nbrun + '+neb.master.ipynb')]
