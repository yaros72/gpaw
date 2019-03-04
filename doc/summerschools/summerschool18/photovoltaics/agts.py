from myqueue.task import task


def create_tasks():
    nbrun = 'gpaw.utilities.nbrun'
    return [
        task(nbrun, args=['pv1.master.ipynb'], tmax='13h'),
        task(nbrun, args=['pv2.master.ipynb'], tmax='13h', cores=8,
             deps=nbrun + '+pv1.master.ipynb'),
        task(nbrun, args=['pv3.master.ipynb'], tmax='13h',
             deps=nbrun + '+pv2.master.ipynb')]
