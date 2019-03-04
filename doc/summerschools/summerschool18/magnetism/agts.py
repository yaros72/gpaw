from myqueue.task import task


def create_tasks():
    nbrun = 'gpaw.utilities.nbrun'
    return [
        task(nbrun, args=['magnetism1.master.ipynb'], tmax='1h'),
        task(nbrun, args=['magnetism2.master.ipynb'], tmax='2h')]
