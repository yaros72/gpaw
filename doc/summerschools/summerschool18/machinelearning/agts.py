from myqueue.task import task


def create_tasks():
    nbrun = 'gpaw.utilities.nbrun'
    return [
        task(nbrun, args=['machinelearning.master.ipynb'], tmax='8h')]
