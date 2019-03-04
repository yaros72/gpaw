from myqueue.task import task


def create_tasks():
    nbrun = 'gpaw.utilities.nbrun'
    t1 = task(nbrun, args=['batteries1.master.ipynb'], tmax='1h')
    t2 = task(nbrun, args=['batteries2.master.ipynb'], tmax='3h')
    t3 = task(nbrun, args=['batteries3.master.ipynb'], tmax='1h', cores=8,
              deps=[t1, t2])
    return [t1, t2, t3]
