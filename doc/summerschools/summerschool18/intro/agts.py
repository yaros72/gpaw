from myqueue.task import task


def create_tasks():
    return [task('gpaw.utilities.nbrun', args=['intro.ipynb'])]
