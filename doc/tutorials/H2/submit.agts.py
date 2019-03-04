from myqueue.task import task


def create_tasks():
    return [
        task('atomize.py@1:30m'),
        task('relax.py@1:30m', deps='atomize.py')]
