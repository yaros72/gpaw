from myqueue.task import task


def create_tasks():
    return [
        task('nio.py'),
        task('n.py'),
        task('check.py', deps='n.py')]
