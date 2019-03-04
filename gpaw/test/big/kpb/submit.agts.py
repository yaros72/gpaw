from myqueue.task import task


def create_tasks():
    return [
        task('molecules.py@1:1h'),
        task('check.py', deps='molecules.py')]
