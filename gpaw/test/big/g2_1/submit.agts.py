from myqueue.task import task


def create_tasks():
    return [task('g21gpaw.py@1:20h'),
            task('analyse.py', deps='g21gpaw.py')]
