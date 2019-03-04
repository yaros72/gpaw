from myqueue.task import task


def create_tasks():
    return [
        task('mnsi.py'),
        task('plot2d.py', deps='mnsi.py')]
