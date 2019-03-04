from myqueue.task import task


def create_tasks():
    return [
        task('calculate.py@1:1h'),
        task('plot.py', deps='calculate.py')]
