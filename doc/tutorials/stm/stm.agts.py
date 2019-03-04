from myqueue.task import task


def create_tasks():
    return [
        task('al111.py'),
        task('stm.py', deps='al111.py')]
