from myqueue.task import task


def create_tasks():
    return [
        task('bandstructure.py@1:5m')]
