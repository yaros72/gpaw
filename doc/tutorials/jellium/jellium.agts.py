from myqueue.task import task


def create_tasks():
    return [
        task('bulk.py@4:6m'),
        task('surface.py@4:6m'),
        task('sigma.py', deps='bulk.py,surface.py'),
        task('fig2.py', deps='sigma.py')]
