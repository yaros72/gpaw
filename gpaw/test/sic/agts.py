def create_tasks():
    from myqueue.task import task
    return [task('scfsic_n2.py@8:10m')]
