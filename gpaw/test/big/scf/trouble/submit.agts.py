from myqueue.task import task


def create_tasks():
    return [task('run.py+16@16:10h'),
            task('run.py+8@8:12h'),
            task('run.py+4@4:5h'),
            task('run.py+1@1:1h'),
            task('analyse.py@1:10m',
                 deps=['run.py+16', 'run.py+8', 'run.py+4', 'run.py+1'])]
