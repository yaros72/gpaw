from myqueue.task import task


def create_tasks():
    return [
        task('ruslab.py@8:10h'),
        task('ruslab.py+H@8:10h'),
        task('ruslab.py+N@8:10h'),
        task('ruslab.py+O@16:15h'),
        task('molecules.py@8:20m'),
        task('results.py',
             deps=['ruslab.py', 'ruslab.py+H', 'ruslab.py+N',
                   'ruslab.py+O', 'molecules.py'])]
