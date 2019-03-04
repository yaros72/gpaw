from myqueue.task import task


def create_tasks():
    return [task('PES_CO.py@8:1h'),
            task('PES_H2O.py@8:1h'),
            task('PES_NH3.py@8:55m'),
            task('PES_plot.py@1:5m',
                deps='PES_CO.py,PES_H2O.py,PES_NH3.py')]
