# Creates: Ec_rpa.png
from myqueue.task import task


def create_tasks():
    return [task('si.groundstate.py'),
            task('si.range_rpa.py@8:30m', deps='si.groundstate.py'),
            task('si.compare.py', deps='si.range_rpa.py'),
            task('plot_ec.py', deps='si.range_rpa.py')]
