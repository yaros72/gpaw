from myqueue.task import task


def create_tasks():
    return [
        task('gs.py@8:10m'),
        task('td.py@8:30m', deps='gs.py'),
        task('tdc.py@8:30m', deps='td.py'),
        task('td_replay.py@8:30m', deps='tdc.py'),
        task('spectrum.py@1:2m', deps='tdc.py'),
        task('td_fdm_replay.py@1:5m', deps='tdc.py'),
        task('ksd_init.py@1:5m', deps='gs.py'),
        task('fdm_ind.py@1:2m', deps='td_fdm_replay.py,ksd_init.py'),
        task('spec_plot.py@1:2m', deps='spectrum.py'),
        task('tcm_plot.py@1:2m',
             deps='ksd_init.py,td_fdm_replay.py,spectrum.py'),
        task('ind_plot.py@1:2m', deps='fdm_ind.py')]
