from myqueue.task import task


def create_tasks():
    return [
        task('gs_BaTiO3.py@8:30m'),
        task('polarization_BaTiO3.py@8:1h', deps='gs_BaTiO3.py'),
        task('born_BaTiO3.py@8:10h', deps='gs_BaTiO3.py'),
        task('get_borncharges.py', deps='born_BaTiO3.py'),
        task('gs_Sn.py@8:30m'),
        task('Sn_parallel_transport.py@8:5h', deps='gs_Sn.py'),
        task('plot_phase.py', deps='Sn_parallel_transport.py')]
