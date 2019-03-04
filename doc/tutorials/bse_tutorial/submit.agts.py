from myqueue.task import task


def create_tasks():
    return [
        task('gs_Si.py@4:20m'),
        task('eps_Si.py@4:6h', deps='gs_Si.py'),
        task('plot_Si.py@1:10m', deps='eps_Si.py'),
        task('gs_MoS2.py@4:1h'),
        task('pol_MoS2.py@64:33h', deps='gs_MoS2.py'),
        task('plot_MoS2.py@1:10m', deps='pol_MoS2.py'),
        task('get_2d_eps.py@1:8h', deps='gs_MoS2.py'),
        task('plot_2d_eps.py@1:10m', deps='get_2d_eps.py'),
        task('alpha_MoS2.py@1:10m', deps='gs_MoS2.py')]
