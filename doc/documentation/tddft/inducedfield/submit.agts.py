from myqueue.task import task


def create_tasks():
    return [
        task('timepropagation_calculate.py@8:1h'),
        task('timepropagation_continue.py@8:1h',
            deps='timepropagation_calculate.py'),
        task('timepropagation_postprocess.py@8:5m',
            deps='timepropagation_continue.py'),
        task('timepropagation_plot.py@1:5m',
            deps='timepropagation_postprocess.py'),
        task('casida_calculate.py@8:1h'),
        task('casida_postprocess.py@8:5m', deps='casida_calculate.py'),
        task('casida_plot.py@1:5m', deps='casida_postprocess.py')]
