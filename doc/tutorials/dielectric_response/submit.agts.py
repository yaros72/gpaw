from myqueue.task import task


def create_tasks():
    return [
        task('plot_freq.py'),
        task('silicon_ABS_simpleversion.py'),
        task('plot_silicon_ABS_simple.py',
             deps='silicon_ABS_simpleversion.py'),
        task('silicon_ABS.py@16:1h'),
        task('plot_ABS.py', deps='silicon_ABS.py'),
        task('aluminum_EELS.py@8:1h'),
        task('plot_aluminum_EELS_simple.py', deps='aluminum_EELS.py'),
        task('graphite_EELS.py@8:1h'),
        task('plot_EELS.py', deps='graphite_EELS.py'),
        task('tas2_dielectric_function.py@8:15m'),
        task('graphene_dielectric_function.py@8:15m')]
