from myqueue.task import task


def create_tasks():
    return [
        task('Be_gs_8bands.py@2:20m'),
        task('Be_8bands_lrtddft.py@2:20m', deps='Be_gs_8bands.py'),
        task('Be_8bands_lrtddft_dE.py@2:20m', deps='Be_gs_8bands.py'),
        task('Na2_relax_excited.py@4:8h')]
