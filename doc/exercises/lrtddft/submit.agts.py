from myqueue.task import task


def create_tasks():
    return [
        task('Na2TDDFT.py@2:1h'),
        task('part2.py', deps='Na2TDDFT.py'),
        task('ground_state.py@8:15s'),
        task('spectrum.py', deps='ground_state.py')]
