from myqueue.task import task


def create_tasks():
    return [
        task('basisgeneration.py@1:10m'),
        # task('lcao_opt.py@8:10m'),  # needs ELPA
        task('lcao_h2o.py@1:10m')]
