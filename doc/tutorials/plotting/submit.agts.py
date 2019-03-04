from myqueue.task import task


def create_tasks():
    return [
        task('CO.py@1:10s'),
        task('CO2cube.py@1:10s', deps='CO.py'),
        task('CO2plt.py@1:10s', deps='CO.py')]
