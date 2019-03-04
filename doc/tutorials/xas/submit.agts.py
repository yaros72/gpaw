# Creates: xas_h2o_spectrum.png, h2o_xas_box.png


def create_tasks():
    from myqueue.task import task
    return [
        task('setups.py'),
        task('run.py@8:25m', deps='setups.py'),
        task('dks.py@8:25m', deps='setups.py'),
        task('h2o_xas_box1.py@8:25m', deps='setups.py'),
        task('submit.agts.py', deps='run.py,dks.py,h2o_xas_box1.py')]


if __name__ == '__main__':
    from gpaw.test import equal
    exec(open('plot.py').read())
    e_dks = float(open('dks.result').readline().split()[2])
    equal(e_dks, 532.502, 0.001)
    exec(open('h2o_xas_box2.py').read())
