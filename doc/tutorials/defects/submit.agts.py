from myqueue.task import task


def create_tasks():
    tasks = [task('gaas.py+1@8:1h'),
             task('gaas.py+2@8:1h'),
             task('gaas.py+3@24:2h'),
             task('gaas.py+4@48:24h'),
             task('electrostatics.py@8:1:15m', deps=['gaas.py+1',
                                                     'gaas.py+2',
                                                     'gaas.py+3',
                                                     'gaas.py+4']),
             task('plot_energies.py', deps='electrostatics.py'),
             task('plot_potentials.py', deps='electrostatics.py'),
             task('BN.py+1@8:1h'),
             task('BN.py+2@8:1h'),
             task('BN.py+3@8:2h'),
             task('BN.py+4@24:2h'),
             task('BN.py+5@48:2h'),
             task('electrostatics_BN.py@1:15m', deps=['BN.py+1',
                                                      'BN.py+2',
                                                      'BN.py+3',
                                                      'BN.py+4',
                                                      'BN.py+5']),
             task('plot_energies_BN.py', deps='electrostatics_BN.py'),
             task('plot_potentials_BN.py', deps='electrostatics_BN.py'),
             task('plot_epsilon.py', deps='electrostatics_BN.py')]
    return tasks
