def create_tasks():
    # Most of these time out at the moment ...
    return []


"""
    gs_H_lda = queue.add('H.ralda_01_lda.py', ncpus=2, walltime=5)
    queue.add('H.ralda_02_rpa_at_lda.py', deps=gs_H_lda, ncpus=16, walltime=20)
    queue.add('H.ralda_03_ralda.py', deps=gs_H_lda, ncpus=16, walltime=200)
    gs_H_pbe = queue.add('H.ralda_04_pbe.py', ncpus=2, walltime=5)
    queue.add('H.ralda_05_rpa_at_pbe.py', deps=gs_H_pbe, ncpus=16, walltime=20)
    queue.add('H.ralda_06_rapbe.py', deps=gs_H_pbe, ncpus=16, walltime=200)
    gs_CO = queue.add('CO.ralda_01_pbe+exx.py', ncpus=1, walltime=1000)
    queue.add('CO.ralda_02_CO_rapbe.py', deps=gs_CO, ncpus=16, walltime=2000)
    queue.add('CO.ralda_03_C_rapbe.py', deps=gs_CO, ncpus=16, walltime=2000)
    queue.add('CO.ralda_04_O_rapbe.py', deps=gs_CO, ncpus=16, walltime=2000)
    gs_diamond = queue.add('diamond.ralda_01_pbe.py', deps=gs_CO,
                           ncpus=1, walltime=100)
    queue.add('diamond.ralda_02_rapbe_rpa.py', deps=gs_diamond,
              ncpus=16, walltime=1200)
    diamkern_gs = queue.add('diam_kern.ralda_01_lda.py', ncpus=8, walltime=2)
    diamkern_ralda = queue.add('diam_kern.ralda_03_ralda_wave.py',
                               deps=diamkern_gs, ncpus=8, walltime=5)
    diamkern_raldac = queue.add('diam_kern.ralda_04_raldac.py',
                                deps=diamkern_ralda, ncpus=8, walltime=5)
    diamkern_JGMs = queue.add('diam_kern.ralda_05_jgm.py',
                              deps=diamkern_raldac, ncpus=8, walltime=5)
    diamkern_CPdyn = queue.add('diam_kern.ralda_06_CP_dyn.py',
                               deps=diamkern_JGMs, ncpus=8, walltime=10)
    diamkern_rangeRPA = queue.add('diam_kern.ralda_07_range_rpa.py',
                                  deps=diamkern_CPdyn, ncpus=8, walltime=5)
    diamkern_RPA = queue.add('diam_kern.ralda_08_rpa.py',
                             deps=diamkern_rangeRPA, ncpus=8, walltime=5)
    queue.add('diam_kern.ralda_09_compare.py',
              deps=diamkern_RPA, ncpus=1, walltime=5)
"""
