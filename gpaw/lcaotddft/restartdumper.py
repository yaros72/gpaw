from gpaw.lcaotddft.observer import TDDFTObserver


class RestartDumper(TDDFTObserver):
    def __init__(self, paw, restart_filename, interval=100):
        TDDFTObserver.__init__(self, paw, interval)
        self.restart_filename = restart_filename

    def _update(self, paw):
        if paw.niter == 0:
            return
        paw.log('%s activated' % self.__class__.__name__)
        for obs, n, args, kwargs in paw.observers:
            if (isinstance(obs, TDDFTObserver) and
                hasattr(obs, 'write_restart')):
                obs.write_restart()
        paw.write(self.restart_filename, mode='all')
