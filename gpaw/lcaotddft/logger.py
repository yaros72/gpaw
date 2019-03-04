from time import localtime
from math import log as ln

from gpaw.lcaotddft.observer import TDDFTObserver
from gpaw.tddft.units import autime_to_attosec


class TDDFTLogger(TDDFTObserver):

    def __init__(self, paw, flush_interval=10, interval=1):
        TDDFTObserver.__init__(self, paw, interval)
        assert flush_interval > 0
        self.flush_interval = flush_interval
        self.flush_next = paw.niter + self.flush_interval - 1

    def _update(self, paw):
        if paw.action == 'init':
            self._write_header(paw)
        self._write_data(paw)

    def _write_header(self, paw):
        paw.log('Logging time propagation')
        paw.log('------------------------')
        line = ('      %4s %9s %11s %9s' %
                ('iter', 'realtime', 'calctime', 'log(norm)'))
        paw.log(line)
        paw.log.flush()

    def _write_data(self, paw):
        density = paw.density
        norm = density.finegd.integrate(density.rhot_g)
        T = localtime()
        paw.log('iter: %4d  %02d:%02d:%02d %11.2f %9.1f' %
                (paw.niter, T[3], T[4], T[5],
                 paw.time * autime_to_attosec,
                 ln(abs(norm) + 1e-16) / ln(10)))
        if paw.niter > self.flush_next:
            paw.log.flush()
            while paw.niter > self.flush_next:
                self.flush_next += self.flush_interval
