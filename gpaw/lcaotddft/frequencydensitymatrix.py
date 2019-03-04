import numpy as np

from gpaw.io import Reader
from gpaw.io import Writer

from gpaw.tddft.folding import Frequency
from gpaw.tddft.folding import FoldedFrequencies
from gpaw.lcaotddft.observer import TDDFTObserver
from gpaw.lcaotddft.utilities import read_uMM
from gpaw.lcaotddft.utilities import read_wuMM
from gpaw.lcaotddft.utilities import write_uMM
from gpaw.lcaotddft.utilities import write_wuMM


class FrequencyDensityMatrix(TDDFTObserver):
    version = 1
    ulmtag = 'FDM'

    def __init__(self,
                 paw,
                 dmat,
                 filename=None,
                 frequencies=None,
                 restart_filename=None,
                 interval=1):
        TDDFTObserver.__init__(self, paw, interval)
        self.has_initialized = False
        self.dmat = dmat
        self.filename = filename
        self.restart_filename = restart_filename
        self.world = paw.world
        self.wfs = paw.wfs
        self.log = paw.log
        self.using_blacs = self.wfs.ksl.using_blacs
        if self.using_blacs:
            ksl_comm = self.wfs.ksl.block_comm
            kd_comm = self.wfs.kd.comm
            assert self.world.size == ksl_comm.size * kd_comm.size

        assert self.world.rank == self.wfs.world.rank

        if filename is not None:
            self.read(filename)
            return

        self.time = paw.time
        if isinstance(frequencies, FoldedFrequencies):
            frequencies = [frequencies]
        self.foldedfreqs_f = frequencies
        self.__generate_freq_w()
        self.Nw = np.sum([len(ff.frequencies) for ff in self.foldedfreqs_f])

    def initialize(self):
        if self.has_initialized:
            return

        if self.wfs.gd.pbc_c.any():
            self.rho0_dtype = complex
        else:
            self.rho0_dtype = float

        self.rho0_uMM = []
        for kpt in self.wfs.kpt_u:
            self.rho0_uMM.append(self.dmat.zeros(self.rho0_dtype))
        self.FReDrho_wuMM = []
        self.FImDrho_wuMM = []
        for w in range(self.Nw):
            self.FReDrho_wuMM.append([])
            self.FImDrho_wuMM.append([])
            for kpt in self.wfs.kpt_u:
                self.FReDrho_wuMM[-1].append(self.dmat.zeros(complex))
                self.FImDrho_wuMM[-1].append(self.dmat.zeros(complex))
        self.has_initialized = True

    def __generate_freq_w(self):
        self.freq_w = []
        for ff in self.foldedfreqs_f:
            for f in ff.frequencies:
                self.freq_w.append(Frequency(f, ff.folding, 'au'))

    def _update(self, paw):
        if paw.action == 'init':
            if self.time != paw.time:
                raise RuntimeError('Timestamp do not match with '
                                   'the calculator')
            self.initialize()
            if paw.niter == 0:
                rho_uMM = self.dmat.get_density_matrix(paw.niter)
                for u, kpt in enumerate(self.wfs.kpt_u):
                    rho_MM = rho_uMM[u]
                    if self.rho0_dtype == float:
                        assert np.max(np.absolute(rho_MM.imag)) == 0.0
                        rho_MM = rho_MM.real
                    self.rho0_uMM[u][:] = rho_MM
            return

        if paw.action == 'kick':
            return

        assert paw.action == 'propagate'

        time_step = paw.time - self.time
        self.time = paw.time

        # Complex exponentials with envelope
        exp_w = []
        for ff in self.foldedfreqs_f:
            exp_i = (np.exp(1.0j * ff.frequencies * self.time) *
                     ff.folding.envelope(self.time) * time_step)
            exp_w.extend(exp_i.tolist())

        rho_uMM = self.dmat.get_density_matrix((paw.niter, paw.action))
        for u, kpt in enumerate(self.wfs.kpt_u):
            Drho_MM = rho_uMM[u] - self.rho0_uMM[u]
            for w, exp in enumerate(exp_w):
                # Update Fourier transforms
                self.FReDrho_wuMM[w][u] += Drho_MM.real * exp
                self.FImDrho_wuMM[w][u] += Drho_MM.imag * exp

    def write_restart(self):
        if self.restart_filename is None:
            return
        self.write(self.restart_filename)

    def write(self, filename):
        self.log('%s: Writing to %s' % (self.__class__.__name__, filename))
        writer = Writer(filename, self.world, mode='w',
                        tag=self.__class__.ulmtag)
        writer.write(version=self.__class__.version)
        writer.write(time=self.time)
        writer.write(foldedfreqs_f=[ff.todict() for ff in self.foldedfreqs_f])
        wfs = self.wfs
        write_uMM(wfs, writer, 'rho0_uMM', self.rho0_uMM)
        wlist = range(self.Nw)
        write_wuMM(wfs, writer, 'FReDrho_wuMM', self.FReDrho_wuMM, wlist)
        write_wuMM(wfs, writer, 'FImDrho_wuMM', self.FImDrho_wuMM, wlist)
        writer.close()

    def read(self, filename):
        reader = Reader(filename)
        tag = reader.get_tag()
        if tag != self.__class__.ulmtag:
            raise RuntimeError('Unknown tag %s' % tag)
        version = reader.version
        if version != self.__class__.version:
            raise RuntimeError('Unknown version %s' % version)
        self.time = reader.time
        self.foldedfreqs_f = [FoldedFrequencies(**ff)
                              for ff in reader.foldedfreqs_f]
        self.__generate_freq_w()
        self.Nw = np.sum([len(ff.frequencies) for ff in self.foldedfreqs_f])
        wfs = self.wfs
        self.rho0_uMM = read_uMM(wfs, reader, 'rho0_uMM')
        self.rho0_dtype = self.rho0_uMM[0].dtype
        wlist = range(self.Nw)
        self.FReDrho_wuMM = read_wuMM(wfs, reader, 'FReDrho_wuMM', wlist)
        self.FImDrho_wuMM = read_wuMM(wfs, reader, 'FImDrho_wuMM', wlist)
        reader.close()
        self.has_initialized = True
