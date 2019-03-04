import numpy as np

from gpaw.mixer import DummyMixer
from gpaw.xc import XC

from gpaw.lcaotddft.utilities import read_uMM
from gpaw.lcaotddft.utilities import write_uMM


class KickHamiltonian(object):
    def __init__(self, paw, ext):
        ham = paw.hamiltonian
        dens = paw.density
        vext_g = ext.get_potential(ham.finegd)
        self.vt_sG = [ham.restrict_and_collect(vext_g)]
        self.dH_asp = ham.setups.empty_atomic_matrix(1, ham.atom_partition)

        W_aL = dens.ghat.dict()
        dens.ghat.integrate(vext_g, W_aL)
        # XXX this is a quick hack to get the distribution right
        dHtmp_asp = ham.atomdist.to_aux(self.dH_asp)
        for a, W_L in W_aL.items():
            setup = dens.setups[a]
            dHtmp_asp[a] = np.dot(setup.Delta_pL, W_L).reshape((1, -1))
        self.dH_asp = ham.atomdist.from_aux(dHtmp_asp)


class TimeDependentHamiltonian(object):
    def __init__(self, fxc=None):
        assert fxc is None or isinstance(fxc, str)
        self.fxc_name = fxc

    def write(self, writer):
        if self.has_fxc:
            self.write_fxc(writer.child('fxc'))

    def write_fxc(self, writer):
        wfs = self.wfs
        writer.write(name=self.fxc_name)
        write_uMM(wfs, writer, 'deltaXC_H_uMM', self.deltaXC_H_uMM)

    def read(self, reader):
        if 'fxc' in reader:
            self.read_fxc(reader.fxc)

    def read_fxc(self, reader):
        assert self.fxc_name is None or self.fxc_name == reader.name
        self.fxc_name = reader.name
        self.deltaXC_H_uMM = read_uMM(self.wfs, reader, 'deltaXC_H_uMM')

    def initialize(self, paw):
        self.timer = paw.timer
        self.timer.start('Initialize TDDFT Hamiltonian')
        self.wfs = paw.wfs
        self.density = paw.density
        self.hamiltonian = paw.hamiltonian
        self.occupations = paw.occupations
        niter = paw.niter

        # Reset the density mixer
        # XXX: density mixer is not written to the gpw file
        # XXX: so we need to set it always
        self.density.set_mixer(DummyMixer())
        self.update()

        # Initialize fxc
        self.initialize_fxc(niter)
        self.timer.stop('Initialize TDDFT Hamiltonian')

    def initialize_fxc(self, niter):
        self.has_fxc = self.fxc_name is not None
        if not self.has_fxc:
            return
        self.timer.start('Initialize fxc')
        # XXX: Similar functionality is available in
        # paw.py: PAW.linearize_to_xc(self, newxc)
        # See test/lcaotddft/fxc_vs_linearize.py

        get_H_MM = self.get_hamiltonian_matrix

        # Calculate deltaXC: 1. take current H_MM
        if niter == 0:
            self.deltaXC_H_uMM = [None] * len(self.wfs.kpt_u)
            for u, kpt in enumerate(self.wfs.kpt_u):
                self.deltaXC_H_uMM[u] = get_H_MM(kpt, addfxc=False)

        # Update hamiltonian.xc
        if self.fxc_name == 'RPA':
            xc_name = 'null'
        else:
            xc_name = self.fxc_name
        # XXX: xc is not written to the gpw file
        # XXX: so we need to set it always
        xc = XC(xc_name)
        xc.initialize(self.density, self.hamiltonian, self.wfs,
                      self.occupations)
        xc.set_positions(self.hamiltonian.spos_ac)
        self.hamiltonian.xc = xc
        self.update()

        # Calculate deltaXC: 2. update with new H_MM
        if niter == 0:
            for u, kpt in enumerate(self.wfs.kpt_u):
                self.deltaXC_H_uMM[u] -= get_H_MM(kpt, addfxc=False)
        self.timer.stop('Initialize fxc')

    def update_projectors(self):
        self.timer.start('Update projectors')
        for kpt in self.wfs.kpt_u:
            self.wfs.atomic_correction.calculate_projections(self.wfs, kpt)
        self.timer.stop('Update projectors')

    def get_hamiltonian_matrix(self, kpt, addfxc=True):
        self.timer.start('Calculate H_MM')
        get_matrix = self.wfs.eigensolver.calculate_hamiltonian_matrix
        H_MM = get_matrix(self.hamiltonian, self.wfs, kpt, root=-1)
        if addfxc and self.has_fxc:
            kpt_rank, u = self.wfs.kd.get_rank_and_index(kpt.s, kpt.k)
            assert kpt_rank == self.wfs.kd.comm.rank
            H_MM += self.deltaXC_H_uMM[u]
        self.timer.stop('Calculate H_MM')
        return H_MM

    def update(self, mode='all'):
        self.timer.start('Update TDDFT Hamiltonian')
        if mode in ['all', 'density']:
            self.update_projectors()
            self.density.update(self.wfs)
        if mode in ['all']:
            self.hamiltonian.update(self.density)
        self.timer.stop('Update TDDFT Hamiltonian')
