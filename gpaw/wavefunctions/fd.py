import numpy as np
from ase.units import Bohr

from gpaw.fd_operators import Laplace, Gradient
from gpaw.kpoint import KPoint
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.lfc import LocalizedFunctionsCollection as LFC
from gpaw.mpi import serial_comm
from gpaw.preconditioner import Preconditioner
from gpaw.projections import Projections
from gpaw.transformers import Transformer
from gpaw.utilities.blas import axpy
from gpaw.wavefunctions.arrays import UniformGridWaveFunctions
from gpaw.wavefunctions.fdpw import FDPWWaveFunctions
from gpaw.wavefunctions.mode import Mode
import _gpaw


class FD(Mode):
    name = 'fd'

    def __init__(self, nn=3, interpolation=3, force_complex_dtype=False):
        self.nn = nn
        self.interpolation = interpolation
        Mode.__init__(self, force_complex_dtype)

    def __call__(self, *args, **kwargs):
        return FDWaveFunctions(self.nn, *args, **kwargs)

    def todict(self):
        dct = Mode.todict(self)
        dct['nn'] = self.nn
        dct['interpolation'] = self.interpolation
        return dct


class FDWaveFunctions(FDPWWaveFunctions):
    mode = 'fd'

    def __init__(self, stencil, parallel, initksl,
                 gd, nvalence, setups, bd,
                 dtype, world, kd, kptband_comm, timer, reuse_wfs_method=None,
                 collinear=True):
        FDPWWaveFunctions.__init__(self, parallel, initksl,
                                   reuse_wfs_method=reuse_wfs_method,
                                   collinear=collinear,
                                   gd=gd, nvalence=nvalence, setups=setups,
                                   bd=bd, dtype=dtype, world=world, kd=kd,
                                   kptband_comm=kptband_comm, timer=timer)

        # Kinetic energy operator:
        self.kin = Laplace(self.gd, -0.5, stencil, self.dtype)

        self.taugrad_v = None  # initialized by MGGA functional

    def empty(self, n=(), global_array=False, realspace=False, q=-1):
        return self.gd.empty(n, self.dtype, global_array)

    def integrate(self, a_xg, b_yg=None, global_integral=True):
        return self.gd.integrate(a_xg, b_yg, global_integral)

    def bytes_per_wave_function(self):
        return self.gd.bytecount(self.dtype)

    def set_setups(self, setups):
        self.pt = LFC(self.gd, [setup.pt_j for setup in setups],
                      self.kd, dtype=self.dtype, forces=True)
        FDPWWaveFunctions.set_setups(self, setups)

    def set_positions(self, spos_ac, atom_partition=None):
        FDPWWaveFunctions.set_positions(self, spos_ac, atom_partition)

    def __str__(self):
        s = 'Wave functions: Uniform real-space grid\n'
        s += '  Kinetic energy operator: %s\n' % self.kin.description
        return s + FDPWWaveFunctions.__str__(self)

    def make_preconditioner(self, block=1):
        return Preconditioner(self.gd, self.kin, self.dtype, block)

    def apply_pseudo_hamiltonian(self, kpt, ham, psit_xG, Htpsit_xG):
        self.timer.start('Apply hamiltonian')
        self.kin.apply(psit_xG, Htpsit_xG, kpt.phase_cd)
        ham.apply_local_potential(psit_xG, Htpsit_xG, kpt.s)
        ham.xc.apply_orbital_dependent_hamiltonian(
            kpt, psit_xG, Htpsit_xG, ham.dH_asp)
        self.timer.stop('Apply hamiltonian')

    def get_pseudo_partial_waves(self):
        phit_aj = [setup.get_partial_waves_for_atomic_orbitals()
                   for setup in self.setups]
        return LFC(self.gd, phit_aj, kd=self.kd, cut=True, dtype=self.dtype)

    def add_to_density_from_k_point_with_occupation(self, nt_sG, kpt, f_n):
        # Used in calculation of response part of GLLB-potential
        nt_G = nt_sG[kpt.s]
        for f, psit_G in zip(f_n, kpt.psit_nG):
            # Same as nt_G += f * abs(psit_G)**2, but much faster:
            _gpaw.add_to_density(f, psit_G, nt_G)

        # Hack used in delta-scf calculations:
        if hasattr(kpt, 'c_on'):
            assert self.bd.comm.size == 1
            d_nn = np.zeros((self.bd.mynbands, self.bd.mynbands),
                            dtype=complex)
            for ne, c_n in zip(kpt.ne_o, kpt.c_on):
                d_nn += ne * np.outer(c_n.conj(), c_n)
            for d_n, psi0_G in zip(d_nn, kpt.psit_nG):
                for d, psi_G in zip(d_n, kpt.psit_nG):
                    if abs(d) > 1.e-12:
                        nt_G += (psi0_G.conj() * d * psi_G).real

    def calculate_kinetic_energy_density(self):
        if self.taugrad_v is None:
            self.taugrad_v = [
                Gradient(self.gd, v, n=3, dtype=self.dtype).apply
                for v in range(3)]

        assert not hasattr(self.kpt_u[0], 'c_on')
        if not isinstance(self.kpt_u[0].psit_nG, np.ndarray):
            return None

        taut_sG = self.gd.zeros(self.nspins)
        dpsit_G = self.gd.empty(dtype=self.dtype)
        for kpt in self.kpt_u:
            for f, psit_G in zip(kpt.f_n, kpt.psit_nG):
                for v in range(3):
                    self.taugrad_v[v](psit_G, dpsit_G, kpt.phase_cd)
                    axpy(0.5 * f, abs(dpsit_G)**2, taut_sG[kpt.s])

        self.kptband_comm.sum(taut_sG)
        return taut_sG

    def apply_mgga_orbital_dependent_hamiltonian(self, kpt, psit_xG,
                                                 Htpsit_xG, dH_asp,
                                                 dedtaut_G):
        a_G = self.gd.empty(dtype=psit_xG.dtype)
        for psit_G, Htpsit_G in zip(psit_xG, Htpsit_xG):
            for v in range(3):
                self.taugrad_v[v](psit_G, a_G, kpt.phase_cd)
                self.taugrad_v[v](dedtaut_G * a_G, a_G, kpt.phase_cd)
                axpy(-0.5, a_G, Htpsit_G)

    def ibz2bz(self, atoms):
        """Transform wave functions in IBZ to the full BZ."""

        assert self.kd.comm.size == 1

        # New k-point descriptor for full BZ:
        kd = KPointDescriptor(self.kd.bzk_kc, nspins=self.nspins)
        kd.set_communicator(serial_comm)

        self.pt = LFC(self.gd, [setup.pt_j for setup in self.setups],
                      kd, dtype=self.dtype)
        self.pt.set_positions(atoms.get_scaled_positions())

        self.initialize_wave_functions_from_restart_file()

        weight = 2.0 / kd.nspins / kd.nbzkpts

        # Build new list of k-points:
        kpt_u = []
        for s in range(self.nspins):
            for k in range(kd.nbzkpts):
                # Index of symmetry related point in the IBZ
                ik = self.kd.bz2ibz_k[k]
                r, u = self.kd.get_rank_and_index(s, ik)
                assert r == 0
                kpt = self.mykpts[u]

                phase_cd = np.exp(2j * np.pi * self.gd.sdisp_cd *
                                  kd.bzk_kc[k, :, np.newaxis])

                # New k-point:
                kpt2 = KPoint(weight, s, k, k, phase_cd)
                kpt2.f_n = kpt.f_n / kpt.weight / kd.nbzkpts * 2 / self.nspins
                kpt2.eps_n = kpt.eps_n.copy()

                # Transform wave functions using symmetry operation:
                Psit_nG = self.gd.collect(kpt.psit_nG)
                if Psit_nG is not None:
                    Psit_nG = Psit_nG.copy()
                    for Psit_G in Psit_nG:
                        Psit_G[:] = self.kd.transform_wave_function(Psit_G, k)
                kpt2.psit = UniformGridWaveFunctions(
                    self.bd.nbands, self.gd, self.dtype,
                    kpt=k, dist=(self.bd.comm, self.bd.comm.size),
                    spin=kpt.s, collinear=True)
                self.gd.distribute(Psit_nG, kpt2.psit_nG)
                # Calculate PAW projections:
                nproj_a = [setup.ni for setup in self.setups]
                kpt2.projections = Projections(
                    self.bd.nbands, nproj_a,
                    kpt.projections.atom_partition,
                    self.bd.comm,
                    collinear=True, spin=s, dtype=self.dtype)

                kpt2.psit.matrix_elements(self.pt, out=kpt2.projections)
                kpt_u.append(kpt2)

        self.kd = kd
        self.mykpts = kpt_u

    def _get_wave_function_array(self, u, n, realspace=True, periodic=False):
        assert realspace
        kpt = self.kpt_u[u]
        psit_G = kpt.psit_nG[n]
        if periodic and self.dtype == complex:
            k_c = self.kd.ibzk_kc[kpt.k]
            return self.gd.plane_wave(-k_c) * psit_G
        return psit_G

    def write(self, writer, write_wave_functions=False):
        FDPWWaveFunctions.write(self, writer)

        if not write_wave_functions:
            return

        writer.add_array(
            'values',
            (self.nspins, self.kd.nibzkpts, self.bd.nbands) +
            tuple(self.gd.get_size_of_global_array()),
            self.dtype)

        for s in range(self.nspins):
            for k in range(self.kd.nibzkpts):
                for n in range(self.bd.nbands):
                    psit_G = self.get_wave_function_array(n, k, s)
                    writer.fill(psit_G * Bohr**-1.5)

    def read(self, reader):
        FDPWWaveFunctions.read(self, reader)

        if 'values' not in reader.wave_functions:
            return

        c = reader.bohr**1.5
        if reader.version < 0:
            c = 1  # old gpw file

        for kpt in self.kpt_u:
            # We may not be able to keep all the wave
            # functions in memory - so psit_nG will be a special type of
            # array that is really just a reference to a file:
            psit_nG = reader.wave_functions.proxy('values', kpt.s, kpt.k)
            psit_nG.scale = c

            kpt.psit = UniformGridWaveFunctions(
                self.bd.nbands, self.gd, self.dtype, psit_nG,
                kpt=kpt.q, dist=(self.bd.comm, self.bd.comm.size),
                spin=kpt.s, collinear=True)

        if self.world.size > 1:
            # Read to memory:
            for kpt in self.kpt_u:
                kpt.psit.read_from_file()

    def initialize_from_lcao_coefficients(self, basis_functions):
        for kpt in self.mykpts:
            kpt.psit = UniformGridWaveFunctions(
                self.bd.nbands, self.gd, self.dtype, kpt=kpt.q,
                dist=(self.bd.comm, self.bd.comm.size, 1),
                spin=kpt.s, collinear=True)
            kpt.psit_nG[:] = 0.0
            mynbands = len(kpt.C_nM)
            basis_functions.lcao_to_grid(kpt.C_nM,
                                         kpt.psit_nG[:mynbands], kpt.q)
            kpt.C_nM = None

    def random_wave_functions(self, nao):
        """Generate random wave functions."""

        gpts = self.gd.N_c[0] * self.gd.N_c[1] * self.gd.N_c[2]

        if self.bd.nbands < gpts / 64:
            gd1 = self.gd.coarsen()
            gd2 = gd1.coarsen()

            psit_G1 = gd1.empty(dtype=self.dtype)
            psit_G2 = gd2.empty(dtype=self.dtype)

            interpolate2 = Transformer(gd2, gd1, 1, self.dtype).apply
            interpolate1 = Transformer(gd1, self.gd, 1, self.dtype).apply

            shape = tuple(gd2.n_c)
            scale = np.sqrt(12 / abs(np.linalg.det(gd2.cell_cv)))

            old_state = np.random.get_state()

            np.random.seed(4 + self.world.rank)

            for kpt in self.kpt_u:
                for psit_G in kpt.psit_nG[nao:]:
                    if self.dtype == float:
                        psit_G2[:] = (np.random.random(shape) - 0.5) * scale
                    else:
                        psit_G2.real = (np.random.random(shape) - 0.5) * scale
                        psit_G2.imag = (np.random.random(shape) - 0.5) * scale

                    interpolate2(psit_G2, psit_G1, kpt.phase_cd)
                    interpolate1(psit_G1, psit_G, kpt.phase_cd)
            np.random.set_state(old_state)

        elif gpts / 64 <= self.bd.nbands < gpts / 8:
            gd1 = self.gd.coarsen()

            psit_G1 = gd1.empty(dtype=self.dtype)

            interpolate1 = Transformer(gd1, self.gd, 1, self.dtype).apply

            shape = tuple(gd1.n_c)
            scale = np.sqrt(12 / abs(np.linalg.det(gd1.cell_cv)))

            old_state = np.random.get_state()

            np.random.seed(4 + self.world.rank)

            for kpt in self.kpt_u:
                for psit_G in kpt.psit_nG[nao:]:
                    if self.dtype == float:
                        psit_G1[:] = (np.random.random(shape) - 0.5) * scale
                    else:
                        psit_G1.real = (np.random.random(shape) - 0.5) * scale
                        psit_G1.imag = (np.random.random(shape) - 0.5) * scale

                    interpolate1(psit_G1, psit_G, kpt.phase_cd)
            np.random.set_state(old_state)

        else:
            shape = tuple(self.gd.n_c)
            scale = np.sqrt(12 / abs(np.linalg.det(self.gd.cell_cv)))

            old_state = np.random.get_state()

            np.random.seed(4 + self.world.rank)

            for kpt in self.kpt_u:
                for psit_G in kpt.psit_nG[nao:]:
                    if self.dtype == float:
                        psit_G[:] = (np.random.random(shape) - 0.5) * scale
                    else:
                        psit_G.real = (np.random.random(shape) - 0.5) * scale
                        psit_G.imag = (np.random.random(shape) - 0.5) * scale

            np.random.set_state(old_state)

    def estimate_memory(self, mem):
        FDPWWaveFunctions.estimate_memory(self, mem)
