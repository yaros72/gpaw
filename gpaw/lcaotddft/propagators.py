import numpy as np
from numpy.linalg import inv, solve

from gpaw import debug
from gpaw.tddft.units import au_to_as
from gpaw.utilities.scalapack import (pblas_simple_hemm, pblas_simple_gemm,
                                      scalapack_inverse, scalapack_solve,
                                      scalapack_zero, pblas_tran,
                                      scalapack_set)


def create_propagator(name, **kwargs):
    if name is None:
        return create_propagator('sicn')
    elif isinstance(name, Propagator):
        return name
    elif isinstance(name, dict):
        kwargs.update(name)
        return create_propagator(**kwargs)
    elif name == 'sicn':
        return SICNPropagator(**kwargs)
    elif name == 'ecn':
        return ECNPropagator(**kwargs)
    elif name.endswith('.ulm'):
        return ReplayPropagator(name, **kwargs)
    else:
        raise ValueError('Unknown propagator: %s' % name)


def equal(a, b, eps=1e-8):
    return abs(a - b) < eps


class Propagator(object):

    def __init__(self):
        object.__init__(self)

    def initialize(self, paw):
        self.timer = paw.timer
        self.log = paw.log

    def kick(self, hamiltonian, time):
        raise NotImplementedError()

    def propagate(self, time, time_step):
        raise NotImplementedError()

    def control_paw(self, paw):
        raise NotImplementedError()

    def todict(self):
        raise NotImplementedError()

    def get_description(self):
        return '%s' % self.__class__.__name__


class LCAOPropagator(Propagator):

    def __init__(self):
        Propagator.__init__(self)

    def initialize(self, paw):
        Propagator.initialize(self, paw)
        self.wfs = paw.wfs
        self.density = paw.density
        self.hamiltonian = paw.td_hamiltonian


class ReplayPropagator(LCAOPropagator):

    def __init__(self, filename, update='all'):
        from gpaw.lcaotddft.wfwriter import WaveFunctionReader
        LCAOPropagator.__init__(self)
        self.filename = filename
        self.update_mode = update
        self.reader = WaveFunctionReader(self.filename)
        self.read_index = 1
        self.read_count = len(self.reader)

    def _align_read_index(self, time):
        while self.read_index < self.read_count:
            r = self.reader[self.read_index]
            if equal(r.time, time):
                break
            self.read_index += 1
        if self.read_index == self.read_count:
            raise RuntimeError('Time not found: %f' % time)

    def _read(self):
        reader = self.reader[self.read_index]
        r = reader.wave_functions
        self.wfs.read_wave_functions(r)
        self.wfs.read_occupations(r)
        self.read_index += 1

    def kick(self, hamiltonian, time):
        self._align_read_index(time)
        # Check that this is the step after kick
        assert not equal(self.reader[self.read_index].time,
                         self.reader[self.read_index + 1].time)
        self._read()
        self.hamiltonian.update(self.update_mode)

    def propagate(self, time, time_step):
        next_time = time + time_step
        self._align_read_index(next_time)
        self._read()
        self.hamiltonian.update(self.update_mode)
        return next_time

    def control_paw(self, paw):
        # Read the initial state
        index = 1
        r = self.reader[index]
        assert r.action == 'init'
        assert equal(r.time, paw.time)
        self.read_index = index
        self._read()
        index += 1
        # Read the rest
        while index < self.read_count:
            r = self.reader[index]
            if r.action == 'init':
                index += 1
            elif r.action == 'kick':
                assert equal(r.time, paw.time)
                paw.absorption_kick(r.kick_strength)
                assert equal(r.time, paw.time)
                index += 1
            elif r.action == 'propagate':
                # Skip earlier times
                if r.time < paw.time or equal(r.time, paw.time):
                    index += 1
                    continue
                # Count the number of steps with the same time step
                time = paw.time
                time_step = r.time - time
                iterations = 0
                while index < self.read_count:
                    r = self.reader[index]
                    if (r.action != 'propagate' or
                        not equal(r.time - time, time_step)):
                        break
                    iterations += 1
                    time = r.time
                    index += 1
                # Propagate
                paw.propagate(time_step * au_to_as, iterations)
                assert equal(time, paw.time)
            else:
                raise RuntimeError('Unknown action: %s' % r.action)

    def __del__(self):
        self.reader.close()

    def todict(self):
        return {'name': self.filename,
                'update': self.update_mode}

    def get_description(self):
        lines = [self.__class__.__name__]
        lines += ['    File: %s' % (self.filename)]
        lines += ['    Update: %s' % (self.update_mode)]
        return '\n'.join(lines)


class ECNPropagator(LCAOPropagator):

    def __init__(self):
        LCAOPropagator.__init__(self)

    def initialize(self, paw, hamiltonian=None):
        LCAOPropagator.initialize(self, paw)
        if hamiltonian is not None:
            self.hamiltonian = hamiltonian

        ksl = self.wfs.ksl
        self.blacs = ksl.using_blacs
        if self.blacs:
            from gpaw.blacs import Redistributor
            self.log('BLACS Parallelization')

            # Parallel grid descriptors
            grid = ksl.blockgrid
            assert grid.nprow * grid.npcol == ksl.block_comm.size
            self.mm_block_descriptor = ksl.mmdescriptor
            self.Cnm_block_descriptor = grid.new_descriptor(ksl.bd.nbands,
                                                            ksl.nao,
                                                            ksl.blocksize,
                                                            ksl.blocksize)
            self.CnM_unique_descriptor = ksl.nM_unique_descriptor

            # Redistributors
            self.Cnm2nM = Redistributor(ksl.block_comm,
                                        self.Cnm_block_descriptor,
                                        self.CnM_unique_descriptor)
            self.CnM2nm = Redistributor(ksl.block_comm,
                                        self.CnM_unique_descriptor,
                                        self.Cnm_block_descriptor)

            if debug:
                nao = ksl.nao
                self.MM_descriptor = grid.new_descriptor(nao, nao, nao, nao)
                self.mm2MM = Redistributor(ksl.block_comm,
                                           self.mm_block_descriptor,
                                           self.MM_descriptor)
                self.MM2mm = Redistributor(ksl.block_comm,
                                           self.MM_descriptor,
                                           self.mm_block_descriptor)

            for kpt in self.wfs.kpt_u:
                scalapack_zero(self.mm_block_descriptor, kpt.S_MM, 'U')
                scalapack_zero(self.mm_block_descriptor, kpt.T_MM, 'U')

    def kick(self, hamiltonian, time):
        # Propagate
        get_matrix = self.wfs.eigensolver.calculate_hamiltonian_matrix
        for kpt in self.wfs.kpt_u:
            Vkick_MM = get_matrix(hamiltonian, self.wfs, kpt,
                                  add_kinetic=False, root=-1)
            for i in range(10):
                self.propagate_wfs(kpt.C_nM, kpt.C_nM, kpt.S_MM, Vkick_MM, 0.1)

        # Update Hamiltonian (and density)
        self.hamiltonian.update()

    def propagate(self, time, time_step):
        for kpt in self.wfs.kpt_u:
            H_MM = self.hamiltonian.get_hamiltonian_matrix(kpt)
            self.propagate_wfs(kpt.C_nM, kpt.C_nM, kpt.S_MM, H_MM, time_step)
        self.hamiltonian.update()
        return time + time_step

    def propagate_wfs(self, sourceC_nM, targetC_nM, S_MM, H_MM, dt):
        self.timer.start('Linear solve')

        if self.blacs:
            # XXX, Preallocate
            target_blockC_nm = self.Cnm_block_descriptor.empty(dtype=complex)
            temp_blockC_nm = self.Cnm_block_descriptor.empty(dtype=complex)
            temp_block_mm = self.mm_block_descriptor.empty(dtype=complex)
            if self.density.gd.comm.rank != 0:
                # XXX Fake blacks nbands, nao, nbands, nao grid because some
                # weird asserts
                # (these are 0,x or x,0 arrays)
                sourceC_nM = self.CnM_unique_descriptor.zeros(dtype=complex)

            # 1. target = (S+0.5j*H*dt) * source
            # Wave functions to target
            self.CnM2nm.redistribute(sourceC_nM, temp_blockC_nm)

            # XXX It can't be this f'n hard to symmetrize a matrix (tri2full)
            # Remove upper diagonal
            scalapack_zero(self.mm_block_descriptor, H_MM, 'U')
            # Lower diagonal matrix:
            temp_block_mm[:] = S_MM - (0.5j * dt) * H_MM
            scalapack_set(self.mm_block_descriptor, temp_block_mm, 0, 0, 'U')
            # Note it's strictly lower diagonal matrix
            # Add transpose of H
            pblas_tran(-0.5j * dt, H_MM, 1.0, temp_block_mm,
                       self.mm_block_descriptor, self.mm_block_descriptor)
            # Add transpose of S
            pblas_tran(1.0, S_MM, 1.0, temp_block_mm,
                       self.mm_block_descriptor, self.mm_block_descriptor)

            pblas_simple_gemm(self.Cnm_block_descriptor,
                              self.mm_block_descriptor,
                              self.Cnm_block_descriptor,
                              temp_blockC_nm,
                              temp_block_mm,
                              target_blockC_nm)
            # 2. target = (S-0.5j*H*dt)^-1 * target
            # temp_block_mm[:] = S_MM + (0.5j*dt) * H_MM
            # XXX It can't be this f'n hard to symmetrize a matrix (tri2full)
            # Lower diagonal matrix:
            temp_block_mm[:] = S_MM + (0.5j * dt) * H_MM
            # Not it's stricly lower diagonal matrix:
            scalapack_set(self.mm_block_descriptor, temp_block_mm, 0, 0, 'U')
            # Add transpose of H:
            pblas_tran(+0.5j * dt, H_MM, 1.0, temp_block_mm,
                       self.mm_block_descriptor, self.mm_block_descriptor)
            # Add transpose of S
            pblas_tran(1.0, S_MM, 1.0, temp_block_mm,
                       self.mm_block_descriptor, self.mm_block_descriptor)

            scalapack_solve(self.mm_block_descriptor,
                            self.Cnm_block_descriptor,
                            temp_block_mm,
                            target_blockC_nm)

            if self.density.gd.comm.rank != 0:  # XXX is this correct?
                # XXX Fake blacks nbands, nao, nbands, nao grid because some
                # weird asserts
                # (these are 0,x or x,0 arrays)
                target = self.CnM_unique_descriptor.zeros(dtype=complex)
            else:
                target = targetC_nM
            self.Cnm2nM.redistribute(target_blockC_nm, target)
            self.density.gd.comm.broadcast(targetC_nM, 0)  # Is this required?
        else:
            # Note: The full equation is conjugated (therefore -+, not +-)
            targetC_nM[:] = \
                solve(S_MM - 0.5j * H_MM * dt,
                      np.dot(S_MM + 0.5j * H_MM * dt,
                             sourceC_nM.T.conjugate())).T.conjugate()

        self.timer.stop('Linear solve')

    def blacs_mm_to_global(self, H_mm):
        if not debug:
            raise RuntimeError('Use debug mode')
        # Someone could verify that this works and remove the error.
        raise NotImplementedError('Method untested and thus unreliable')
        target = self.MM_descriptor.empty(dtype=complex)
        self.mm2MM.redistribute(H_mm, target)
        self.wfs.world.barrier()
        return target

    def blacs_nm_to_global(self, C_nm):
        # Someone could verify that this works and remove the error.
        raise NotImplementedError('Method untested and thus unreliable')
        target = self.CnM_unique_descriptor.empty(dtype=complex)
        self.Cnm2nM.redistribute(C_nm, target)
        self.wfs.world.barrier()
        return target

    def todict(self):
        return {'name': 'ecn'}


class SICNPropagator(ECNPropagator):

    def __init__(self):
        ECNPropagator.__init__(self)

    def initialize(self, paw):
        ECNPropagator.initialize(self, paw)
        # Allocate kpt.C2_nM arrays
        for kpt in self.wfs.kpt_u:
            kpt.C2_nM = np.empty_like(kpt.C_nM)

    def propagate(self, time, time_step):
        # --------------
        # Predictor step
        # --------------
        # 1. Store current C_nM
        self.save_wfs()  # kpt.C2_nM = kpt.C_nM
        for kpt in self.wfs.kpt_u:
            # H_MM(t) = <M|H(t)|M>
            kpt.H0_MM = self.hamiltonian.get_hamiltonian_matrix(kpt)
            # 2. Solve Psi(t+dt) from
            #    (S_MM - 0.5j*H_MM(t)*dt) Psi(t+dt)
            #       = (S_MM + 0.5j*H_MM(t)*dt) Psi(t)
            self.propagate_wfs(kpt.C_nM, kpt.C_nM, kpt.S_MM, kpt.H0_MM,
                               time_step)
        # ---------------
        # Propagator step
        # ---------------
        # 1. Calculate H(t+dt)
        self.hamiltonian.update()
        for kpt in self.wfs.kpt_u:
            # 2. Estimate H(t+0.5*dt) ~ 0.5 * [ H(t) + H(t+dt) ]
            kpt.H0_MM *= 0.5
            kpt.H0_MM += 0.5 * self.hamiltonian.get_hamiltonian_matrix(kpt)
            # 3. Solve Psi(t+dt) from
            #    (S_MM - 0.5j*H_MM(t+0.5*dt)*dt) Psi(t+dt)
            #       = (S_MM + 0.5j*H_MM(t+0.5*dt)*dt) Psi(t)
            self.propagate_wfs(kpt.C2_nM, kpt.C_nM, kpt.S_MM, kpt.H0_MM,
                               time_step)
            kpt.H0_MM = None
        # 4. Calculate new Hamiltonian (and density)
        self.hamiltonian.update()
        return time + time_step

    def save_wfs(self):
        for kpt in self.wfs.kpt_u:
            kpt.C2_nM[:] = kpt.C_nM

    def todict(self):
        return {'name': 'sicn'}


class TaylorPropagator(Propagator):

    def __init__(self):
        Propagator.__init__(self)
        raise NotImplementedError('TaylorPropagator not implemented')

    def initialize(self, paw):
        if 1:
            # XXX to propagator class
            if self.propagator == 'taylor' and self.blacs:
                # cholS_mm = self.mm_block_descriptor.empty(dtype=complex)
                for kpt in self.wfs.kpt_u:
                    kpt.invS_MM = kpt.S_MM.copy()
                    scalapack_inverse(self.mm_block_descriptor,
                                      kpt.invS_MM, 'L')
            if self.propagator == 'taylor' and not self.blacs:
                tmp = inv(self.wfs.kpt_u[0].S_MM)
                self.wfs.kpt_u[0].invS = tmp

    def taylor_propagator(self, sourceC_nM, targetC_nM, S_MM, H_MM, dt):
        self.timer.start('Taylor propagator')

        if self.blacs:
            # XXX, Preallocate
            target_blockC_nm = self.Cnm_block_descriptor.empty(dtype=complex)
            if self.density.gd.comm.rank != 0:
                # XXX Fake blacks nbands, nao, nbands, nao grid because some
                # weird asserts
                # (these are 0,x or x,0 arrays)
                sourceC_nM = self.CnM_unique_descriptor.zeros(dtype=complex)

            # Zeroth order taylor to target
            self.CnM2nm.redistribute(sourceC_nM, target_blockC_nm)

            # XXX, preallocate, optimize use of temporal arrays
            temp_blockC_nm = target_blockC_nm.copy()
            temp2_blockC_nm = target_blockC_nm.copy()

            order = 4
            assert self.wfs.kd.comm.size == 1
            for n in range(order):
                # Multiply with hamiltonian
                pblas_simple_hemm(self.mm_block_descriptor,
                                  self.Cnm_block_descriptor,
                                  self.Cnm_block_descriptor,
                                  H_MM,
                                  temp_blockC_nm,
                                  temp2_blockC_nm, side='R')
                # XXX: replace with not simple gemm
                temp2_blockC_nm *= -1j * dt / (n + 1)
                # Multiply with inverse overlap
                pblas_simple_hemm(self.mm_block_descriptor,
                                  self.Cnm_block_descriptor,
                                  self.Cnm_block_descriptor,
                                  self.wfs.kpt_u[0].invS_MM,  # XXX
                                  temp2_blockC_nm,
                                  temp_blockC_nm, side='R')
                target_blockC_nm += temp_blockC_nm
            if self.density.gd.comm.rank != 0:  # Todo: Change to gd.rank
                # XXX Fake blacks nbands, nao, nbands, nao grid because
                # some weird asserts
                # (these are 0,x or x,0 arrays)
                target = self.CnM_unique_descriptor.zeros(dtype=complex)
            else:
                target = targetC_nM
            self.Cnm2nM.redistribute(target_blockC_nm, target)

            self.density.gd.comm.broadcast(targetC_nM, 0)
        else:
            assert self.wfs.kd.comm.size == 1
            if self.density.gd.comm.rank == 0:
                targetC_nM[:] = sourceC_nM[:]
                tempC_nM = sourceC_nM.copy()
                order = 4
                for n in range(order):
                    tempC_nM[:] = \
                        np.dot(self.wfs.kpt_u[0].invS,
                               np.dot(H_MM, 1j * dt / (n + 1) *
                                      tempC_nM.T.conjugate())).T.conjugate()
                    targetC_nM += tempC_nM
            self.density.gd.comm.broadcast(targetC_nM, 0)

        self.timer.stop('Taylor propagator')
