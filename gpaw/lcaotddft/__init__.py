import numpy as np

from ase.units import Bohr, Hartree

from gpaw import GPAW
from gpaw.external import ConstantElectricField
from gpaw.lcaotddft.hamiltonian import KickHamiltonian
from gpaw.lcaotddft.hamiltonian import TimeDependentHamiltonian
from gpaw.lcaotddft.logger import TDDFTLogger
from gpaw.lcaotddft.propagators import create_propagator
from gpaw.tddft.units import attosec_to_autime


class LCAOTDDFT(GPAW):
    def __init__(self, filename=None,
                 propagator=None, fxc=None, **kwargs):
        self.time = 0.0
        self.niter = 0
        self.kick_strength = np.zeros(3)
        self.tddft_initialized = False
        self.action = None
        self.td_hamiltonian = TimeDependentHamiltonian(fxc=fxc)

        self.propagator = propagator
        if filename is None:
            kwargs['mode'] = kwargs.get('mode', 'lcao')
        GPAW.__init__(self, filename, **kwargs)

        # Restarting from a file
        if filename is not None:
            # self.initialize()
            self.set_positions()

    def _write(self, writer, mode):
        GPAW._write(self, writer, mode)
        if self.tddft_initialized:
            w = writer.child('tddft')
            w.write(time=self.time,
                    niter=self.niter,
                    kick_strength=self.kick_strength,
                    propagator=self.propagator.todict())
            self.td_hamiltonian.write(w.child('td_hamiltonian'))

    def read(self, filename):
        reader = GPAW.read(self, filename)
        if 'tddft' in reader:
            r = reader.tddft
            self.time = r.time
            self.niter = r.niter
            self.kick_strength = r.kick_strength
            if self.propagator is None:
                self.propagator = r.propagator
            else:
                self.log('Note! Propagator changed!')
            self.td_hamiltonian.wfs = self.wfs
            self.td_hamiltonian.read(r.td_hamiltonian)

    def tddft_init(self):
        if self.tddft_initialized:
            return

        self.log('-----------------------------------')
        self.log('Initializing time-propagation TDDFT')
        self.log('-----------------------------------')
        self.log()

        assert self.wfs.dtype == complex

        self.timer.start('Initialize TDDFT')

        # Initialize Hamiltonian
        self.td_hamiltonian.initialize(self)

        # Initialize propagator
        self.propagator = create_propagator(self.propagator)
        self.propagator.initialize(self)

        self.log('Propagator:')
        self.log(self.propagator.get_description())
        self.log()

        # Add logger
        TDDFTLogger(self)

        # Call observers before propagation
        self.action = 'init'
        self.call_observers(self.niter)

        self.tddft_initialized = True
        self.timer.stop('Initialize TDDFT')

    def absorption_kick(self, kick_strength):
        self.tddft_init()

        self.timer.start('Kick')

        self.kick_strength = np.array(kick_strength, dtype=float)
        magnitude = np.sqrt(np.sum(self.kick_strength**2))
        direction = self.kick_strength / magnitude

        self.log('----  Applying absorption kick')
        self.log('----  Magnitude: %.8f Hartree/Bohr' % magnitude)
        self.log('----  Direction: %.4f %.4f %.4f' % tuple(direction))

        # Create hamiltonian object for absorption kick
        cef = ConstantElectricField(magnitude * Hartree / Bohr, direction)
        kick_hamiltonian = KickHamiltonian(self, cef)

        # Propagate kick
        self.propagator.kick(kick_hamiltonian, self.time)

        # Call observers after kick
        self.action = 'kick'
        self.call_observers(self.niter)

        self.niter += 1

        self.timer.stop('Kick')

    def propagate(self, time_step=10, iterations=2000):
        self.tddft_init()

        time_step *= attosec_to_autime
        self.maxiter = self.niter + iterations

        self.log('----  About to do %d propagation steps' % iterations)

        self.timer.start('Propagate')
        while self.niter < self.maxiter:
            # Propagate one step
            self.time = self.propagator.propagate(self.time, time_step)

            # Call registered callback functions
            self.action = 'propagate'
            self.call_observers(self.niter)

            self.niter += 1
        self.timer.stop('Propagate')

    def replay(self, **kwargs):
        self.propagator = create_propagator(**kwargs)
        self.tddft_init()
        self.propagator.control_paw(self)
