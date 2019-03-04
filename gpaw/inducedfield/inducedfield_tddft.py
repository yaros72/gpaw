import numpy as np

from ase.parallel import parprint

from gpaw import debug
from gpaw.analyse.observers import Observer
from gpaw.transformers import Transformer
from gpaw.lfc import BasisFunctions
from gpaw.utilities import unpack2, is_contiguous

from gpaw.inducedfield.inducedfield_base import BaseInducedField, \
    sendreceive_dict


class TDDFTInducedField(BaseInducedField, Observer):
    """Induced field class for time propagation TDDFT.
    
    Attributes (see also ``BaseInducedField``):
    -------------------------------------------
    time: float
        Current time
    Fnt_wsG: ndarray (complex)
        Fourier transform of induced pseudo density
    n0t_sG: ndarray (float)
        Ground state pseudo density
    FD_awsp: dict of ndarray (complex)
        Fourier transform of induced D_asp
    D0_asp: dict of ndarray (float)
        Ground state D_asp
    """
    
    def __init__(self, filename=None, paw=None,
                 frequencies=None, folding='Gauss', width=0.08,
                 interval=1, restart_file=None
                 ):
        """
        Parameters (see also ``BaseInducedField``):
        -------------------------------------------
        paw: TDDFT object
            TDDFT object for time propagation
        width: float
            Width in eV for the Gaussian (sigma) or Lorentzian (eta) folding
            Gaussian   = exp(- (1/2) * sigma^2 * t^2)
            Lorentzian = exp(- eta * t)
        interval: int
            Number of timesteps between calls (used when attaching)
        restart_file: string
            Name of the restart file
        """

        Observer.__init__(self, interval)
        # From observer:
        # self.niter
        # self.interval
        
        # Restart file
        self.restart_file = restart_file
        
        # These are allocated in allocate()
        self.Fnt_wsG = None
        self.n0t_sG = None
        self.FD_awsp = None
        self.D0_asp = None

        self.readwritemode_str_to_list = \
            {'': ['Fnt', 'n0t', 'FD', 'D0', 'atoms'],
             'all': ['Fnt', 'n0t', 'FD', 'D0',
                     'Frho', 'Fphi', 'Fef', 'Ffe', 'atoms'],
             'field': ['Frho', 'Fphi', 'Fef', 'Ffe', 'atoms']}

        BaseInducedField.__init__(self, filename, paw,
                                  frequencies, folding, width)
        
    def initialize(self, paw, allocate=True):
        BaseInducedField.initialize(self, paw, allocate)
        
        if self.has_paw:
            assert hasattr(paw, 'time') and hasattr(paw, 'niter'), 'Use TDDFT!'
            self.time = paw.time                # !
            self.niter = paw.niter
            
            # TODO: remove this requirement
            assert np.count_nonzero(paw.kick_strength) > 0, \
                'Apply absorption kick before %s' % self.__class__.__name__
        
            # Background electric field
            self.Fbgef_v = paw.kick_strength

            # Attach to PAW-type object
            paw.attach(self, self.interval)
            # TODO: write more details (folding, freqs, etc)
            parprint('%s: Attached ' % self.__class__.__name__)

    def set_folding(self, folding, width):
        BaseInducedField.set_folding(self, folding, width)
        
        if self.folding is None:
            self.envelope = lambda t: 1.0
        else:
            if self.folding == 'Gauss':
                self.envelope = lambda t: np.exp(- 0.5 * self.width**2 * t**2)
            elif self.folding == 'Lorentz':
                self.envelope = lambda t: np.exp(- self.width * t)
            else:
                raise RuntimeError('unknown folding "' + self.folding + '"')
        
    def allocate(self):
        if not self.allocated:
            
            # Ground state pseudo density
            self.n0t_sG = self.density.nt_sG.copy()
            
            # Fourier transformed pseudo density
            self.Fnt_wsG = self.gd.zeros((self.nw, self.nspins),
                                         dtype=self.dtype)
    
            # Ground state D_asp
            self.D0_asp = {}
            for a, D_sp in self.density.D_asp.items():
                self.D0_asp[a] = D_sp.copy()
            
            # Size of D_p for each atom
            self.np_a = {}
            for a, D_sp in self.D0_asp.items():
                self.np_a[a] = np.array([len(D_sp[0])])
            
            # Fourier transformed D_asp
            self.FD_awsp = {}
            for a, np_ in self.np_a.items():
                self.FD_awsp[a] = np.zeros((self.nw, self.nspins, np_[0]),
                                           dtype=self.dtype)
            
            self.allocated = True
            
        if debug:
            assert is_contiguous(self.Fnt_wsG, self.dtype)

    def deallocate(self):
        BaseInducedField.deallocate(self)
        self.n0t_sG = None
        self.Fnt_wsG = None
        self.D0_asp = None
        self.FD_awsp = None
        
    def update(self):
        # Update time
        self.time = self.paw.time
        time_step = self.paw.time_step

        # Complex exponential with envelope
        f_w = np.exp(1.0j * self.omega_w * self.time) * \
            self.envelope(self.time) * time_step

        # Time-dependent quantities
        nt_sG = self.density.nt_sG
        D_asp = self.density.D_asp

        # Update Fourier transforms
        for w in range(self.nw):
            self.Fnt_wsG[w] += (nt_sG - self.n0t_sG) * f_w[w]
            for a, D_sp in D_asp.items():
                self.FD_awsp[a][w] += (D_sp - self.D0_asp[a]) * f_w[w]

        # Restart file
        if self.restart_file is not None and \
           self.niter % self.paw.dump_interval == 0:
            self.write(self.restart_file)
            parprint('%s: Wrote restart file' % self.__class__.__name__)
    
    def interpolate_pseudo_density(self, gridrefinement=2):
        
        gd = self.gd
        Fnt_wsg = self.Fnt_wsG.copy()
        
        # Find m for
        # gridrefinement = 2**m
        m1 = np.log(gridrefinement) / np.log(2.)
        m = int(np.round(m1))
        
        # Check if m is really integer
        if np.absolute(m - m1) < 1e-8:
            for i in range(m):
                gd2 = gd.refine()
                
                # Interpolate
                interpolator = Transformer(gd, gd2, self.stencil,
                                           dtype=self.dtype)
                Fnt2_wsg = gd2.empty((self.nw, self.nspins), dtype=self.dtype)
                for w in range(self.nw):
                    for s in range(self.nspins):
                        interpolator.apply(Fnt_wsg[w][s], Fnt2_wsg[w][s],
                                           np.ones((3, 2), dtype=complex))

                gd = gd2
                Fnt_wsg = Fnt2_wsg
        else:
            raise NotImplementedError
        
        return Fnt_wsg, gd
    
    def comp_charge_correction(self, gridrefinement=2):
    
        # TODO: implement for gr==1 also
        assert gridrefinement == 2
        
        # Density
        Fnt_wsg, gd = self.interpolate_pseudo_density(gridrefinement)
        Frhot_wg = Fnt_wsg.sum(axis=1)
        
        tmp_g = gd.empty(dtype=float)
        for w in range(self.nw):
            # Determine compensation charge coefficients:
            FQ_aL = {}
            for a, FD_wsp in self.FD_awsp.items():
                FQ_aL[a] = np.dot(FD_wsp[w].sum(axis=0),
                                  self.setups[a].Delta_pL)

            # Add real part of compensation charges
            tmp_g[:] = 0
            FQ2_aL = {}
            for a, FQ_L in FQ_aL.items():
                # Take copy to make array contiguous
                FQ2_aL[a] = FQ_L.real.copy()
#                print is_contiguous(FQ2_aL[a])
#                print is_contiguous(FQ_L.real)
            self.density.ghat.add(tmp_g, FQ2_aL)
            Frhot_wg[w] += tmp_g
            
            # Add imag part of compensation charges
            tmp_g[:] = 0
            FQ2_aL = {}
            for a, FQ_L in FQ_aL.items():
                FQ2_aL[a] = FQ_L.imag.copy()
            self.density.ghat.add(tmp_g, FQ2_aL)
            Frhot_wg[w] += 1.0j * tmp_g

        return Frhot_wg, gd
    
    def paw_corrections(self, gridrefinement=2):
        
        Fn_wsg, gd = self.interpolate_pseudo_density(gridrefinement)
        
        # Splines
        splines = {}
        phi_aj = []
        phit_aj = []
        for a, id in enumerate(self.setups.id_a):
            if id in splines:
                phi_j, phit_j = splines[id]
            else:
                # Load splines:
                phi_j, phit_j = self.setups[a].get_partial_waves()[:2]
                splines[id] = (phi_j, phit_j)
            phi_aj.append(phi_j)
            phit_aj.append(phit_j)

        # Create localized functions from splines
        phi = BasisFunctions(gd, phi_aj, dtype=float)
        phit = BasisFunctions(gd, phit_aj, dtype=float)
#        phi = BasisFunctions(gd, phi_aj, dtype=complex)
#        phit = BasisFunctions(gd, phit_aj, dtype=complex)
        spos_ac = self.atoms.get_scaled_positions()
        phi.set_positions(spos_ac)
        phit.set_positions(spos_ac)
        
        tmp_g = gd.empty(dtype=float)
        rho_MM = np.zeros((phi.Mmax, phi.Mmax), dtype=self.dtype)
        rho2_MM = np.zeros_like(rho_MM)
        for w in range(self.nw):
            for s in range(self.nspins):
                rho_MM[:] = 0
                M1 = 0
                for a, setup in enumerate(self.setups):
                    ni = setup.ni
                    FD_wsp = self.FD_awsp.get(a)
                    if FD_wsp is None:
                        FD_p = np.empty((ni * (ni + 1) // 2), dtype=self.dtype)
                    else:
                        FD_p = FD_wsp[w][s]
                    if gd.comm.size > 1:
                        gd.comm.broadcast(FD_p, self.rank_a[a])
                    D_ij = unpack2(FD_p)
                    # unpack does complex conjugation that we don't want so
                    # remove conjugation
                    D_ij = np.triu(D_ij, 1) + np.conj(np.tril(D_ij))
                    
#                    if FD_wsp is None:
#                        FD_wsp = np.empty((self.nw, self.nspins,
#                                           ni * (ni + 1) // 2),
#                                          dtype=self.dtype)
#                    if gd.comm.size > 1:
#                        gd.comm.broadcast(FD_wsp, self.rank_a[a])
#                    D_ij = unpack2(FD_wsp[w][s])
#                    D_ij = np.triu(D_ij, 1) + np.conj(np.tril(D_ij))
                    
                    M2 = M1 + ni
                    rho_MM[M1:M2, M1:M2] = D_ij
                    M1 = M2
     
                # Add real part of AE corrections
                tmp_g[:] = 0
                rho2_MM[:] = rho_MM.real
                # TODO: use ae_valence_density_correction
                phi.construct_density(rho2_MM, tmp_g, q=-1)
                phit.construct_density(-rho2_MM, tmp_g, q=-1)
#                phi.lfc.ae_valence_density_correction(rho2_MM, tmp_g,
#                                                      np.zeros(len(phi.M_W),
#                                                               np.intc),
#                                                      np.zeros(self.na))
#                phit.lfc.ae_valence_density_correction(-rho2_MM, tmp_g,
#                                                      np.zeros(len(phi.M_W),
#                                                               np.intc),
#                                                      np.zeros(self.na))
                Fn_wsg[w][s] += tmp_g
                
                # Add imag part of AE corrections
                tmp_g[:] = 0
                rho2_MM[:] = rho_MM.imag
                # TODO: use ae_valence_density_correction
                phi.construct_density(rho2_MM, tmp_g, q=-1)
                phit.construct_density(-rho2_MM, tmp_g, q=-1)
#                phi.lfc.ae_valence_density_correction(rho2_MM, tmp_g,
#                                                      np.zeros(len(phi.M_W),
#                                                               np.intc),
#                                                      np.zeros(self.na))
#                phit.lfc.ae_valence_density_correction(-rho2_MM, tmp_g,
#                                                      np.zeros(len(phi.M_W),
#                                                               np.intc),
#                                                      np.zeros(self.na))
                Fn_wsg[w][s] += 1.0j * tmp_g
        
        return Fn_wsg, gd
        
    def get_induced_density(self, from_density, gridrefinement):
        # Return charge density (electrons = negative charge)
        if from_density == 'pseudo':
            Fn_wsg, gd = self.interpolate_pseudo_density(gridrefinement)
            Frho_wg = - Fn_wsg.sum(axis=1)
            return Frho_wg, gd
        elif from_density == 'comp':
            Frho_wg, gd = self.comp_charge_correction(gridrefinement)
            Frho_wg = - Frho_wg
            return Frho_wg, gd
        elif from_density == 'ae':
            Fn_wsg, gd = self.paw_corrections(gridrefinement)
            Frho_wg = - Fn_wsg.sum(axis=1)
            return Frho_wg, gd
        else:
            raise RuntimeError('unknown from_density "' + from_density + '"')
    
    def _read(self, reader, reads):
        BaseInducedField._read(self, reader, reads)

        r = reader
        time = r.time
        if self.has_paw:
            # Test time
            if abs(time - self.time) >= 1e-9:
                raise IOError('Timestamp is incompatible with calculator.')
        else:
            self.time = time

        # Allocate
        self.allocate()

        # Dimensions for D_p for all atoms
        self.np_a = r.np_a

        def readarray(name):
            if name.split('_')[0] in reads:
                self.gd.distribute(r.get(name), getattr(self, name))

        # Read arrays
        readarray('n0t_sG')
        readarray('Fnt_wsG')
        
        if 'D0' in reads:
            D0_asp = r.D0_asp
            self.D0_asp = {}
            for a in range(self.na):
                if self.domain_comm.rank == self.rank_a[a]:
                    self.D0_asp[a] = D0_asp[a]
        
        if 'FD' in reads:
            FD_awsp = r.FD_awsp
            self.FD_awsp = {}
            for a in range(self.na):
                if self.domain_comm.rank == self.rank_a[a]:
                    self.FD_awsp[a] = FD_awsp[a]

    def _write(self, writer, writes):
        BaseInducedField._write(self, writer, writes)

        # Collect np_a to master
        if self.kpt_comm.rank == 0 and self.band_comm.rank == 0:
            
            # Create empty dict on domain master
            if self.domain_comm.rank == 0:
                np_a = {}
                for a in range(self.na):
                    np_a[a] = np.empty(1, dtype=int)
            else:
                np_a = {}
            # Collect dict to master
            sendreceive_dict(self.domain_comm, np_a, 0,
                             self.np_a, self.rank_a, range(self.na))
        
        # Write time propagation status
        writer.write(time=self.time, np_a=np_a)

        def writearray(name, shape, dtype):
            if name.split('_')[0] in writes:
                writer.add_array(name, shape, dtype)
            a_wxg = getattr(self, name)
            for w in range(self.nw):
                writer.fill(self.gd.collect(a_wxg[w]))

        ng = tuple(self.gd.get_size_of_global_array())

        # Write time propagation arrays
        if 'n0t' in writes:
            writer.write(n0t_sG=self.gd.collect(self.n0t_sG))
        writearray('Fnt_wsG', (self.nw, self.nspins) + ng, self.dtype)
        
        if 'D0' in writes:
            # Collect D0_asp to world master
            if self.kpt_comm.rank == 0 and self.band_comm.rank == 0:
                # Create empty dict on domain master
                if self.domain_comm.rank == 0:
                    D0_asp = {}
                    for a in range(self.na):
                        npa = np_a[a]
                        D0_asp[a] = np.empty((self.nspins, npa[0]),
                                             dtype=float)
                else:
                    D0_asp = {}
                # Collect dict to master
                sendreceive_dict(self.domain_comm, D0_asp, 0,
                                 self.D0_asp, self.rank_a, range(self.na))
            # Write
            writer.write(D0_asp=D0_asp)

        if 'FD' in writes:
            # Collect FD_awsp to world master
            if self.kpt_comm.rank == 0 and self.band_comm.rank == 0:
                # Create empty dict on domain master
                if self.domain_comm.rank == 0:
                    FD_awsp = {}
                    for a in range(self.na):
                        npa = np_a[a]
                        FD_awsp[a] = np.empty((self.nw, self.nspins, npa[0]),
                                              dtype=complex)
                else:
                    FD_awsp = {}
                # Collect dict to master
                sendreceive_dict(self.domain_comm, FD_awsp, 0,
                                 self.FD_awsp, self.rank_a, range(self.na))
            # Write
            writer.write(FD_awsp=FD_awsp)
