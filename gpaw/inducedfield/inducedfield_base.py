import numpy as np
from ase.units import Bohr, Hartree

import gpaw.mpi as mpi
from gpaw.tddft.units import eV_to_aufrequency
from gpaw.poisson import PoissonSolver
from gpaw.fd_operators import Gradient
from gpaw.grid_descriptor import GridDescriptor
from gpaw.utilities.extend_grid import extended_grid_descriptor, \
    extend_array, deextend_array, move_atoms


def sendreceive_dict(comm, a_i, dest, b_i, src_i, iitems):
    """
    Send iitems of dictionary b_i distributed according to src_i
    to dictionary a_i in dest.
    """
    requests = []
    for i in iitems:
        # Send and receive
        if comm.rank == dest:
            # Check if dest has it already
            if dest == src_i[i]:
                a_i[i] = b_i[i].copy()
            else:
                requests.append(comm.receive(a_i[i], src_i[i], tag=i,
                                             block=False))
        elif comm.rank == src_i[i]:
            requests.append(comm.send(b_i[i], dest, tag=i, block=False))
    comm.waitall(requests)


class BaseInducedField(object):
    """Partially virtual base class for induced field calculations.

    Attributes:
    -----------
    omega_w: ndarray
        Frequencies for Fourier transform in atomic units
    folding: string
        Folding type ('Gauss' or 'Lorentz' or None)
    width: float
        Width parameter for folding
    Frho_wg: ndarray (complex)
        Fourier transform of induced charge density
    Fphi_wg: ndarray (complex)
        Fourier transform of induced electric potential
    Fef_wvg: ndarray (complex)
        Fourier transform of induced electric field
    Ffe_wg: ndarray (float)
        Fourier transform of field enhancement
    Fbgef_v: ndarray (float)
        Fourier transform of background electric field
    """

    def __init__(self, filename=None, paw=None,
                 frequencies=None, folding='Gauss', width=0.08,
                 readmode=''):
        """
        Parameters:
        -----------
        filename: string
            Filename of a previous ``InducedField`` to be loaded.
            Setting filename disables parameters ``frequencies``,
            ``folding`` and ``width``.
        paw: PAW object
            PAW object for InducedField
        frequencies: ndarray or list of floats
            Frequencies in eV for Fourier transforms.
            This parameter is neglected if ``filename`` is given.
        folding: string
            Folding type: 'Gauss' or 'Lorentz' or None:
            Gaussian or Lorentzian folding or no folding.
            This parameter is neglected if ``filename`` is given.
        width: float
            Width in eV for the Gaussian (sigma) or Lorentzian (eta) folding
            This parameter is neglected if ``filename`` is given.
        """
        self.dtype = complex

        # These are allocated when calculated
        self.fieldgd = None
        self.Frho_wg = None
        self.Fphi_wg = None
        self.Fef_wvg = None
        self.Ffe_wg = None
        self.Fbgef_v = None

        # has variables
        self.has_paw = False
        self.has_field = False

        if not hasattr(self, 'readwritemode_str_to_list'):
            self.readwritemode_str_to_list = \
                {'': ['Frho', 'atoms'],
                 'all': ['Frho', 'Fphi', 'Fef', 'Ffe', 'atoms'],
                 'field': ['Frho', 'Fphi', 'Fef', 'Ffe', 'atoms']}

        if filename is not None:
            self.initialize(paw, allocate=False)
            self.read(filename, mode=readmode)
            return

        self.folding = folding
        self.width = width * eV_to_aufrequency
        self.set_folding(folding, width * eV_to_aufrequency)

        self.omega_w = np.asarray(frequencies) * eV_to_aufrequency
        self.nw = len(self.omega_w)

        self.nv = 3  # dimensionality of the space

        self.initialize(paw, allocate=True)

    def initialize(self, paw, allocate=True):
        self.allocated = False
        self.has_paw = paw is not None

        if self.has_paw:
            # If no paw is given, then the variables
            # marked with "# !" are created in _read().
            # Other variables are not accessible without
            # paw (TODO: could be implemented...).
            self.paw = paw
            self.world = paw.wfs.world                            # !
            self.domain_comm = paw.wfs.gd.comm                    # !
            self.band_comm = paw.wfs.bd.comm                      # !
            self.kpt_comm = paw.wfs.kd.comm                       # !
            self.rank_a = paw.wfs.atom_partition.rank_a
            self.nspins = paw.density.nspins                      # !
            self.setups = paw.wfs.setups
            self.density = paw.density
            self.atoms = paw.atoms                                # !
            self.na = len(self.atoms.get_atomic_numbers())        # !
            self.gd = self.density.gd                             # !
            self.stencil = self.density.stencil

        if allocate:
            self.allocate()

    def allocate(self):
        self.allocated = True

    def deallocate(self):
        self.fieldgd = None
        self.Frho_wg = None
        self.Fphi_wg = None
        self.Fef_wvg = None
        self.Ffe_wg = None
        self.Fbgef_v = None

    def set_folding(self, folding, width):
        """
        width: float
            Width in atomic units
        """
        if width is None:
            folding = None

        self.folding = folding
        if self.folding is None:
            self.width = None
        else:
            self.width = width

    def get_induced_density(self, from_density, gridrefinement):
        raise RuntimeError('Virtual member function called')

    def calculate_induced_field(self, from_density='comp',
                                gridrefinement=2,
                                extend_N_cd=None,
                                deextend=False,
                                poissonsolver=None,
                                gradient_n=3):
        if self.has_field and \
           from_density == self.field_from_density and \
           self.Frho_wg is not None:
            Frho_wg = self.Frho_wg
            gd = self.fieldgd
        else:
            Frho_wg, gd = self.get_induced_density(from_density,
                                                   gridrefinement)

        # Always extend a bit to get field without jumps
        if extend_N_cd is None:
            extend_N = max(8, 2**int(np.ceil(np.log(gradient_n) / np.log(2.))))
            extend_N_cd = extend_N * np.ones(shape=(3, 2), dtype=np.int)
            deextend = True

        # Extend grid
        oldgd = gd
        egd, cell_cv, move_c = \
            extended_grid_descriptor(gd, extend_N_cd=extend_N_cd)
        Frho_we = egd.zeros((self.nw,), dtype=self.dtype)
        for w in range(self.nw):
            extend_array(gd, egd, Frho_wg[w], Frho_we[w])
        Frho_wg = Frho_we
        gd = egd
        if not deextend:
            # TODO: this will make atoms unusable with original grid
            self.atoms.set_cell(cell_cv, scale_atoms=False)
            move_atoms(self.atoms, move_c)

        # Allocate arrays
        Fphi_wg = gd.zeros((self.nw,), dtype=self.dtype)
        Fef_wvg = gd.zeros((self.nw, self.nv,), dtype=self.dtype)
        Ffe_wg = gd.zeros((self.nw,), dtype=float)

        # Poissonsolver
        if poissonsolver is None:
            poissonsolver = PoissonSolver(name='fd', eps=1e-20)
        poissonsolver.set_grid_descriptor(gd)

        for w in range(self.nw):
            # TODO: better output of progress
            # parprint('%d' % w)
            calculate_field(gd, Frho_wg[w], self.Fbgef_v,
                            Fphi_wg[w], Fef_wvg[w], Ffe_wg[w],
                            poissonsolver=poissonsolver,
                            nv=self.nv,
                            gradient_n=gradient_n)

        # De-extend grid
        if deextend:
            Frho_wo = oldgd.zeros((self.nw,), dtype=self.dtype)
            Fphi_wo = oldgd.zeros((self.nw,), dtype=self.dtype)
            Fef_wvo = oldgd.zeros((self.nw, self.nv,), dtype=self.dtype)
            Ffe_wo = oldgd.zeros((self.nw,), dtype=float)
            for w in range(self.nw):
                deextend_array(oldgd, gd, Frho_wo[w], Frho_wg[w])
                deextend_array(oldgd, gd, Fphi_wo[w], Fphi_wg[w])
                deextend_array(oldgd, gd, Ffe_wo[w], Ffe_wg[w])
                for v in range(self.nv):
                    deextend_array(oldgd, gd, Fef_wvo[w][v], Fef_wvg[w][v])
            Frho_wg = Frho_wo
            Fphi_wg = Fphi_wo
            Fef_wvg = Fef_wvo
            Ffe_wg = Ffe_wo
            gd = oldgd

        # Store results
        self.has_field = True
        self.field_from_density = from_density
        self.fieldgd = gd
        self.Frho_wg = Frho_wg
        self.Fphi_wg = Fphi_wg
        self.Fef_wvg = Fef_wvg
        self.Ffe_wg = Ffe_wg

    def _parse_readwritemode(self, mode):
        if isinstance(mode, str):
            try:
                readwrites = self.readwritemode_str_to_list[mode]
            except KeyError:
                raise IOError('unknown readwrite mode string')
        elif isinstance(mode, list):
            readwrites = mode
        else:
            raise IOError('unknown readwrite mode type')

        if any(k in readwrites for k in ['Frho', 'Fphi', 'Fef', 'Ffe']):
            readwrites.append('field')

        return readwrites

    def read(self, filename, mode='', idiotproof=True):
        if idiotproof and not filename.endswith('.ind'):
            raise IOError('Filename must end with `.ind`.')

        reads = self._parse_readwritemode(mode)

        # Open reader
        from gpaw.io import Reader
        reader = Reader(filename)
        self._read(reader, reads)
        reader.close()
        self.world.barrier()

    def _read(self, reader, reads):
        r = reader

        # Test data type
        dtype = {'float': float, 'complex': complex}[r.dtype]
        if dtype != self.dtype:
            raise IOError('Data is an incompatible type.')

        # Read dimensions
        na = r.na
        self.nv = r.nv
        nspins = r.nspins
        ng = r.ng

        # Background electric field
        Fbgef_v = r.Fbgef_v

        if self.has_paw:
            # Test dimensions
            if na != self.na:
                raise IOError('natoms is incompatible with calculator')
            if nspins != self.nspins:
                raise IOError('nspins is incompatible with calculator')
            if (ng != self.gd.get_size_of_global_array()).any():
                raise IOError('grid is incompatible with calculator')
            if (Fbgef_v != self.Fbgef_v).any():
                raise IOError('kick is incompatible with calculator')
        else:
            # Construct objects / assign values without paw
            self.na = na
            self.nspins = nspins
            self.Fbgef_v = Fbgef_v

            from ase.io.trajectory import read_atoms
            self.atoms = read_atoms(r.atoms)

            self.world = mpi.world
            self.gd = GridDescriptor(ng + 1, self.atoms.get_cell() / Bohr,
                                     pbc_c=False, comm=self.world)
            self.domain_comm = self.gd.comm
            self.band_comm = mpi.SerialCommunicator()
            self.kpt_comm = mpi.SerialCommunicator()

        # Folding
        folding = r.folding
        width = r.width
        self.set_folding(folding, width)

        # Frequencies
        self.omega_w = r.omega_w
        self.nw = len(self.omega_w)

        # Read field
        if 'field' in reads:
            nfieldg = r.nfieldg
            self.has_field = True
            self.field_from_density = r.field_from_density
            self.fieldgd = self.gd.new_descriptor(N_c=nfieldg + 1)

            def readarray(name, shape, dtype):
                if name.split('_')[0] in reads:
                    setattr(self, name, self.fieldgd.empty(shape, dtype=dtype))
                    self.fieldgd.distribute(r.get(name), getattr(self, name))

            readarray('Frho_wg', (self.nw,), self.dtype)
            readarray('Fphi_wg', (self.nw,), self.dtype)
            readarray('Fef_wvg', (self.nw, self.nv), self.dtype)
            readarray('Ffe_wg', (self.nw,), float)

    def write(self, filename, mode='', idiotproof=True):
        """
        Parameters
        ----------
        mode: string or list of strings


        """
        if idiotproof and not filename.endswith('.ind'):
            raise IOError('Filename must end with `.ind`.')

        writes = self._parse_readwritemode(mode)

        if 'field' in writes and self.fieldgd is None:
            raise IOError('field variables cannot be written ' +
                          'before they are calculated')

        from gpaw.io import Writer
        writer = Writer(filename, self.world, tag='INDUCEDFIELD')
        # Actual write
        self._write(writer, writes)
        # Make sure slaves don't return before master is done
        writer.close()
        self.world.barrier()

    def _write(self, writer, writes):
        # Write parameters/dimensions
        writer.write(dtype={float: 'float', complex: 'complex'}[self.dtype],
                     folding=self.folding,
                     width=self.width,
                     na=self.na,
                     nv=self.nv,
                     nspins=self.nspins,
                     ng=self.gd.get_size_of_global_array())

        # Write field grid
        if 'field' in writes:
            writer.write(field_from_density=self.field_from_density,
                         nfieldg=self.fieldgd.get_size_of_global_array())

        # Write frequencies
        writer.write(omega_w=self.omega_w)

        # Write background electric field
        writer.write(Fbgef_v=self.Fbgef_v)

        from ase.io.trajectory import write_atoms
        write_atoms(writer.child('atoms'), self.atoms)

        if 'field' in writes:
            def writearray(name, shape, dtype):
                if name.split('_')[0] in writes:
                    writer.add_array(name, shape, dtype)
                a_wxg = getattr(self, name)
                for w in range(self.nw):
                    writer.fill(self.fieldgd.collect(a_wxg[w]))

            shape = (self.nw,) + tuple(self.fieldgd.get_size_of_global_array())
            writearray('Frho_wg', shape, self.dtype)
            writearray('Fphi_wg', shape, self.dtype)
            writearray('Ffe_wg', shape, float)

            shape3 = shape[:1] + (self.nv,) + shape[1:]
            writearray('Fef_wvg', shape3, self.dtype)


def calculate_field(gd, rho_g, bgef_v,
                    phi_g, ef_vg, fe_g,  # preallocated numpy arrays
                    poissonsolver,
                    nv=3,
                    gradient_n=3):

    dtype = rho_g.dtype
    yes_complex = dtype == complex

    phi_g[:] = 0.0
    ef_vg[:] = 0.0
    fe_g[:] = 0.0
    tmp_g = gd.zeros(dtype=float)

    # Potential, real part
    poissonsolver.solve(tmp_g, rho_g.real.copy())
    phi_g += tmp_g
    # Potential, imag part
    if yes_complex:
        tmp_g[:] = 0.0
        poissonsolver.solve(tmp_g, rho_g.imag.copy())
        phi_g += 1.0j * tmp_g

    # Gradient
    gradient = [Gradient(gd, v, scale=1.0, n=gradient_n)
                for v in range(nv)]
    for v in range(nv):
        # Electric field, real part
        gradient[v].apply(-phi_g.real, tmp_g)
        ef_vg[v] += tmp_g
        # Electric field, imag part
        if yes_complex:
            gradient[v].apply(-phi_g.imag, tmp_g)
            ef_vg[v] += 1.0j * tmp_g

    # Electric field enhancement
    tmp_g[:] = 0.0  # total electric field norm
    bgefnorm = 0.0  # background electric field norm
    for v in range(nv):
        tmp_g += np.absolute(bgef_v[v] + ef_vg[v])**2
        bgefnorm += np.absolute(bgef_v[v])**2

    tmp_g = np.sqrt(tmp_g)
    bgefnorm = np.sqrt(bgefnorm)

    fe_g[:] = tmp_g / bgefnorm


def zero_pad(a):
    """Add zeros to both sides of all axis on array a.

    Not parallel safe.
    """

    z_shape = np.zeros(shape=len(a.shape))
    z_shape[-3:] = 2
    b_shape = np.array(a.shape) + z_shape

    b = np.zeros(shape=b_shape, dtype=a.dtype)

    b[..., 1:-1, 1:-1, 1:-1] = a
    return b


# TOOD: remove/edit this function
def calculate_oscstr(Fn_wg, omega_w, box, kick):
    omega_w = np.array(omega_w)
    omega_w *= eV_to_aufrequency  # to a.u.
    box = box.copy() / Bohr  # to a.u
    volume = box.prod()
    ng = np.array(Fn_wg[0].shape)
    dV = volume / (ng - 1).prod()  # Note: data is zeropadded to both sides

    Fn_wg_im = Fn_wg.imag
    if kick[0] != 0.0:
        osc_w = -2 * omega_w / np.pi / kick[0] * \
            ((Fn_wg_im.sum(axis=2)).sum(axis=2) *
             np.linspace(0, box[0], ng[0])).sum(axis=1) * dV
    elif kick[1] != 0.0:
        osc_w = -2 * omega_w / np.pi / kick[1] * \
            ((Fn_wg_im.sum(axis=1)).sum(axis=2) *
             np.linspace(0, box[1], ng[1])).sum(axis=1) * dV
    elif kick[2] != 0.0:
        osc_w = -2 * omega_w / np.pi / kick[2] * \
            ((Fn_wg_im.sum(axis=1)).sum(axis=1) *
             np.linspace(0, box[2], ng[2])).sum(axis=1) * dV
    return osc_w / Hartree  # to 1/eV


# TOOD: remove/edit this function
def calculate_polarizability(Fn_wg, box, kick):
    box = box.copy() / Bohr  # to a.u
    volume = box.prod()
    ng = np.array(Fn_wg[0].shape)
    dV = volume / (ng - 1).prod()  # Note: data is zeropadded to both sides

    if kick[0] != 0.0:
        pol_w = -1.0 / kick[0] * \
            ((Fn_wg.sum(axis=2)).sum(axis=2) *
             np.linspace(0, box[0], ng[0])).sum(axis=1) * dV
    elif kick[1] != 0.0:
        pol_w = -1.0 / kick[1] * \
            ((Fn_wg.sum(axis=1)).sum(axis=2) *
             np.linspace(0, box[1], ng[1])).sum(axis=1) * dV
    elif kick[2] != 0.0:
        pol_w = -1.0 / kick[2] * \
            ((Fn_wg.sum(axis=1)).sum(axis=1) *
             np.linspace(0, box[2], ng[2])).sum(axis=1) * dV
    return pol_w / Hartree**2  # to 1/eV**2
