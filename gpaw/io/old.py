"""Code for reading old gpw files."""
import warnings

import numpy as np
import ase.io.ulm as ulm
from ase import Atoms
from ase.dft.kpoints import monkhorst_pack
from ase.io.trajectory import write_atoms
from ase.units import AUT, Bohr, Ha
from ase.utils import devnull

from gpaw.wavefunctions.pw import PW
from gpaw.occupations import FermiDirac
from gpaw.io.tar import Reader


def wrap_old_gpw_reader(filename):
    warnings.warn('You are reading an old-style gpw-file.  Please check '
                  'the results carefully!')

    r = Reader(filename)

    data = {'version': -1,
            'gpaw_version': '1.0',
            'ha': Ha,
            'bohr': Bohr,
            'scf.': {'converged': True},
            'atoms.': {},
            'wave_functions.': {}}

    class DictBackend:
        def write(self, **kwargs):
            data['atoms.'].update(kwargs)

    write_atoms(DictBackend(), read_atoms(r))

    e_total_extrapolated = r.get('PotentialEnergy').item() * Ha
    magmom_a = r.get('MagneticMoments')
    data['results.'] = {
        'energy': e_total_extrapolated,
        'magmoms': magmom_a,
        'magmom': magmom_a.sum()}

    if r.has_array('CartesianForces'):
        data['results.']['forces'] = r.get('CartesianForces') * Ha / Bohr

    p = data['parameters.'] = {}

    p['xc'] = r['XCFunctional']
    p['nbands'] = r.dimension('nbands')
    p['spinpol'] = (r.dimension('nspins') == 2)

    bzk_kc = r.get('BZKPoints', broadcast=True)
    if r.has_array('NBZKPoints'):
        p['kpts'] = r.get('NBZKPoints', broadcast=True)
        if r.has_array('MonkhorstPackOffset'):
            offset_c = r.get('MonkhorstPackOffset', broadcast=True)
            if offset_c.any():
                p['kpts'] = monkhorst_pack(p['kpts']) + offset_c
    else:
        p['kpts'] = bzk_kc

    if r['version'] < 4:
        usesymm = r['UseSymmetry']
        if usesymm is None:
            p['symmetry'] = {'time_reversal': False, 'point_group': False}
        elif usesymm:
            p['symmetry'] = {'time_reversal': True, 'point_group': True}
        else:
            p['symmetry'] = {'time_reversal': True, 'point_group': False}
    else:
        p['symmetry'] = {'point_group': r['SymmetryOnSwitch'],
                         'symmorphic': r['SymmetrySymmorphicSwitch'],
                         'time_reversal': r['SymmetryTimeReversalSwitch'],
                         'tolerance': r['SymmetryToleranceCriterion']}

    p['basis'] = r['BasisSet']

    try:
        h = r['GridSpacing']
    except KeyError:  # CMR can't handle None!
        h = None
    if h is not None:
        p['h'] = Bohr * h
    if r.has_array('GridPoints'):
        p['gpts'] = r.get('GridPoints')

    p['lmax'] = r['MaximumAngularMomentum']
    p['setups'] = r['SetupTypes']
    p['fixdensity'] = r['FixDensity']
    nbtc = r['NumberOfBandsToConverge']
    if not isinstance(nbtc, (int, str)):
        # The string 'all' was eval'ed to the all() function!
        nbtc = 'all'
    p['convergence'] = {'density': r['DensityConvergenceCriterion'],
                        'energy': r['EnergyConvergenceCriterion'] * Ha,
                        'eigenstates': r['EigenstatesConvergenceCriterion'],
                        'bands': nbtc}
    mixer = r['MixClass']
    weight = r['MixWeight']

    for key in ['basis', 'setups']:
        dct = p[key]
        if isinstance(dct, dict) and None in dct:
            dct['default'] = dct.pop(None)

    if mixer == 'Mixer':
        from gpaw.mixer import Mixer
    elif mixer == 'MixerSum':
        from gpaw.mixer import MixerSum as Mixer
    elif mixer == 'MixerSum2':
        from gpaw.mixer import MixerSum2 as Mixer
    elif mixer == 'MixerDif':
        from gpaw.mixer import MixerDif as Mixer
    elif mixer == 'DummyMixer':
        from gpaw.mixer import DummyMixer as Mixer
    else:
        Mixer = None

    if Mixer is None:
        p['mixer'] = None
    else:
        p['mixer'] = Mixer(r['MixBeta'], r['MixOld'], weight)

    p['stencils'] = (r['KohnShamStencil'],
                     r['InterpolationStencil'])

    vt_sG = r.get('PseudoPotential') * Ha
    ps = r['PoissonStencil']
    if isinstance(ps, int) or ps == 'M':
        poisson = {'name': 'fd'}
        poisson['nn'] = ps
        if data['atoms.']['pbc'] == [1, 1, 0]:
            v1, v2 = vt_sG[0, :, :, [0, -1]].mean(axis=(1, 2))
            if abs(v1 - v2) > 0.01:
                warnings.warn('I am guessing that this calculation was done '
                              'with a dipole-layer correction?')
                poisson['dipolelayer'] = 'xy'
        p['poissonsolver'] = poisson

    p['charge'] = r['Charge']
    fixmom = r['FixMagneticMoment']

    p['occupations'] = FermiDirac(r['FermiWidth'] * Ha,
                                  fixmagmom=fixmom)

    p['mode'] = r['Mode']

    if p['mode'] == 'pw':
        p['mode'] = PW(ecut=r['PlaneWaveCutoff'] * Ha)

    if len(bzk_kc) == 1 and not bzk_kc[0].any():
        # Gamma point only:
        if r['DataType'] == 'Complex':
            p['dtype'] = complex

    data['occupations.'] = {
        'fermilevel': r['FermiLevel'] * Ha,
        'split': r.parameters.get('FermiSplit', 0) * Ha,
        'homo': np.nan,
        'lumo': np.nan}

    data['density.'] = {
        'density': r.get('PseudoElectronDensity') * Bohr**-3,
        'atomic_density_matrices': r.get('AtomicDensityMatrices')}

    data['hamiltonian.'] = {
        'e_coulomb': r['Epot'] * Ha,
        'e_entropy': -r['S'] * Ha,
        'e_external': r['Eext'] * Ha,
        'e_kinetic': r['Ekin'] * Ha,
        'e_total_extrapolated': e_total_extrapolated,
        'e_xc': r['Exc'] * Ha,
        'e_zero': r['Ebar'] * Ha,
        'potential': vt_sG,
        'atomic_hamiltonian_matrices': r.get('NonLocalPartOfHamiltonian') * Ha}
    data['hamiltonian.']['e_total_free'] = (
        sum(data['hamiltonian.'][e] for e in ['e_coulomb', 'e_entropy',
                                              'e_external', 'e_kinetic',
                                              'e_xc', 'e_zero']))

    if r.has_array('GLLBPseudoResponsePotential'):
        data['hamiltonian.']['xc.'] = {
            'gllb_pseudo_response_potential':
            r.get('GLLBPseudoResponsePotential') * Ha,
            'gllb_dxc_pseudo_response_potential':
            r.get('GLLBDxcPseudoResponsePotential') * Ha / Bohr,
            'gllb_atomic_density_matrices':
            r.get('GLLBAtomicDensityMatrices'),
            'gllb_atomic_response_matrices':
            r.get('GLLBAtomicResponseMatrices'),
            'gllb_dxc_atomic_density_matrices':
            r.get('GLLBDxcAtomicDensityMatrices'),
            'gllb_dxc_atomic_response_matrices':
            r.get('GLLBDxcAtomicResponseMatrices')}

    special = [('eigenvalues', 'Eigenvalues'),
               ('occupations', 'OccupationNumbers'),
               ('projections', 'Projections')]

    if r['Mode'] == 'pw':
        special.append(('coefficients', 'PseudoWaveFunctions'))
        try:
            data['wave_functions.']['indices'] = r.get('PlaneWaveIndices')
        except KeyError:
            pass
    elif r['Mode'] == 'fd':
        special.append(('values', 'PseudoWaveFunctions'))
    else:
        special.append(('coefficients', 'WaveFunctionCoefficients'))

    for name, old in special:
        try:
            fd, shape, size, dtype = r.get_file_object(old, ())
        except KeyError:
            continue
        offset = fd
        data['wave_functions.'][name + '.'] = {
            'ndarray': (shape, dtype.name, offset)}

    new = ulm.Reader(devnull, data=data,
                     little_endian=r.byteswap ^ np.little_endian)

    for ref in new._data['wave_functions']._data.values():
        try:
            ref.fd = ref.offset
        except AttributeError:
            continue
        ref.offset = 0

    return new


def read_atoms(reader):
    positions = reader.get('CartesianPositions', broadcast=True) * Bohr
    numbers = reader.get('AtomicNumbers', broadcast=True)
    cell = reader.get('UnitCell', broadcast=True) * Bohr
    pbc = reader.get('BoundaryConditions', broadcast=True)
    tags = reader.get('Tags', broadcast=True)
    magmoms = reader.get('MagneticMoments', broadcast=True)

    # Create instance of Atoms object, and set_tags and magnetic moments
    atoms = Atoms(positions=positions,
                  numbers=numbers,
                  cell=cell,
                  pbc=pbc)

    if tags.any():
        atoms.set_tags(tags)
    if magmoms.any():
        atoms.set_initial_magnetic_moments(magmoms)

    if reader.has_array('CartesianVelocities'):
        velocities = reader.get('CartesianVelocities',
                                broadcast=True) * Bohr / AUT
        atoms.set_velocities(velocities)

    return atoms
