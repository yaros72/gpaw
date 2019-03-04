from __future__ import print_function
import sys
import numpy as np

from ase.units import _hbar, _c, _e, _me, Hartree, Bohr
from gpaw import __version__ as version
from gpaw.utilities.folder import Folder


def get_folded_spectrum(
        exlist=None,
        emin=None,
        emax=None,
        de=None,
        energyunit='eV',
        folding='Gauss',
        width=0.08,  # Gauss/Lorentz width
        form='r'):
    """Return folded spectrum."""

    x = []
    y = []
    for ex in exlist:
        x.append(ex.get_energy() * Hartree)
        y.append(ex.get_oscillator_strength(form))

    if energyunit == 'nm':
        # transform to experimentally used wavelength [nm]
        x = 1.e+9 * 2 * np.pi * _hbar * _c / _e / np.array(x)
    elif energyunit != 'eV':
        raise RuntimeError('currently only eV and nm are supported')

    return Folder(width, folding).fold(x, y, de, emin, emax)


def spectrum(exlist=None,
             filename=None,
             emin=None,
             emax=None,
             de=None,
             energyunit='eV',
             folding='Gauss',
             width=0.08,  # Gauss/Lorentz width
             comment=None,
             form='r'):
    """Write out a folded spectrum.

    Parameters
    ----------
    exlist: ExcitationList
    filename:
        File name for the output file, STDOUT if not given
    emin:
        min. energy, set to cover all energies if not given
    emax:
        max. energy, set to cover all energies if not given
    de:
        energy spacing
    energyunit: {'eV', 'nm'}
        Energy unit, default 'eV'
    folding: {'Gauss', 'Lorentz'}
    width:
        folding width in terms of the chosen energyunit
    """

    # output
    out = sys.stdout
    if filename is not None:
        out = open(filename, 'w')
    if comment:
        print('#', comment, file=out)

    print('# Photoabsorption spectrum from linear response TD-DFT', file=out)
    print('# GPAW version:', version, file=out)
    if folding is not None:  # fold the spectrum
        print('# %s folded, width=%g [%s]' % (folding, width,
                                              energyunit), file=out)
    if form == 'r':
        out.write('# length form')
    else:
        assert(form == 'v')
        out.write('# velocity form')
    print('# om [%s]     osz          osz x       osz y       osz z'
          % energyunit, file=out)

    energies, values = get_folded_spectrum(exlist, emin, emax, de,
                                           energyunit, folding,
                                           width, form)

    for e, val in zip(energies, values):
        print('%10.5f %12.7e %12.7e %11.7e %11.7e' %
              (e, val[0], val[1], val[2], val[3]), file=out)

    if filename is not None:
        out.close()

        
def get_adsorbance_pre_factor(atoms):
    """Return the absorbance pre-factor for solids. Unit m^-1.

    Use this factor to multiply the folded oscillatior strength
    obtained from spectrum()

    Robert C. Hilborn, Am. J. Phys. 50, 982 (1982)
    """
    return np.pi * _e**2 / 2. / _me / _c

    
def rotatory_spectrum(exlist=None,
                      filename=None,
                      emin=None,
                      emax=None,
                      de=None,
                      energyunit='eV',
                      folding='Gauss',
                      width=0.08,  # Gauss/Lorentz width
                      comment=None
                      ):
    """Write out a folded rotatory spectrum.

    See spectrum() for explanation of the parameters.
    """

    # output
    out = sys.stdout
    if filename is not None:
        out = open(filename, 'w')
    if comment:
        print('#', comment, file=out)

    print('# Rotatory spectrum from linear response TD-DFT', file=out)
    print('# GPAW version:', version, file=out)
    if folding is not None:  # fold the spectrum
        print('# %s folded, width=%g [%s]' % (folding, width,
                                              energyunit), file=out)
    print('# om [%s]     R [cgs]'
          % energyunit, file=out)

    x = []
    y = []
    for ex in exlist:
        x.append(ex.get_energy() * Hartree)
        y.append(ex.get_rotatory_strength())

    if energyunit == 'nm':
        # transform to experimentally used wavelength [nm]
        x = 1.e+9 * 2 * np.pi * _hbar * _c / _e / np.array(x)
        y = np.array(y)
    elif energyunit != 'eV':
        raise RuntimeError('currently only eV and nm are supported')

    energies, values = Folder(width, folding).fold(x, y, de, emin, emax)
    for e, val in zip(energies, values):
        print('%10.5f %12.7e' % (e, val), file=out)

    if filename is not None:
        out.close()


class Writer(Folder):
    """Object to write data to a file"""
    def __init__(self, folding=None, width=0.08):
        """Evaluate the polarizability from sum over states.

        Parameters
        ----------
        folding : string or None(default)
          Name of the folding function, see gpaw/folder.py for options.
          None means do not fold at all.
        width : float
          The width to be transferred to the folding function. 
        """
        self.folding = folding
        if folding is not None:
            Folder.__init__(self, width, folding)

    def write(self, filename=None,
              emin=None, emax=None, de=None,
              comment=None):

        out = sys.stdout
        if filename is not None:
            out = open(filename, 'w')

        print('#', self.title, file=out)
        print('# GPAW version:', version, file=out)
        if comment:
            print('#', comment, file=out)
        if self.folding is not None:
            print('# %s folded, width=%g [eV]' % (self.folding,
                                                  self.width), file=out)
            energies, values = self.fold(self.energies, self.values,
                                         de, emin, emax)
        else:
            energies, values = self.energies, self.values

        print('#', self.fields, file=out)

        for e, val in zip(energies, values):
            string = '%10.5f' % e
            for vf in val:
                string += ' %12.7e' % vf
            print(string, file=out)

        if filename is not None:
            out.close()

            
def polarizability(exlist, omega, form='v',
                   tensor=False, index=0):
    """Evaluate the polarizability from sum over states.

    Parameters
    ----------
    exlist: ExcitationList
    omega:
        Photon energy (eV)
    form: {'v', 'r'}
        Form of the dipole matrix element
    index: {0, 1, 2, 3}
        0: averaged, 1,2,3:alpha_xx, alpha_yy, alpha_zz
    tensor:
        if True returns alpha_ij, i,j=x,y,z
        index is ignored

    Returns
    -------
    alpha:
        Unit (e^2 Angstrom^2 / eV).
        Multiply with Bohr * Ha to get (Angstrom^3)
    """
    if tensor:
        alpha = np.zeros(np.array(omega).shape + (3, 3), dtype=complex)
        for ex in exlist:
            alpha += ex.get_dipole_tensor(form=form) / (
                (ex.energy * Hartree)**2 - omega**2)
    else:
        alpha = np.zeros_like(omega)
        for ex in exlist:
            alpha += ex.get_oscillator_strength(form=form)[index] / (
                (ex.energy * Hartree)**2 - omega**2)
            
    return alpha * Bohr**2 * Hartree
