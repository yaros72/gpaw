from __future__ import print_function
import numpy as np

from gpaw import __version__ as version
from gpaw.mpi import world
from gpaw.tddft.units import (au_to_as, au_to_fs, au_to_eV)
from gpaw.tddft.folding import FoldedFrequencies
from gpaw.tddft.folding import Folding


def read_dipole_moment_file(fname, remove_duplicates=True):
    def parse_kick_line(line):
        kick_str_v = line.split('[', 1)[1].split(']', 1)[0].split(',')
        kick_v = np.array([float(x) for x in kick_str_v])
        data_i = line.split('Time =')
        if len(data_i) == 1:
            time = 0.0
        else:
            time = float(data_i[1])
        return kick_v, time

    # Search kicks
    kick_i = []
    with open(fname, 'r') as f:
        for line in f:
            if line.startswith('# Kick'):
                kick_v, time = parse_kick_line(line)
                kick_i.append({'strength_v': kick_v, 'time': time})

    # Read data
    data_tj = np.loadtxt(fname)
    time_t = data_tj[:, 0]
    norm_t = data_tj[:, 1]
    dm_tv = data_tj[:, 2:]

    # Remove duplicates due to abruptly stopped and restarted calculation
    if remove_duplicates:
        flt_t = np.ones_like(time_t, dtype=bool)
        maxtime = time_t[0]
        for t in range(1, len(time_t)):
            if time_t[t] > maxtime:
                maxtime = time_t[t]
            else:
                flt_t[t] = False

        time_t = time_t[flt_t]
        norm_t = norm_t[flt_t]
        dm_tv = dm_tv[flt_t]

        ndup = len(flt_t) - flt_t.sum()
        if ndup > 0:
            print('Removed %d duplicates' % ndup)

    return kick_i, time_t, norm_t, dm_tv


def calculate_polarizability(data, foldedfrequencies):
    kick_v, time_t, dm_tv = data
    ff = foldedfrequencies
    omega_w = ff.frequencies
    envelope = ff.folding.envelope

    time_t = time_t - time_t[0]
    dt_t = np.insert(time_t[1:] - time_t[:-1], 0, 0.0)
    dm_tv = dm_tv[:] - dm_tv[0]

    kick_magnitude = np.sum(kick_v**2)

    Nw = len(omega_w)
    alpha_wv = np.zeros((Nw, 3), dtype=complex)
    f_wt = np.exp(1.0j * np.outer(omega_w, time_t))
    dm_vt = np.swapaxes(dm_tv, 0, 1)
    alpha_wv = np.tensordot(f_wt, dt_t * envelope(time_t) * dm_vt, axes=(1, 1))
    alpha_wv *= kick_v / kick_magnitude
    return alpha_wv


def calculate_photoabsorption(data, foldedfrequencies):
    omega_w = foldedfrequencies.frequencies
    alpha_wv = calculate_polarizability(data, foldedfrequencies)
    abs_wv = 2 / np.pi * omega_w[:, np.newaxis] * alpha_wv.imag
    return abs_wv


def write_spectrum(dipole_moment_file, spectrum_file,
                   folding, width, e_min, e_max, delta_e,
                   title, symbol, calculate):
    def str_list(v_i, fmt='%g'):
        return '[%s]' % ', '.join(map(lambda v: fmt % v, v_i))

    r = read_dipole_moment_file(dipole_moment_file)
    kick_i, time_t, norm_t, dm_tv = r
    dt_t = time_t[1:] - time_t[:-1]

    if len(kick_i) > 1:
        raise RuntimeError('Multiple kicks in %s' % dipole_moment_file)
    kick = kick_i[0]
    kick_v = kick['strength_v']
    kick_time = kick['time']

    # Discard times before kick
    flt_t = time_t > (kick_time - 0.5 * dt_t.min())
    time_t = time_t[flt_t]
    norm_t = norm_t[flt_t]
    dm_tv = dm_tv[flt_t]
    dt_t = time_t[1:] - time_t[:-1]

    freqs = np.arange(e_min, e_max + 0.5 * delta_e, delta_e)
    folding = Folding(folding, width)
    ff = FoldedFrequencies(freqs, folding)
    omega_w = ff.frequencies
    spec_wv = calculate((kick_v, time_t, dm_tv), ff)

    # Write spectrum file header
    with open(spectrum_file, 'w') as f:
        def w(s):
            f.write('%s\n' % s)

        w('# %s spectrum from real-time propagation' % title)
        w('# GPAW version: %s' % version)
        w('# Total time = %.4f fs, Time steps = %s as' %
          (dt_t.sum() * au_to_fs,
           str_list(np.unique(np.around(dt_t, 6)) * au_to_as, '%.4f')))
        w('# Kick = %s' % str_list(kick_v))
        w('# %sian folding, Width = %.4f eV = %lf Hartree'
          ' <=> FWHM = %lf eV' %
          (folding.folding, folding.width * au_to_eV, folding.width,
           folding.fwhm * au_to_eV))

        col_i = []
        data_iw = [omega_w * au_to_eV]
        for v in range(len(kick_v)):
            h = '%s_%s' % (symbol, 'xyz'[v])
            if spec_wv.dtype == complex:
                col_i.append('Re[%s]' % h)
                data_iw.append(spec_wv[:, v].real)
                col_i.append('Im[%s]' % h)
                data_iw.append(spec_wv[:, v].imag)
            else:
                col_i.append(h)
                data_iw.append(spec_wv[:, v])

        w('# %10s %s' % ('om (eV)', ' '.join(['%20s' % s for s in col_i])))

    # Write spectrum file data
    with open(spectrum_file, 'ab') as f:
        np.savetxt(f, np.array(data_iw).T,
                   fmt='%12.6lf' + (' %20.10le' * len(col_i)))

    return folding.envelope(time_t[-1])


def photoabsorption_spectrum(dipole_moment_file, spectrum_file,
                             folding='Gauss', width=0.2123,
                             e_min=0.0, e_max=30.0, delta_e=0.05):
    """Calculates photoabsorption spectrum from the time-dependent
    dipole moment.

    Parameters:

    dipole_moment_file: string
        Name of the time-dependent dipole moment file from which
        the spectrum is calculated
    spectrum_file: string
        Name of the spectrum file
    folding: 'Gauss' or 'Lorentz'
        Whether to use Gaussian or Lorentzian folding
    width: float
        Width of the Gaussian (sigma) or Lorentzian (Gamma)
        Gaussian =     1/(sigma sqrt(2pi)) exp(-(1/2)(omega/sigma)^2)
        Lorentzian =  (1/pi) (1/2) Gamma / [omega^2 + ((1/2) Gamma)^2]
    e_min: float
        Minimum energy shown in the spectrum (eV)
    e_max: float
        Maximum energy shown in the spectrum (eV)
    delta_e: float
        Energy resolution (eV)
    """
    if world.rank == 0:
        print('Calculating photoabsorption spectrum from file "%s"'
              % dipole_moment_file)

        def calculate(*args):
            return calculate_photoabsorption(*args) / au_to_eV
        sinc = write_spectrum(dipole_moment_file, spectrum_file,
                              folding, width, e_min, e_max, delta_e,
                              'Photoabsorption', 'S', calculate)
        print('Sinc contamination %.8f' % sinc)
        print('Calculated photoabsorption spectrum saved to file "%s"'
              % spectrum_file)


def polarizability_spectrum(dipole_moment_file, spectrum_file,
                            folding='Gauss', width=0.2123,
                            e_min=0.0, e_max=30.0, delta_e=0.05):
    """Calculates polarizability spectrum from the time-dependent
    dipole moment.

    Parameters:

    dipole_moment_file: string
        Name of the time-dependent dipole moment file from which
        the spectrum is calculated
    spectrum_file: string
        Name of the spectrum file
    folding: 'Gauss' or 'Lorentz'
        Whether to use Gaussian or Lorentzian folding
    width: float
        Width of the Gaussian (sigma) or Lorentzian (Gamma)
        Gaussian =     1/(sigma sqrt(2pi)) exp(-(1/2)(omega/sigma)^2)
        Lorentzian =  (1/pi) (1/2) Gamma / [omega^2 + ((1/2) Gamma)^2]
    e_min: float
        Minimum energy shown in the spectrum (eV)
    e_max: float
        Maximum energy shown in the spectrum (eV)
    delta_e: float
        Energy resolution (eV)
    """
    if world.rank == 0:
        print('Calculating polarizability spectrum from file "%s"'
              % dipole_moment_file)

        def calculate(*args):
            return calculate_polarizability(*args) / au_to_eV**2
        sinc = write_spectrum(dipole_moment_file, spectrum_file,
                              folding, width, e_min, e_max, delta_e,
                              'Polarizability', 'alpha', calculate)
        print('Sinc contamination %.8f' % sinc)
        print('Calculated polarizability spectrum saved to file "%s"'
              % spectrum_file)
