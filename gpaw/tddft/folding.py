import numpy as np

from gpaw.tddft.units import au_to_eV
from gpaw.tddft.units import eV_to_au


def frequencies(frequencies, foldings, widths, units='eV'):
    f_w = []
    for folding in np.array([foldings]).ravel():
        for width in np.array([widths]).ravel():
            folding = Folding(folding, width, units)
            f_w.append(FoldedFrequencies(frequencies, folding, units))
    return f_w


def convert_to_au(val, units='eV'):
    if units == 'eV':
        return val * eV_to_au
    elif units == 'au':
        return val
    raise RuntimeError('Unknown units: %s' % units)


class FoldedFrequencies(object):
    def __init__(self, frequencies, folding, units='eV'):
        freqs = np.array([frequencies], dtype=float).ravel()
        self.frequencies = convert_to_au(freqs, units=units)
        if isinstance(folding, dict):
            self.folding = Folding(**folding)
        else:
            self.folding = folding

    def todict(self):
        d = dict(units='au')
        for arg in ['frequencies', 'folding']:
            val = getattr(self, arg)
            if hasattr(val, 'todict'):
                val = val.todict()
            d[arg] = val
        return d

    def __repr__(self):
        s = 'With %s: ' % self.folding
        s += ', '.join(['%.5f' % (f * au_to_eV) for f in self.frequencies])
        s += ' eV'
        return s


class Frequency(object):
    def __init__(self, freq, folding, units='eV'):
        ff = FoldedFrequencies(freq, folding, units)
        self.folding = ff.folding
        self.freq = ff.frequencies[0]


class Folding(object):
    def __init__(self, folding, width, units='eV'):
        if width is None:
            folding = None

        self.folding = folding
        if self.folding is None:
            self.width = None
        else:
            self.width = convert_to_au(float(width), units=units)

        if self.folding not in [None, 'Gauss', 'Lorentz']:
            raise RuntimeError('Unknown folding: %s' % self.folding)

        if self.folding is None:
            self.fwhm = 0.0
        elif self.folding == 'Gauss':
            self.fwhm = self.width * (2. * np.sqrt(2. * np.log(2.0)))
        elif self.folding == 'Lorentz':
            self.fwhm = 2. * self.width

    def envelope(self, time):
        if self.folding is None:
            return 0 * time + 1
        elif self.folding == 'Gauss':
            return np.exp(- 0.5 * self.width**2 * time**2)
        elif self.folding == 'Lorentz':
            return np.exp(- self.width * time)

    def todict(self):
        d = dict(units='au')
        for arg in ['folding', 'width']:
            d[arg] = getattr(self, arg)
        return d

    def __repr__(self):
        if self.folding is None:
            return 'No folding'
        return '%s(%.5f eV)' % (self.folding, self.width * au_to_eV)
