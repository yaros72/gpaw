import numpy as np

from gpaw.gauss import Gauss

class Lorentz:
    """Normalized Lorentzian distribution"""
    def __init__(self, width=0.08):
        self.dtype = float
        self.set_width(width)

    def get(self, x, x0=0):
        return self.norm / ((x - x0)**2 + self.width2)

    def set_width(self, width=0.08):
        self.norm = width / np.pi
        self.width2 = width**2
        self._fwhm = 2. * width

    @property
    def fwhm(self):
        return self._fwhm

    @fwhm.setter
    def fwhm(self, value):
        self.set_width(value / 2.)


class Voigt:
    """Voigt profile.

    See http://reference.wolfram.com/language/ref/VoigtDistribution.html"""
    def __init__(self, width=0.08):
        self.dtype = float
        self.set_width(width)
        
    def get(self, x, x0=0):
        from scipy.special import erfc
        argm = (-1j * (x - x0) + self.delta) * self.argpre
        argp = (1j * (x - x0) + self.delta) * self.argpre
        res = np.exp(argm**2) * erfc(argm)
        res += np.exp(argp**2) * erfc(argp)
        return res.real * self.prefactor
        
    def set_width(self, width=0.08):
        """Width is interpreted as [delta, sigma]"""
        if not hasattr(width, '__iter__'):
            width = [width, width]
        self.delta, self.sigma = width
        self.width = np.linalg.norm(np.array(width))
        self.prefactor = 1. / 2 / np.sqrt(2 * np.pi) / self.sigma
        self.argpre = 1. / np.sqrt(2) / self.sigma

        """Full width at half maximum approximation after
        Olivero, J. J.; R. L. Longbothum
        "Empirical fits to the Voigt line width: A brief review"
        J. Quantitative Spectroscopy and Radiative Transfer. 17 (1977) 233-236
        """
        fl = 2 * self.delta
        fg = np.sqrt(8 * np.log(2)) * self.sigma
        self._fwhm = 0.5346 * fl + np.sqrt(0.2166 * fl**2 + fg**2)

    @property
    def fwhm(self):
        return self._fwhm

    @fwhm.setter
    def fwhm(self, value):
        "setting fwhm is not uniquely defined"
        raise NotImplementedError


class ComplexLorentz:
    def __init__(self, width=0.08):
        self.dtype = complex
        self.set_width(width)
        
    def get(self, x, x0):
        return 0.5 / x0 * (((x + x0) / ((x + x0)**2 + self.width2) -
                            (x - x0) / ((x - x0)**2 + self.width2)) -
                           1.0j * self.width *
                           (1 / ((x + x0)**2 + self.width2) -
                            1 / ((x - x0)**2 + self.width2))
                           )
        
    def set_width(self, width=0.08):
        self.width = width
        self.width2 = width**2


class ComplexGauss:
    def __init__(self, width=0.08):
        self.dtype = complex
        self.set_width(width)
        
    def get(self, x, x0):
        from scipy.special import dawsn
        return 0.5 / x0 * (2 * self.wm1 *
                           (dawsn((x + x0) * self.wm1) -
                            dawsn((x - x0) * self.wm1)) -
                           1.0j * self.norm *
                           (np.exp(-((x + x0) * self.wm1)**2) -
                            np.exp(-((x - x0) * self.wm1)**2))
                           )
    
    def set_width(self, width=0.08):
        self.norm = 1. / width * np.sqrt(np.pi / 2)
        self.wm1 = np.sqrt(.5) / width


class LorentzPole:
    """Pole of a damped harmonic oscillator.

    See: e.g. Frederick Wooten, Optical properties of solids,
    Academic Press Inc. (1972)"""
    def __init__(self, width=0.08, imag=True):
        self.dtype = float
        if imag:
            self.get = self.get_imaginary
        else:
            self.get = self.get_real
        self.set_width(width)
        
    def get_real(self, x, x0):
        return (x0**2 - x**2) / (
            (x0**2 - x**2)**2 + self.width**2 * x**2)
    
    def get_imaginary(self, x, x0):
        return self.width * x / (
            (x0**2 - x**2)**2 + self.width**2 * x**2)
    
    def set_width(self, width=0.08):
        self.width = width
    

class Folder:
    """Fold a function with normalised Gaussians or Lorentzians

    Example: fold a function y(x) by Lorentzians of 0.2 width

    >>> xlist = ... # list of x-values
    >>> ylist = ... # corresponding list of y(x)-values

    >>> from gpaw.utilities.folder import Folder
    >>> fxlist, fylist = Folder(width=0.2, folding='Lorentz').fold(
                     xlist, ylist, dx=0.1, xmin=-1, xmax=1)
"""
    def __init__(self, width,
                 folding='Gauss'):
        self.width = width
        if folding == 'Gauss':
            self.func = Gauss(width)
        elif folding == 'Lorentz':
            self.func = Lorentz(width)
        elif folding == 'ComplexGauss':
            self.func = ComplexGauss(width)
        elif folding == 'ComplexLorentz':
            self.func = ComplexLorentz(width)
        elif folding == 'RealLorentzPole':
            self.func = LorentzPole(width, imag=False)
        elif folding == 'ImaginaryLorentzPole':
            self.func = LorentzPole(width, imag=True)
        elif folding == 'Voigt':
            self.func = Voigt(width)
        else:
            raise RuntimeError('unknown folding "' + folding + '"')

    def fold(self, x, y, dx=None, xmin=None, xmax=None):
        X = np.array(x)
        assert len(X.shape) == 1
        Y = np.array(y)
        assert X.shape[0] == Y.shape[0]

        if xmin is None:
            xmin = np.min(X) - 4 * self.width
        if xmax is None:
            xmax = np.max(X) + 4 * self.width
        if dx is None:
            try:
                dx = self.func.width / 4.
            except AttributeError:
                dx = self.width / 4.

        xl = np.arange(xmin, xmax + 0.5 * dx, dx)
        
        return self.fold_values(x, y, xl)

    def fold_values(self, x, y, xl=None):
        X = np.array(x)
        assert len(X.shape) == 1
        Y = np.array(y)
        assert X.shape[0] == Y.shape[0]

        if xl is None:
            Xl = np.unique(X)
        else:
            Xl = np.array(xl)
            assert len(Xl.shape) == 1
            
        # weight matrix
        weightm = np.empty((Xl.shape[0], X.shape[0]),
                           dtype=self.func.dtype)
        for i, x in enumerate(X):
            weightm[:, i] = self.func.get(Xl, x)

        yl = np.tensordot(weightm, Y, axes=(1, 0))

        return Xl, yl
