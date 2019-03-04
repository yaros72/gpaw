from math import exp, pi, sqrt
import numpy as np

from gpaw.gauss import Gauss
from gpaw.test import equal
from gpaw.utilities.folder import Folder, Lorentz, Voigt  # noqa

# Gauss and Lorentz functions

width = 0.5
x = 1.5

equal(Gauss(width).get(x),
      exp(- x**2 / 2 / width**2) / sqrt(2 * pi) / width,
      1.e-15)
equal(Gauss(width).fwhm, width * np.sqrt(8 * np.log(2)), 1.e-15) 
equal(Lorentz(width).get(x),
      width / (x**2 + width**2) / pi,
      1.e-15)
equal(Lorentz(width).fwhm, width * 2, 1.e-15) 

# folder function

for func in [Gauss, Lorentz, Voigt]:
    folder = Folder(width, func(width).__class__.__name__)

    x = [0, 2]
    y = [[2, 0, 1], [1, 1, 1]]

    xl, yl = folder.fold(x, y, dx=.7)

    # check first value
    yy = np.dot(np.array(y)[:, 0], func(width).get(xl[0] - np.array(x)))
    equal(yl[0, 0], yy, 1.e-15)
