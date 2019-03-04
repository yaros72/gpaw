from pathlib import Path
import numpy as np
from gpaw import GPAW
from gpaw.spinorbit import get_anisotropy

theta_i = [i * np.pi / 20 for i in range(21)]
for theta in theta_i:
    calc = GPAW('gs_Co.gpw', txt=None)
    E_so = get_anisotropy(calc, theta=theta, phi=0.0)
    with open('anisotropy.dat', 'a') as f:
        print(theta, E_so, file=f)
Path('gs_Co.gpw').unlink()  # remove very large file
