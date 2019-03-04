from __future__ import print_function
import numpy as np

from gpaw.atom.radialgd import fsbt, RadialGridDescriptor as RGD


N = 1024
L = 50.0
h = L / N
alpha = 5.0
r = np.linspace(0, L, N, endpoint=False)
G = np.linspace(0, np.pi / h, N // 2 + 1)
n = np.exp(-alpha * r**2)

for l in range(7):
    f = fsbt(l, n * r**l, r, G)
    f0 = (np.pi**0.5 / alpha**(l + 1.5) / 2**l * G**l / 4 *
          np.exp(-G**2 / (4 * alpha)))
    tol = 3e-6 * 10**(-7 + l)
    print(l, abs(f - f0).max(), 'tol=', tol)
    assert abs(f - f0).max() < tol

rgd = RGD(r, r * 0 + r[1])
g, f = rgd.fft(n * r)
f0 = 4 * np.pi**1.5 / alpha**1.5 / 4 * np.exp(-g**2 / 4 / alpha)
assert abs(f - f0).max() < 1e-6

# This is how to do the inverse FFT:
ggd = RGD(g, g * 0 + g[1])
r, f = ggd.fft(f * g)
assert abs(np.exp(-alpha * r**2) - f / 8 / np.pi**3).max() < 2e-3
