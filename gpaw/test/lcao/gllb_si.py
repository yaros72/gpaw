from ase.build import bulk
from gpaw import GPAW, LCAO

# This test calculates a GLLB quasiparticle gap with LCAO and verifies
# that it does not change from a reference value.  Note that the
# calculation, physically speaking, is garbage.

si = bulk('Si', 'diamond', a=5.421)
calc = GPAW(mode=LCAO(interpolation=2),
            h=0.3,
            basis='sz(dzp)',
            xc='GLLBSC',
            kpts={'size': (2, 2, 2), 'gamma': True},
            txt='si.txt')


def stopcalc():
    calc.scf.converged = True


calc.attach(stopcalc, 1)

si.set_calculator(calc)
si.get_potential_energy()

response = calc.hamiltonian.xc.xcs['RESPONSE']
response.calculate_delta_xc()
EKs, Dxc = response.calculate_delta_xc_perturbation()
refgap = 3.02333
gap = EKs + Dxc
print('GAP', gap)
err = abs(gap - refgap)
assert err < 1e-4, err
