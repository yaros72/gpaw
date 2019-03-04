# Test that the atomic corrections of LCAO work correctly,
# by verifying that the different implementations yield the same numbers.
#
# For example the corrections P^* dH P to the Hamiltonian.
#
# This is done by invoking GPAW once for each type of calculation.

from ase.build import molecule
from gpaw import GPAW, LCAO, PoissonSolver
from gpaw.lcao.atomic_correction import (DenseAtomicCorrection,
                                         ScipyAtomicCorrection)
from gpaw.mpi import world

# Use a cell large enough that some overlaps are zero.
# Thus the matrices will have at least some sparsity.
system = molecule('H2O')
system.center(vacuum=3.0)
system.pbc = (0, 0, 1)
system = system.repeat((1, 1, 2))
# Break symmetries so we don't get funny degeneracy effects.
system.rattle(stdev=0.05)

corrections = [DenseAtomicCorrection(), ScipyAtomicCorrection(tolerance=0.0)]


energies = []
for correction in corrections:
    parallel = {}
    if world.size >= 4:
        parallel['band'] = 2
    if correction.name != 'dense':
        parallel['sl_auto'] = True
    calc = GPAW(mode=LCAO(atomic_correction=correction),
                basis='sz(dzp)',
                #kpts=(1, 1, 4),
                #spinpol=True,
                poissonsolver=PoissonSolver('fd', relax='J', eps=1e100, nn=1),
                parallel=parallel,
                h=0.35)

    def stopcalc():
        calc.scf.converged = True

    calc.attach(stopcalc, 2)
    system.set_calculator(calc)
    energy = system.get_potential_energy()
    energies.append(energy)

master = calc.wfs.world.rank == 0
if master:
    print('energies', energies)

eref = energies[0]
errs = []
for energy, c in zip(energies, corrections):
    err = abs(energy - eref)
    nops = calc.wfs.world.sum(c.nops)
    errs.append(err)
    if master:
        print('err=%e :: name=%s :: nops=%d' % (err, c.name, nops))

assert max(errs) < 1e-11
