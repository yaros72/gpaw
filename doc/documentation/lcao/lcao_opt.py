from ase.build import molecule
from ase.optimize import QuasiNewton
from gpaw import GPAW

atoms = molecule('CH3CH2OH', vacuum=4.0)
atoms.rattle(stdev=0.1)  # displace positions randomly a bit

calc = GPAW(mode='lcao',
            basis='dzp',
            nbands='110%',
            parallel=dict(band=2,  # band parallelization
                          augment_grids=True,  # use all cores for XC/Poisson
                          sl_auto=True,  # enable ScaLAPACK parallelization
                          use_elpa=True))  # enable Elpa eigensolver
atoms.calc = calc

opt = QuasiNewton(atoms, trajectory='opt.traj')
opt.run(fmax=0.05)
