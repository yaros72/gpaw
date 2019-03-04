def quick(project='bulk', system=None):
    """Create Python script to get going quickly.
    
    project: str
        Must be 'bulk' or 'molecule'.
    system: str
        A string representing the system ('H2', 'Si').
    """
    if project == 'bulk':
        template = """\
from ase.build import bulk
from gpaw import GPAW

atoms = bulk('{0}')
atoms.calc = GPAW(kpts={{'size': (4, 4, 4)}},
                  txt='{0}.txt')
e = atoms.get_potential_energy()
f = atoms.get_forces()"""
    else:
        template = """\
from ase.build import molecule
from ase.optimize import QuasiNewton
from gpaw import GPAW

atoms = molecule('{0}')
atoms.center(vacuum=3.0)
atoms.calc = GPAW(txt='{0}.txt')
e = atoms.get_potential_energy()
opt = QuasiNewton(atoms, trajectory='{0}.traj')
opt.run(fmax=0.05)"""
    print(template.format(system))
