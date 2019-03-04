"""Check TDDFT ionizations with Yukawa potential."""
from ase import Atoms
from ase.units import Hartree
from gpaw import GPAW
from gpaw.mpi import world
from gpaw.cluster import Cluster
from gpaw.occupations import FermiDirac
from gpaw.test import equal
from gpaw.eigensolvers import RMMDIIS
from gpaw.lrtddft import LrTDDFT

o_plus = Cluster(Atoms('Be', positions=[[0, 0, 0]]))
o_plus.set_initial_magnetic_moments([1.0])
o_plus.minimal_box(2.5, h=0.35)


def get_paw():
    """Return calculator object."""
    c = {'energy': 0.05, 'eigenstates': 0.05, 'density': 0.05}
    return GPAW(convergence=c, eigensolver=RMMDIIS(),
                nbands=3,
                xc='PBE',
#                experimental={'niter_fixdensity': 2},
                parallel={'domain': world.size}, h=0.35,
                occupations=FermiDirac(width=0.0, fixmagmom=True))


calc_plus = get_paw()
calc_plus.set(txt='Be_plus_LCY_PBE_083.log', charge=1)
o_plus.set_calculator(calc_plus)
e_o_plus = o_plus.get_potential_energy()
calc_plus.set(xc='LCY-PBE:omega=0.83:unocc=True', experimental={'niter_fixdensity': 2})
e_o_plus = o_plus.get_potential_energy()
lr = LrTDDFT(calc_plus, txt='LCY_TDDFT_Be.log', istart=0, jend=1)
equal(lr.xc.omega, 0.83)
lr.write('LCY_TDDFT_Be.ex.gz')
e_ion = 9.3
ip_i = 13.36
# reading is problematic with EXX on more than one core
if world.rank == 0:
    lr2 = LrTDDFT('LCY_TDDFT_Be.ex.gz')
    lr2.diagonalize()
    equal(lr2.xc.omega, 0.83)
    ion_i = lr2[0].get_energy() * Hartree + e_ion
    equal(ion_i, ip_i, 0.3)
