"""This calculation has the following structure.
1) Calculate the ground state of Diamond.
2) Calculate the band structure of diamond in order to obtain accurate KS
   band gap for Diamond.
3) Calculate ground state again, and calculate the potential discontinuity
   using accurate band gap.
4) Calculate band structure again, and apply the discontinuity to CBM.

Compare to reference.
"""
from ase.build import bulk
from ase.units import Ha
from gpaw import GPAW, Davidson, Mixer, restart
from gpaw.test import gen
from gpaw import setup_paths
from gpaw.mpi import world


xc = 'GLLBSC'
gen('C', xcname=xc)
setup_paths.insert(0, '.')

# Calculate ground state
atoms = bulk('C', 'diamond', a=3.567)
# We want sufficiently many grid points that the calculator
# can use wfs.world for the finegd, to test that part of the code.
calc = GPAW(h=0.2,
            kpts=(4, 4, 4),
            xc=xc,
            nbands=8,
            mixer=Mixer(0.5, 5, 10.0),
            parallel=dict(domain=min(world.size, 2),
                          band=1),
            eigensolver=Davidson(niter=2))
atoms.set_calculator(calc)
atoms.get_potential_energy()
calc.write('Cgs.gpw')

# Calculate accurate KS-band gap from band structure
calc = GPAW('Cgs.gpw',
            kpts={'path': 'GX', 'npoints': 12},
            fixdensity=True,
            symmetry='off',
            nbands=8,
            convergence=dict(bands=6),
            eigensolver=Davidson(niter=4))
calc.get_atoms().get_potential_energy()
# Get the accurate KS-band gap
homolumo = calc.get_homo_lumo()
homo, lumo = homolumo
print('band gap', lumo - homo)

# Redo the ground state calculation
calc = GPAW(h=0.2, kpts=(4, 4, 4), xc=xc, nbands=8,
            mixer=Mixer(0.5, 5, 10.0),
            eigensolver=Davidson(niter=4))
atoms.set_calculator(calc)
atoms.get_potential_energy()
# And calculate the discontinuity potential with accurate band gap
response = calc.hamiltonian.xc.xcs['RESPONSE']
response.calculate_delta_xc(homolumo=homolumo / Ha)
calc.write('CGLLBSC.gpw')

# Redo the band structure calculation
atoms, calc = restart('CGLLBSC.gpw',
                      kpts={'path': 'GX', 'npoints': 12},
                      fixdensity=True,
                      symmetry='off',
                      convergence=dict(bands=6),
                      eigensolver=Davidson(niter=4))
atoms.get_potential_energy()
response = calc.hamiltonian.xc.xcs['RESPONSE']
KS, dxc = response.calculate_delta_xc_perturbation()

energy = KS + dxc
ref = 5.41
err = abs(energy - ref)
if calc.wfs.world.rank == 0:
    print('energy', energy, 'ref', ref)
assert err < 0.10, err
# M. Kuisma et. al, Phys. Rev. B 82, 115106, QP gap for C, 5.41eV, expt. 5.48eV
