from __future__ import print_function
import numpy as np
from ase.build import molecule

from gpaw import GPAW
from gpaw.poisson import FDPoissonSolver
from gpaw.lcao.projected_wannier import get_lcao_projections_HSP

atoms = molecule('C2H2')
atoms.center(vacuum=3.0)
calc = GPAW(gpts=(32, 32, 48),
            experimental={'niter_fixdensity': 2},
            poissonsolver=FDPoissonSolver(),
            eigensolver='rmm-diis')
atoms.set_calculator(calc)
atoms.get_potential_energy()

V_qnM, H_qMM, S_qMM, P_aqMi = get_lcao_projections_HSP(
    calc, bfs=None, spin=0, projectionsonly=False)


# Test H and S
eig = sorted(np.linalg.eigvals(np.linalg.solve(S_qMM[0], H_qMM[0])).real)
eig_ref = np.array([-17.879390089021275, -13.248786165187855,
                    -11.431259875271436, -7.125783046831621,
                    -7.125783046831554, 0.5927193710189584,
                    0.5927193710191426, 3.9251078324544673,
                    7.450995963662965, 26.734277387029234])
print(eig)
assert np.allclose(eig, eig_ref)
