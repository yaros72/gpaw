from gpaw.lcaotddft import LCAOTDDFT
from gpaw.lcaotddft.densitymatrix import DensityMatrix
from gpaw.lcaotddft.frequencydensitymatrix import FrequencyDensityMatrix
from gpaw.tddft.folding import frequencies

# Read the ground-state file
td_calc = LCAOTDDFT('gs.gpw')

# Attach analysis tools
dmat = DensityMatrix(td_calc)
freqs = frequencies([1.12, 2.48], 'Gauss', 0.1)
fdm = FrequencyDensityMatrix(td_calc, dmat, frequencies=freqs)

# Replay the propagation
td_calc.replay(name='wf.ulm', update='none')

# Store the density matrix
fdm.write('fdm.ulm')
