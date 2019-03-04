from gpaw.lcaotddft import LCAOTDDFT
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter
from gpaw.lcaotddft.wfwriter import WaveFunctionWriter

# Read the ground-state file
td_calc = LCAOTDDFT('gs.gpw', txt='td.out')

# Attach any data recording or analysis tools
DipoleMomentWriter(td_calc, 'dm.dat')
WaveFunctionWriter(td_calc, 'wf.ulm')

# Kick and propagate
td_calc.absorption_kick([1e-5, 0., 0.])
td_calc.propagate(20, 1500)

# Save the state for restarting later
td_calc.write('td.gpw', mode='all')
