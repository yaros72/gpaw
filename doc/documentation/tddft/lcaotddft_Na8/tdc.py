from gpaw.lcaotddft import LCAOTDDFT
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter
from gpaw.lcaotddft.wfwriter import WaveFunctionWriter

# Read the restart file
td_calc = LCAOTDDFT('td.gpw', txt='tdc.out')

# Attach the data recording for appending the new data
DipoleMomentWriter(td_calc, 'dm.dat')
WaveFunctionWriter(td_calc, 'wf.ulm')

# Continue propagation
td_calc.propagate(20, 500)

# Save the state for restarting later
td_calc.write('td.gpw', mode='all')
