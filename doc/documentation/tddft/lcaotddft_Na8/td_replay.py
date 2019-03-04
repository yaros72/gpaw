from gpaw.lcaotddft import LCAOTDDFT
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter

# Read the ground-state file
td_calc = LCAOTDDFT('gs.gpw')

# Attach analysis tools
DipoleMomentWriter(td_calc, 'dm_replayed.dat')

# Replay the propagation
td_calc.replay(name='wf.ulm', update='all')
