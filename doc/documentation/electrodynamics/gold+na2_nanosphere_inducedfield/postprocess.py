from gpaw.tddft import TDDFT
from gpaw.inducedfield.inducedfield_fdtd import (
    FDTDInducedField, calculate_hybrid_induced_field)
from gpaw.inducedfield.inducedfield_tddft import TDDFTInducedField

td_calc = TDDFT('td.gpw')

# Classical subsystem
cl_ind = FDTDInducedField(filename='cl.ind', paw=td_calc)
cl_ind.calculate_induced_field(gridrefinement=2)
cl_ind.write('cl_field.ind', mode='all')

# Quantum subsystem
qm_ind = TDDFTInducedField(filename='qm.ind', paw=td_calc)
qm_ind.calculate_induced_field(gridrefinement=2)
qm_ind.write('qm_field.ind', mode='all')

# Total system, interpolate/extrapolate to a grid with spacing h
tot_ind = calculate_hybrid_induced_field(cl_ind, qm_ind, h=1.0)
tot_ind.write('tot_field.ind', mode='all')
