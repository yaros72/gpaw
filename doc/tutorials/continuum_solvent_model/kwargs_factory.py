import gpaw.solvation as solv

# ...

# convenient way to use HW14 water parameters:
calc = solv.SolvationGPAW(
    xc='PBE', h=0.2,  # non-solvent DFT parameters
    **solv.get_HW14_water_kwargs()
)

# ...
