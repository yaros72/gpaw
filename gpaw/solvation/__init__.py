"""The gpaw.solvation package.

This packages extends GPAW to be used with different
continuum solvent models.
"""

from gpaw.solvation.calculator import SolvationGPAW
from gpaw.solvation.cavity import (EffectivePotentialCavity,
                                   Power12Potential,
                                   ElDensity,
                                   SSS09Density,
                                   ADM12SmoothStepCavity,
                                   FG02SmoothStepCavity,
                                   GradientSurface,
                                   KB51Volume)
from gpaw.solvation.dielectric import (LinearDielectric,
                                       CMDielectric)
from gpaw.solvation.interactions import (SurfaceInteraction,
                                         VolumeInteraction,
                                         LeakedDensityInteraction)


def get_HW14_water_kwargs():
    """Return kwargs for initializing a SolvationGPAW instance.

    Parameters for water as a solvent as in
    A. Held and M. Walter, J. Chem. Phys. 141, 174108 (2014).
    """
    from ase.units import Pascal, m
    from ase.data.vdw import vdw_radii
    u0 = 0.180
    epsinf = 78.36
    st = 18.4 * 1e-3 * Pascal * m
    T = 298.15
    vdw_radii = vdw_radii.copy()
    vdw_radii[1] = 1.09
    atomic_radii = lambda atoms: [vdw_radii[n] for n in atoms.numbers]
    kwargs = {
        'cavity': EffectivePotentialCavity(
            effective_potential=Power12Potential(atomic_radii, u0),
            temperature=T,
            surface_calculator=GradientSurface()
        ),
        'dielectric': LinearDielectric(epsinf=epsinf),
        'interactions': [SurfaceInteraction(surface_tension=st)]
    }
    return kwargs


__all__ = [SolvationGPAW,
           EffectivePotentialCavity,
           Power12Potential,
           ElDensity,
           SSS09Density,
           ADM12SmoothStepCavity,
           FG02SmoothStepCavity,
           GradientSurface,
           KB51Volume,
           LinearDielectric,
           CMDielectric,
           SurfaceInteraction,
           VolumeInteraction,
           LeakedDensityInteraction,
           get_HW14_water_kwargs]
