import sys

import gpaw.mpi as mpi

from gpaw.response.df import DielectricFunction


class TransverseMagneticSusceptibility(DielectricFunction):
    """
    This class can calculate the transverse magnetic susceptibility
    and related physical quantities.
    """

    def __init__(self, calc, response='+-', name=None, frequencies=None,
                 domega0=0.1, omega2=10.0, omegamax=None,
                 ecut=400, nbands=None, eta=0.02,
                 ftol=1e-6, threshold=1, gammacentered=True,
                 nblocks=1, world=mpi.world, txt=sys.stdout,
                 gate_voltage=None, integrationmode=None, pbc=None, rate=0.0,
                 omegacutlower=None, omegacutupper=None, eshift=0.0):

        assert response in ['+-', '-+']
        
        hilbert = False
        disable_point_group = True
        disable_time_reversal = True
        DielectricFunction.__init__(**locals())

        assert self.chi0.eta > 0.0
        assert not self.chi0.timeordered
