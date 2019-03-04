from gpaw.response.g0w0 import G0W0

gw = G0W0(calc='C_groundstate.gpw',
          nbands=30,      # number of bands for calculation of self-energy
          bands=(3, 5),   # VB and CB
          ecut=20.0,      # plane-wave cutoff for self-energy
          filename='C-g0w0',
          savepckl=True)  # save a .pckl file with results
result = gw.calculate()
