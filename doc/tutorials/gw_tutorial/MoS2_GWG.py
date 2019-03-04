from gpaw.response.g0w0 import G0W0

ecut = 40
gw = G0W0(calc='MoS2_fulldiag.gpw',
          xc='rALDA',
          av_scheme='wavevector',
          fxc_mode='GWG',
          bands=(8, 18),
          ecut=ecut,
          truncation='2D',
          nblocksmax=True,
          q0_correction=True,
          filename='MoS2_g0w0g_%s' %ecut,
          savepckl=True)

gw.calculate()

