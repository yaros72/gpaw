# Creates: W_r.svg
import numpy as np
import matplotlib.pyplot as plt
from gpaw.response.qeh import Heterostructure


thick_MoS2 = 6.2926
thick_WSe2 = 6.718

d_MoS2_WSe2 = (thick_MoS2 + thick_WSe2) / 2
inter_mass = 0.244

HS = Heterostructure(structure=['MoS2_int', 'WSe2_int'],
                     d=[d_MoS2_WSe2],
                     qmax=None,
                     wmax=0,
                     d0=thick_WSe2)

hl_array = np.array([0., 0., 1., 0.])
el_array = np.array([1., 0., 0., 0.])


# Getting the interlayer exciton screened interaction on a real grid
r, W_r = HS.get_exciton_screened_potential_r(
    r_array=np.linspace(1e-1, 30, 1000),
    e_distr=el_array,
    h_distr=hl_array)

plt.plot(r, W_r, '-g')
plt.title(r'Screened Interaction Energy')
plt.xlabel(r'${\bf r}$ (Ang)', fontsize=20)
plt.ylabel(r'$W({\bf r})$ (Ha)', fontsize=20)
plt.savefig('W_r.svg')
plt.show()

ee, ev = HS.get_exciton_binding_energies(eff_mass=inter_mass,
                                         e_distr=el_array,
                                         h_distr=hl_array)

print('The interlayer exciton binding energy is:', -ee[0])

