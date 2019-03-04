.. module:: gpaw.response.bse
.. _bse tutorial:

========================================
The Bethe-Salpeter equation and Excitons
========================================

For a brief introduction to the Bethe-Salpeter equation and the details of its
implementation in GPAW, see :ref:`bse theory`.


Absorption spectrum of bulk silicon
=======================================
 
We start by calculating the ground state density and diagonalizing the resulting Hamiltonian.
Below we will set up the Bethe-Salpeter Hamiltonian in a basis of the 4 valence bands and 4 conduction bands. However, the screened interaction that enters the Hamiltonian needs to be converged with respect the number of unoccupied bands. The calculaton is erfored with the following script :download:`gs_Si.py`. It takes a few minutes on a single CPU. The last line in the script creates a .gpw file which contains all the informations of the system, including the wavefunctions.

Next we calculate the dynamical dielectric function using the Bethe-Salpeter equation. The imaginary part is proportional to the absorption spectrum. The calculation can be done with the script :download:`eps_Si.py`, which also calculates the dielectric function within the Random Phase Approximation (see :ref:`df_tutorial`). It takes about ~12 hours on a single CPU but parallelizes very well. Note the .csv output files that contains the spectre. The script also generates a .dat file that contains the eigenvalues of the BSE eigenvalues for easy application. The spectrum essentially consists of a number of peaks centered on the eigenvalues. It can be plottet with :download:`plot_Si.py` and the result is shown below

.. image:: bse_Si.png
    :height: 400 px

The ``write_v`` keyword ensures that the eigenstates and eigenvalues are stored in ``v_TS.gpw``. These are easily read by the get_dielectric_function method, such that if you want to calculate the spectrum at a different energy range or broadening you can simply do::

    bse.get_dielectric_function(filename='bse_0.1.csv',
                                readfile='v_TS.gpw',
                                eta=0.1,
                                w_w=np.linspace(0, 20, 10001))

This should finish almost instantaneously.

The parameters that needs to be converged in the calculation are the k-points in the initial ground state calculation. In addition the following keywords in the BSE object should be converged: the plane wave cutoff ``ecut``, the numbers of bands used to calculate the screened interaction ``nbands``, the list of valence bands ``valence_bands`` and the list of conduction bands ``conduction_bands`` included in the Hamiltonian. It is also possible to provide an array ``gw_skn``, with GW eigenvalues to be used in the non-interacting part of th Hamiltonian. Here, the indices denote spin, k-points and bands, which has to match the spin, k-point sampling and the number of specified valence and conduction bands in the ground state calculation.

For large calculations, it may be useful to write the screened interaction, which is the first quantity that is calculated and may be restarted in a subsequent calculation. This may be done with the keyword ``wfile='W_qGG.pckl'``, where the .pckl file contains the screened interaction matrices at all q-points and a few other variables. It may also be useful to set ``write_h=False`` and ``write_v=False``, since these files may become quite large for big calculations.

Excitons in monolayer MoS2 with Spin-orbit Coupling
===================================================

Spectrum from the Bethe-Salpeter equation
-----------------------------------------

The screening plays a fundamental role in the Bethe-Salpeter equation and for 2D systems the screening requires a special treatment. In particular we use a truncated Coulomb interaction inorder to decouple the screening between periodic images. We refer to Ref. [#Huser]_ for details on the truncated Coulomb interaction in GPAW. As before, we calculate the ground state of `MoS_2` with the script :download:`gs_MoS2.py`, which takes a few minutes. Note the large density of k-points, which are required to converge the BSE spectrum of two-dimensional systems.

The macroscopic dielectric function is calculated as an average of the microscopic screening over the unit cell. Clearly, for a 2D system this will depend on the unit cell size in the direction orthogonal to the slab and in the converged limit the dielectric function becomes unity. Instead we may calculate the longitudinal part of 2D polarizability which is independent of unit cell size. This is done in RPA as well as BSE with the scripts :download:`pol_MoS2.py`, which takes ~20 hours on 16 CPUs. Note that the BSE polarizability is calculated with and without Coulomb truncation for comparison. In both case spin-orbit coupling is included through the ``spinors`` keyword. We refer to Ref. [#Olsenspin]_ for details on the spin-orbit implementation. The results can be plottet with :download:`plot_MoS2.py` and is shown below.

.. image:: bse_MoS2.png
    :height: 400 px

The excitonic effects are much stronger than in the case of Si due to the reduced screening in 2D. In particular, we can identify a distinct spin-orbit split exciton well below the band edge. Note that without Coulomb truncation, the BSE spectrum is shifted upward in energy due the screening of electron-hole interactions from periodic images.

2D screening with and without Coulomb truncation
------------------------------------------------

To see the effect of the Coulomb truncation, which eliminates the screening from layers in periodic images, we will now calculate the dielectric constant evaluated at the center of the layer `z_0` and averaged in the plane. This is accomplished with

.. math:: \epsilon_{2D}^{-1}(\mathbf{q})=\sum_{\mathbf{G}|G_{\parallel=0}}e^{iG_zz_0}\epsilon_{\mathbf{G}\mathbf{0}}^{-1}(\mathbf{q})

The script :download:`get_2d_eps.py` carries out this calculations with and without Coulomb truncation and the result is shown below :download:`plot_2d_eps.py`. Note that the truncated screening is bound to become one at `\Gamma` due to the different behavior of Coulomb interaction (in `q`-space) in 2D systems. For small values of `q` the screening is linear, which makes convergence tricky in standard Brillouin zone sampling schemes. Since the `\Gamma`-point is always sampled, the screening is typically underestimated and the exciton binding energy is too high at finite `k`-point samplings.

.. image:: 2d_eps.png
    :height: 400 px

Mott-Wannier model for excitons in 2D materials
-----------------------------------------------
 
In 3D materials the Mott-Wannier model of excitons has been highly succesful and simply regards the exciton as a "hydrogen atom" with bindings energies that has been rescaled by the exciton effective mass and dielectric screening. Thus in atomic units the binding energy is

.. math:: E_B^{3D}=\frac{\mu}{2\epsilon_0^2}

where `\mu^{-1}=m_v^{-1}+m_c^{-1}` and `m_v` and `m_c` are the masses of valence and conduction electrons respectively. The 3D expression relies on the fact that the screening is local in real space and thus approximately independent of `q`. This is clearly not the case in 2D where we always have

.. math:: \epsilon_{2D}(\mathbf{q})=1+2\pi\alpha|\mathbf{q}|

for small `q`. It is thus expected that the hydrogenic binding energy in 2D becomes renormalized by the slope `\alpha` in addition to the effective mass. Indeed in Ref. [#Olsen]_ it was shown that the binding energy in 2D can be approximated by

.. math:: E_B^{2D}=\frac{8\mu}{(1+\sqrt{1+32\pi\alpha\mu/3})^2}

From the band structure of MoS2 it is straigtforward to obtain `\mu=0.27` and all we need now is `\alpha`. In principle we could read of the slope from the figure above, but there is a more direct an accurate way to do it. As it turns out, the slope is needed for any calculation of the response function in the optical limit and it is simply obtained with the script :download:`alpha_MoS2.py`. This runs on a single CPU in a minute or so. It should produce a value of `\alpha=5.27` Å. Transforming to atomic units and inserting into the formula above yields

.. math:: E_B^{MoS_2}=0.50\; eV,

which is in good agreement with the BSE computation above

.. [#Huser] F. Huser, T. Olsen and K. S. Thygesen
            *Phys. Rev. B* **88**, 245309 (2013)

.. [#Olsenspin] T. Olsen
            *Phys. Rev. B.* **94**, 235106 (2016)

.. [#Olsen] T. Olsen, S. Latini, F. Rasmussen and K. S. Thygesen
            *Phys. Rev. Lett.* **116**, 056401 (2016)

