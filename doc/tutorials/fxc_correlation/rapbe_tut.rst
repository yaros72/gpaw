.. _rapbe_tut:

===============================
Correlation energies from TDDFT
===============================

The Random Phase Approximation (RPA) for correlation energies comprises a nice non-empirical expression 
for the correlation energy that can be naturally combined with exact exchange to calculate binding energies. 
Due to the non-local nature of the approximation, RPA gives a good account of dispersive forces and is the 
only xc approximation capable of describing intricate binding regimes where covalent and van der Waals 
interactions are equally important. However, RPA does not provide a very accurate description of strong 
covalent bonds and typically performs worse than standard GGA functionals for molecular atomization 
energies and cohesive energies of solids.

In general, the exact correlation energy can be expressed in terms of the exact response function as 

.. math::

  E_c = -\int_0^{\infty}\frac{d\omega}{2\pi}\int_0^1d\lambda\text{Tr}\Big[v\chi^\lambda(\omega)-v\chi^{KS}(\omega)\Big],

and the RPA approximation for correlation energies is simply obtained from the RPA approximation for the response function. 
From the point of view of TDDFT, the response function can be expressed exactly in terms of the Kohn-Sham response 
function and the exchange-correlation kernel `f_{xc}`:

.. math::

  \chi^\lambda(\omega) = \chi^{KS}(\omega) + \chi^{KS}(\omega)\Big[\lambda v+f_{xc}^\lambda(\omega)\Big]\chi^\lambda(\omega).

The RPA is obtained by neglecting the exchange-correlation kernel and it should be possible to improve RPA by including 
simple approximations for this kernel. However, if one tries to use a simple adiabatic kernel, one encounters severe 
convergence problems and results become worse than RPA! The reason for this is that the locality of adiabatic kernels 
renders the pair-correlation function divergent. As it turns out, the adiabatic correlation hole can be renormalized 
by a simple non-empirical procedure, which results in a density-dependent non-locality in the kernel. This can be done 
for any adiabatic kernel and the method has implemented for LDA and PBE. We refer to these approximations as renormalized 
adiabatic LDA (rALDA) and renormalized adiabatic PBE (rAPBE). We only include the exchange part of the kernel, since this 
part is linear in `\lambda` and the kernel thus only needs to be evaluated for `\lambda=1`.

For more details on the theory and implementation of RPA we refer to :ref:`rpa` and the tutorial :ref:`rpa_tut`. 
The RPA tutorial should be studied before the present tutorial, which inherits much of the terminology from RPA. 
Details on the theory, implementation and benchmarking of the renormalized kernels can be found in Refs. [#Olsen1]_, [#Olsen2]_, and [#Olsen3]_.

Below we give examples on how to calculate the correlation energy of a Hydrogen atom as well as the rAPBE atomization energy 
of a `CO` molecule and the rAPBE cohesive energy of diamond. 
Note that some of the calculations in this tutorial will need a lot of CPU time and are essentially not possible without a supercomputer.

Finally, we note that there is some freedom in deciding how to include the density dependence in the kernel.
By default the kernel is constructed from a two-point average of the density.
However as shown in Example 4 it is possible to instead use a reciprocal-space averaging procedure.  
Within this averaging scheme it is possible to explore different
approximations for `f_{xc}`, for instance a simple dynamical kernel, or a jellium-with-gap model, which displays
a `1/q^2` divergence for small `q`.  More details can be found below and in [#Patrick]_.

Example 1: Correlation energy of the Hydrogen atom
==================================================

As a first demonstration of the deficiencies of RPA, we calculate the correlation energy of a Hydrogen atom. 
The exact correlation energy should vanish for any one-electron system. The calculations can be performed with the following scripts,
starting with a standard DFT-LDA calculation:

.. literalinclude:: H.ralda_01_lda.py

followed by an RPA calculation:

.. literalinclude:: H.ralda_02_rpa_at_lda.py

and finally one using the rALDA kernel:

.. literalinclude:: H.ralda_03_ralda.py

The analogous set of scripts for PBE/rAPBE are :download:`H.ralda_04_pbe.py`, :download:`H.ralda_05_rpa_at_pbe.py`
and :download:`H.ralda_06_rapbe.py`.
The computationally-heavy RPA/rALDA/rAPBE parts can be parallelized efficiently on multiple CPUs. 
After running the scripts the LDA and PBE correlation energies may be found in the file ``H.ralda.DFT_corr_energies.txt``.
Note that a rather small unit cell is used and the results may not be completely converged with respect 
to cutoff and unit cell. Also note that the correlation energy is calculated at different cutoff energies up to 
300 eV and the values based on two-point extrapolation is printed at the end (see :ref:`rpa_tut` and :ref:`rpa` for a 
discussion on extrapolation). The results in eV are summarized below.

=====   =======   ======
 LDA    RPA       rALDA 
=====   =======   ======
-0.56    -0.55    -0.029
=====   =======   ======

=====   =======    ======
PBE     RPA        rAPBE
=====   =======    ======
-0.16    -0.55     -0.007
=====   =======    ======

The fact that RPA gives such a dramatic underestimation of the correlation energy is a general problem with the method, 
which is also seen for bulk systems. For example, for the homogeneous electron gas RPA underestimates the correlation energy 
by ~0.5 eV per electron for a wide range of densities. 
 
Example 2: Atomization energy of CO
===================================

Although RPA severely underestimates absolute correlation energies in general, energy differences are often of decent quality 
due to extended error cancellation. Nevertheless, RPA tends to underbind and performs slightly worse than PBE for atomization 
energies of molecules. The following example shows that rAPBE not only corrects the absolute correlation energies, but also 
performs better than RPA for atomization energies.

First we set up a ground state calculation with lots of unoccupied bands. This is done with the script:

.. literalinclude:: CO.ralda_01_pbe+exx.py

which takes on the order of 6-7 CPU hours. The script generates three gpw files containing the wavefunctions,
which are the input to the rAPBE calculation. The PBE and non-selfconsistent Hartree-Fock atomization energies 
are also calculated and written to the file ``CO.ralda.PBE_HF_CO.dat``. 
Next we calculate the RPA and rAPBE energies for CO with the script

.. literalinclude:: CO.ralda_02_CO_rapbe.py

The energies for C and O are obtained from the corresponding scripts
:download:`CO.ralda_03_C_rapbe.py` and :download:`CO.ralda_04_O_rapbe.py`.
The results for various cutoffs are written to the files like ``CO.ralda_rpa_CO.dat``
and ``CO.ralda_rapbe_CO.dat``.
We also print the correlation energies of the C atom to be used in a tutorial below. 
As in the case of RPA the converged result is obtained by extrapolation using the script 

.. literalinclude:: CO.ralda_05_extrapolate.py

If pylab is installed, the plot=False can be change to plot=True to visualize the quality of the extrapolation. The final results are displayed below

======   =====   =====   ======       ============
PBE      HF      RPA     rAPBE        Experimental
======   =====   =====   ======       ============
11.71    7.36    10.60    11.31         11.23
======   =====   =====   ======       ============

Example 3: Cohesive energy of diamond
=====================================
The error cancellation in RPA works best when comparing systems with similar electronic structure. 
In the case of cohesive energies of solids where the bulk energy is compared to the energy of isolated atoms, 
RPA becomes even worse than for atomization energies of molecules. Here we illustrate this for the cohesive 
energy of diamond and show that the rAPBE approximation corrects the problem. 
The initial orbitals are obtained with the script

.. literalinclude:: diamond.ralda_01_pbe.py

which takes roughly 5 minutes. The script generates ``diamond.ralda.pbe_wfcs.gpw``
and uses a previous calculation of the C atom to calculate the EXX and PBE cohesive 
energies that are written to ``diamond.ralda.PBE_HF_diamond.dat``. The RPA and rAPBE 
correlation energies are obtained with the script:

.. literalinclude:: diamond.ralda_02_rapbe_rpa.py

This takes on the order of 30 CPU hours, but can be parallelized efficiently. Finally the 
correlation part of the cohesive energies are obtained by extrapolation with the script

.. literalinclude:: diamond.ralda_03_extrapolate.py

The results are summarized below

====   ====   ====   ======       ============
PBE     HF     RPA    rAPBE       Experimental
====   ====   ====   ======       ============
7.75   5.17   7.04     7.61             7.55
====   ====   ====   ======       ============

As anticipated, RPA severely underestimates the cohesive energy, while PBE performs much better, and rAPBE comes very close to the experimental value.


Example 4: Correlation energy of diamond with different kernels
===============================================================
Finally we provide an example where we try different exchange-correlation kernels.
For illustrative purposes we use a basic computational setup - as a result these numbers
should not be considered converged!
As usual we start with a ground state calculation to get the electron density and wavefunctions
for diamond:

.. literalinclude:: diam_kern.ralda_01_lda.py

The default method of constructing the kernel is to use a two-point density average.
Therefore the following simple script gets the rALDA correlation energy within this averaging scheme:

.. literalinclude:: diam_kern.ralda_02_ralda_dens.py

However, an alternative method of constructing the kernel is to work in reciprocal space,
and average over wavevectors rather than density.
To use this averaging scheme, we add the flag ``av_scheme='wavevector'``:

.. literalinclude:: diam_kern.ralda_03_ralda_wave.py

Using this averaging scheme opens a few more possible choices for the kernel.
For example, we can include the correlation part of the ALDA which is left out of the rALDA
by setting ``xc='rALDAc'``:

.. literalinclude:: diam_kern.ralda_04_raldac.py

Alternatively, we can look at more exotic kernels, such as a simplified version of the
jellium-with-gap kernel of Ref. [#Trevisanutto]_ (JGMs).
This kernel diverges as `1/q^2` for small `q`, with the strength of the divergence
depending on the size of the band gap.
To use this kernel, the gap must be specified as ``Eg=X``, where X is in eV:

.. literalinclude:: diam_kern.ralda_05_jgm.py

Another interesting avenue is the simple dynamical kernel of Constantin and Pitarke (CP_dyn) [#Constantin]_:

.. literalinclude:: diam_kern.ralda_06_CP_dyn.py

Finally, for the enthusiast there is a basic implementation of the range-separated
RPA approach of Ref. [#Bruneval]_.  By separating the Coulomb interaction into long
and short-range parts and taking the short range part from the electron gas, one can
dramatically reduce the number of plane waves needed to converge the RPA energy.
In this approach it is necessary to specify a range-separation parameter ``range_rc=Y``, where
Y is in Bohr.  It is important to bear in mind that this feature is relatively untested.

.. literalinclude:: diam_kern.ralda_07_range_rpa.py

For comparison, one can see that the RPA converges much more slowly:

.. literalinclude:: diam_kern.ralda_08_rpa.py

Here we summarize the above calculations and show the correlation energy/electron (in eV), 
obtained at an (unconverged) cutoff of 131 eV:

=================  ================   ======  ======  ======  ===================  ======
rALDA (dens. av.)  rALDA (wave. av)   rALDAc  JGMs    CP_dyn  range separated RPA   RPA  
=================  ================   ======  ======  ======  ===================  ======
-1.161              -1.134            -1.127  -1.134  -1.069        -1.730         -1.396
=================  ================   ======  ======  ======  ===================  ======

Incidentally, a fully-converged RPA calculation gives a correlation energy 
of -1.781 eV per electron.

We conclude with some practical points.  The wavevector-averaging scheme is less intuitive
than the density-average, but avoids some difficulties such as having to describe the `1/r`
divergence of the Coulomb interaction in real space.  It also provides a natural framework
to construct the JGMs kernel, and can be faster to construct for systems with many k points.
However it is also worth remembering that kernels which scale linearly in the coupling constant
(e.g rALDA) need only be constructed once per k point.  Those that do not scale linearly (e.g. rALDAc) 
need to be constructed `N_\lambda` times, and the CP_dyn kernel must be constructed
at each frequency point as well, i.e. `N_\lambda N_\omega` times.  Assuming standard values
of 8 and 16 for `N_\lambda` and `N_\omega` means there is a factor 100 cost
in constructing and storing a dynamical kernel compared to rALDA.  Finally we point out that 
the rALDA and rAPBE kernels are also special because they have explicit spin-polarized forms.

.. [#Olsen1] T. Olsen and K. S. Thygesen
              *Phys. Rev. B* **86**, 081103(R) (2012)

.. [#Olsen2] T. Olsen and K. S. Thygesen
              *Phys. Rev. B* **88**, 115131 (2013)

.. [#Olsen3] T. Olsen and K. S. Thygesen
              *Phys. Rev. Lett* **112**, 203001 (2014)

.. [#Patrick] C. E. Patrick and K. S. Thygesen
              *J. Chem. Phys.* **143**, 102802 (2015)

.. [#Trevisanutto] P. E. Trevisanutto et al.
              *Phys. Rev. B* **87**, 205143 (2013)

.. [#Constantin] L. A. Constantin and J. M. Pitarke
              *Phys. Rev. B* **75**, 245127 (2007)

.. [#Bruneval] F. Bruneval
              *Phys. Rev. Lett* **108**, 256403 (2012)
