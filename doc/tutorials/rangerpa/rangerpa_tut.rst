.. _rangerpa_tut:

===================================================
Correlation energies within the range-separated RPA
===================================================

One of the less attractive features of calculating the electronic correlation energy within the random-phase approximation 
(RPA) is having to describe the `1/r` divergence of the Coulomb interaction.  Describing this divergence in a plane-wave
basis set requires in turn a large basis set for the response function `\chi^{KS}(\omega)`, and this soon becomes
very demanding on computational resources.

The scheme proposed in Ref. [#Bruneval]_ tries to avoid this problem by considering the RPA energy with an effective
Coulomb interaction `v^{LR}`, i.e.

.. math::

  E_c^{LR-RPA} = \int_0^{\infty}\frac{d\omega}{2\pi}\text{Tr}\Big[\text{ln}\{1-\chi^0(i\omega)v^{LR}\}+\chi^0(i\omega)v^{LR}\Big],

where:

.. math::

  v^{LR}(r) = \frac{\text{erf}(r/r_c)}{r}.

The error function `\text{erf}(x)` quickly goes to zero at the origin and tends to 1 at large `x`.  Thus  `v^{LR}`
is identical to the Coulomb interaction in the long-range (LR) limit, but goes smoothly to zero at small distance.
The transition between long and short-range (SR) behaviour is governed by the range-separation parameter `r_c`,
which is chosen by the user.  In the limit of very small `r_c` the full RPA is restored.

The remaining problem is how to restore the SR part of the Coulomb interaction.  The solution of Ref. [#Bruneval]_ is to
use a local-density approximation and the homogeneous electron gas, and write

.. math::

  E_c^{SR-RPA} = \int d\vec{r} \varepsilon_c^{SR}(n^v(\vec{r}),r_c)

where

.. math::

  \varepsilon_c^{SR}(n,r_c) = \varepsilon_c^{RPA}(n) - \varepsilon_c^{LR}(n,r_c).

The quantities `\varepsilon_c^{RPA}(n)` and `\varepsilon_c^{LR}(n,r_c)`
are the correlation energies (normalized to the appropriate number of electrons) 
of the homogeneous electron gas (HEG), calculated with the full Coulomb interaction
and with only the long-range part, respectively.  The total correlation energy
is evaluted as `E_c^{LR-RPA} + E_c^{SR-RPA}`.  Note that the quantity `n^v(\vec{r})`
is the density of valence electrons, i.e. only the electrons which are used to construct
`\chi^{KS}`.  In PAW language this quantity is the "all-electron valence density".

Of course it should be remembered that partitioning the correlation energy in this way
will only yield the exact RPA result either in the limit of vanishing `r_c` or
(trivially) for the HEG.  Ref. [#Bruneval]_ applies the scheme for a variety of systems.
Here we focus on one example and calculate the correlation energy of bulk Si as a function
of `r_c`, and compare it to the full RPA result.

Example 1: Correlation energy of silicon
========================================

The range-separated RPA calculations are performed in the same framework as
the RPA, so it may also be useful to consult the tutorials :ref:`rpa_tut`.
We start with a converged ground-state calculation to get the electronic wavefunctions:

.. literalinclude:: si.groundstate.py

This calculation will take about a minute on a single CPU.
Now we use the following script to get the RPA correlation in the range-separated
approach, using a number of different values for `r_c`.  For the values
of the plane-wave cutoff and number of bands used to evaluate `\chi^{KS}`, we
use the values reported in Ref. [#Bruneval]_:

.. literalinclude:: si.range_rpa.py

This script should take about 10 minutes when run on 8 CPUs.  If you look in one of
the output files, e.g. ``si_range.4.0.txt``, you should find the line

``Short range correlation energy/unit cell = -11.7496 eV``

which is `E_c^{SR-RPA}`.  The code then reports the total RPA energy
`E_c^{LR-RPA} + E_c^{SR-RPA}` at the end of the file.

Below we plot these numbers, and compare to the RPA energy calculated in the
standard approach (`r_c=0`).  The same plot is reported in [#Bruneval]_ (Fig. 1).
One can see that for `r_c<2`, there is pretty good agreement between
the range-separated and standard approaches.  The difference is, the range-separated
approach requires less computational firepower (e.g. a cutoff of 80 eV at `r_c=2`,
compared to 400 eV for the standard approach).

.. image:: Ec_rpa.png
           :height: 500 px

We end with the reminder that there is no such thing as a free lunch, and this
scheme requires careful testing on a system-by-system basis.  As its name suggests,
`r_c` is a parameter; larger values allow faster convergence, but reduced
accuracy.  Also, the construction
of the all-electron valence density on the grid can sometimes throw up problems,
so you are strongly advised to check that the ``Density integrates to XXX electrons``
line in the output file delivers the expected number of valence electrons.

.. [#Bruneval] F. Bruneval
              *Phys. Rev. Lett* **108**, 256403 (2012)
