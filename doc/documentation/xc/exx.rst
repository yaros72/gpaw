.. _exx:

==============
Exact exchange
==============

**THIS PAGE IS PARTLY OUTDATED**

.. Can one define what is outdated?
.. Fractional occupations are fixed by now

Inclusion of the non-local Fock operator as an exchange-correclation
functional is an experimental feature in gpaw.

The current implementation *lacks* the following features:

* Support for periodic systems.
  Actually, the code won't complain, but the results have not been tested.
* Support for k-point sampling.
  No consideration has been made as to multiple k-points, or even comlplex
  wave functions, so this definitely won't work.
* Forces.
  Force evaluations when including (a fraction of -) the fock operator in
  the xc-functional has been implemented, but has not been tested.
* Speed.
  Inclusion of Fock exchange is exceedingly slow. The bottleneck is solving the
  poisson integrals of the Fock operator, which is currently done using an
  iterative real-space solver with a zero initial guess for the potential at
  each SCF cycle. This should be optimized.

One way to speed up an exact-exchange (or hybrid) calculation is to use the
coarse grid (used for wave functions) instead of the finegrid (used for for
densities) for the Fock potentials. This should give a speed-up factor of 8.
This can be specified in the ``xc`` keyword like in this example
:git:`~gpaw/test/exx/coarse.py`

Parallelization using domain decomposition is fully supported.

The Fock operator can be used to do the hybrid functional PBE0, and of course,
Hartree-Fock type EXX. These are accessed by setting the ``xc`` keyword to
``PBE0`` or ``EXX`` respectively.

The following functionals are suppported:

========== ======= =========
Functional Type    Reference
========== ======= =========
EXX        Global
PBE0       Global  [AB98]_
B3LYP      Global  [Ba94]_
HSE03      RSF-SR
HSE06      RSF-SR
CAMY-B3LYP RSF-LR  [SZ12]_
CAMY-BLYP  RSF-LR  [AT08]_
CAMY-B3LYP RSF-LR  [SZ12]_
LCY-BLYP   RSF-LR  [SZ12]_
LCY-PBE    RSF-LR  [SZ12]_
========== ======= =========

Here "Global" denotes global hybrids, which use the same percentage of
Hartree-Fock exchange for every point in space, while "RSF-SR" and "RSF-LR"
denotes range-separated functionals which mix the fraction of Hartree-Fock and
DFT exchange based on the spatial distance between two points, where for a
"RSF-SR" the amount of Hartree-Fock exchange decrease with the distance and
increase for a "RSF-LR". See :ref:`rsf` for more detailed information on
RSF(-LR).

A thesis on the implementation of EXX in the PAW framework, and the
specifics of the GPAW project can be seen on the :ref:`literature
<literature_reports_presentations_and_theses>` page.

A comparison of the atomization energies of the g2-1 test-set calculated in
VASP, Gaussian03, and GPAW is shown in the below two figures for the PBE and
the PBE0 functional respectively.

.. image:: g2test_pbe.png

.. image:: g2test_pbe0.png

In the last figure, the curve marked ``GPAW (nonself.)`` is a non-
selfconsistent PBE0 calculation using self-consistent PBE orbitals.

It should be noted, that the implementation lacks an optimized effective
potential. Therefore the unoccupied states utilizing EXX as implemented in
GPAW usually approximate (excited) electron affinities. Therefore calculations
utilizing Hartree-Fock exchange are usually a bad basis for the calculation of
optical excitations by lrTDDFT. As a remedy, the improved virtual orbitals
(IVOs, [HA71]_) were implemented. The requested excitation basis can be chosen
by the keyword ``excitation`` and the state by ``excited`` where the state is
counted from the HOMO downwards:

.. literalinclude:: ivo_hft.py

Support for IVOs in lrTDDFT is done along the work of Berman and Kaldor
[BK79]_.

If the number of bands in the calculation exceeds the number of bands delivered
by the datasets, GPAW initializes the missing bands randomly. Calculations utilizing
Hartree-Fock exchange can only use the RMM-DIIS eigensolver. Therefore the states
might not converge to the energetically lowest states. To circumvent this problem
on can made a calculation using a semi-local functional like PBE and uses this
wave-functions as a basis for the following calculation utilizing Hartree-Fock exchange
as shown in the following code snippet which uses PBE0 in conjuncture with
the IVOs:

.. literalinclude:: rsf_ivo_nacl.py

.. [AB98] C. Adamo and V. Barone.
   *Toward Chemical Accuracy in the Computation of NMR Shieldings: The PBE0
   Model.*.
   Chem. Phys. Lett. 298.1 (11. Dec. 1998), S. 113–119.

.. [Ba94] V. Barone.
   *Inclusion of Hartree–Fock exchange in density functional methods.
   Hyperfine structure of second row atoms and hydrides*.
   Jour. Chem. Phys. 101.8 (1994), S. 6834–6838.

.. [BK79] M. Berman and U. Kaldor.
   *Fast calculation of excited-state potentials for rare-gas
   diatomic molecules: Ne2 and Ar2*.
   Chem. Phys. 43.3 (1979), S. 375–383.

.. [HA71] S. Huzinaga and C. Arnau.
   *Virtual Orbitals in Hartree–Fock Theory. II*.
   Jour. Chem. Phys. 54.5 (1. Ma. 1971), S. 1948–1951.
