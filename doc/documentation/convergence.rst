.. _convergence:

==================
Convergence Issues
==================

*Try to use default parameters for the calculator. Simple and often useful.*

Here you find a list of suggestions that should be considered when
encountering convergence problems:

1) Make sure the geometry and spin-state is physically sound.

   Remember that ASE uses Angstrom and not Bohr or nm!
   For spin polarized systems, make sure you have sensible initial magnetic
   moments. Don't do spin-paired calculations for molecules with an odd
   number of electrons. Before performing calculations of isolated atoms
   see :ref:`atomization_energy`.

2) Use less aggressive density mixing.

   Try something like ``mixer=Mixer(0.02, 5, 100)`` or
   ``mixer=MixerSum(0.02, 5, 100)``, ``mixer=MixerDif(0.02, 5, 100)``
   for spin-polarized calculations and remember to import the mixer classes::

       from gpaw import Mixer, MixerSum, MixerDif

   For some systems (for example transition metal atoms) it is helpful to
   reduce the number of history steps in the mixer to ``1`` (instead of ``5``).

3) Solve the eigenvalue problem more accurately at each scf-step.

   Import the Davidson eigensolver::

       from gpaw import Davidson

   and increase the number iterations per scf-step ``eigensolver=Davidson(3)``.

   CG eigensolver tends converge fastest the unoccupied bands
   ``eigensolver='cg'``.

4) Use a smoother distribution function for the occupation numbers.

   Remember that for systems without periodic boundary conditions
   (molecules) the Fermi temperature is set to zero by default.
   You might want to specify a finite Fermi temperature as described
   :ref:`here <manual_occ>` and check the convergence of
   the results with respect to the temperature!

5) Try adding more empty states.

   If you are specifying the :ref:`number of bands <manual_nbands>`
   manually, try to increase the number of empty states. You might also
   let GPAW choose the default number, which is in general large enough.

6) Use enough k-points.

   Try something like ``kpts={'density': 3.5, 'even': True}``
   (see :ref:`manual_kpts`).

7) Don't let your structure optimization algorithm take too large steps.

8) Better initial guess for the wave functions.

   The initial guess for the wave functions is always calculated
   using the LCAO scheme, with a default single-zeta basis, i.e. one
   orbital for each valence electron.
   It is possible to use ``basis='szp(dzp)'`` to extract
   the single-zeta polarization basis set from the double-zeta
   polarization basis sets that are distributed together with
   the latest PAW datasets. You can also try to make a better initial guess
   by enlarging the :ref:`manual_basis`. Note that you first need to generate
   the basis file, as described in :ref:`LCAO mode <lcao>`.

   Warning: this may in some cases worsen the convergence, and improves
   it usually only when the number of empty states is significantly increased.
