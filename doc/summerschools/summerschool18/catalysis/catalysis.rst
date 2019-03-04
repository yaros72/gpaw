.. _catalysis:

=========
Catalysis
=========

This exercise studies the splitting of the |N2| molecule on a Ruthenium
surface.   |N2| splitting is the critical step in ammonia synthesis, which is
the main source of biologically accessible nitrogen for fertilizers.

Note that the |N2| splitting occurs most readily at the bottom of a step on
the close-packed (0001) surface.  However, to keep system sizes and computer
time down at a manageable level, we shall look at a flat surface.

Tools used in this exercise:

* Structural energy minimization.

* Nudged Elastic Band (NEB) for finding transition states.

* If you have time: Extra exercise on vibrational energy


Part 1: |N2| adsorption on a flat Ru surface
============================================

:download:`n2_on_metal.ipynb`, :download:`N2Ru_hollow.png`,
:download:`2NadsRu.png`

The notebook ``n2_on_metal.ipynb`` shows how to set up a molecule on a flat
metal surface.

* Set up a clean metal surface.

* Relax the topmost layer (ca. 10 min running time).

  - While running: study the gpaw text output, to learn about e.g. number of
    irreducible k-points (important for parallel simulations).

* Set up and relax a single |N2| molecule (ca. 1 min running time).

* Add the molecule standing on the metal on an on-top site.

* Relax the combined system.


Part 2: Splitting |N2|: initial and final geometry
==================================================

(Begin this while the last step above runs)

The |N2| molecule will not split while standing up on an on-top site.  The
molecule can also bind to the surface in a flat geometry - we here ignore the
barrier between the two states and just use the lying-down molecule as the
initial configuration.

Create scripts setting up and energy-minimizing the initial and final
structures, as described in the final part of the notebook from Part 1.
Submit these scripts as parallel batch jobs.  When submitting make sure that
the number of CPU cores matches the number of irreducible k-point in the
calculation, as k-point parallelization is much more efficient than other
forms of parallelization.

See :ref:`gbar submitting`.


Part 3: Learning about Nudged Elastic Band
==========================================

:download:`neb.ipynb`

While the calculations from the previous step runs, you can learn about the
Nudged Elastic Band method for finding transition states and barriers from the
notebook ``neb.ipynb``.


Part 4: Run a parallel NEB calculation
======================================

Prepare a script running NEB using the GPAW calculator and the initial and
final states from part 2 to find the barrier for |N2| dissociation.

When doing this you should parallelize over the images in the NEB
calculation. A more detailed description of how to do this can be found in
the *Exercise* part of the ``neb.ipynb`` along with some suitable input
parameters for the NEB.


Extra exercise: Vibrational energy
======================================

:download:`vibrations.ipynb`, :download:`TS.xyz`

The notebook ``vibrations.ipynb`` will guide you through how to calculate the
vibrations of the adsorbate in the inital and final state and use the
Thermochamistry module to calculate the reaction free energy. The final part
of the exercise shows what happens when you calculate the vibrations of a
well-converged NEB transition state.


Extra material: Convergence test
================================

:download:`convergence.ipynb`, :download:`convergence.db`,
:download:`check_convergence.py`

We look at the adsorption energy and height of a nitrogen atom on a Ru(0001)
surface in the hcp site.  We check for convergence with respect to:

* number of layers
* number of k-points in the BZ
* plane-wave cutoff energy


.. |N2| replace:: N\ :sub:`2`
