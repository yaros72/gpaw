.. _releasenotes:

=============
Release notes
=============


Git master branch
=================

:git:`master <>`.

* Corresponding ASE release: ASE-3.17.1b1

* Berry phases can now be calculated.  See the :ref:`berry tutorial` tutorial
  for how to use it to calculate spontaneous polarization, Born effective
  charges and other physical properties.

* How to do :ref:`ehrenfest` has now been documented.


Version 1.5.1
=============

23 Jan 2019: :git:`1.5.1 <../1.5.1>`

* Corresponding ASE release: ASE-3.17.0.

* Small bug fixes related to latest versions of Python, Numpy and Libxc.


Version 1.5.0
=============

11 Jan 2019: :git:`1.5.0 <../1.5.0>`

* Corresponding ASE release: ASE-3.17.0.

* Last release to support Python 2.7.

* The default finite-difference stencils used for gradients in GGA and MGGA
  calculations have been changed.

  * The range of the stencil has been increased
    from 1 to 2 thereby decreasing the error from `O(h^2)` to `O(h^4)`
    (where `h` is the grid spacing).  Use ``xc={'name': 'PBE', 'stencil': 1}``
    to get the old, less accurate, stencil.

  * The stencils are now symmetric also for non-orthorhombic
    unit cells.  Before, the stencils would only have weight on the
    nighboring grid-points in the 6 directions along the lattice vectors.
    Now, grid-points along all nearest neighbor directions can have a weight
    in the  stencils.  This allows for creating stencils that have all the
    crystal symmetries.

* PW-mode calculations can now be parallelized over plane-wave coefficients.

* The PW-mode code is now much faster.  The "hot spots" have been moved
  from Python to C-code.

* Wavefunctions are now updated when the atomic positions change by
  default, improving the initial wavefunctions across geometry steps.
  Corresponds to ``GPAW(experimental={'reuse_wfs_method': 'paw'})``.
  To get the old behaviour, set the option to ``'keep'`` instead.
  The option is disabled for TDDFT/Ehrenfest.

* Add interface to Elpa eigensolver for LCAO mode.
  Using Elpa is strongly recommended for large calculations.
  Use::

      GPAW(mode='lcao',
           basis='dzp',
           parallel={'sl_auto': True, 'use_elpa': True})

  See also documentation on the :ref:`parallel keyword <manual_parallel>`.

* Default eigensolver is now ``Davidson(niter=2)``.

* Default number of bands is now `1.2 \times N_{\text{occ}} + 4`, where
  `N_{\text{occ}}` is the number of occupied bands.

* Solvated jellium method has been implemented, see
  :ref:`the documentation <solvated_jellium_method>`.

* Added FastPoissonSolver which is faster and works well for any cell.
  This replaces the old Poisson solver as default Poisson solver.

* :ref:`rsf` and improved virtual orbitals, the latter from Hartree-Fock
  theory.

* New Jupyter notebooks added for teaching DFT and many-body methods.  Topics
  cover: :ref:`catalysis`, :ref:`magnetism`, :ref:`machinelearning`,
  :ref:`photovoltaics`, :ref:`batteries` and :ref:`intro`.

* New experimental local **k**-point refinement feature:
  :git:`gpaw/test/kpt_refine.py`.

* A module and tutorial have been added for calculating electrostatic
  corrections to DFT total energies for charged systems involving localised
  defects: :ref:`defects`.

* Default for FFTW planning has been changed from ``ESTIMATE`` to ``MEASURE``.
  See :class:`gpaw.wavefunctions.pw.PW`.


Version 1.4.0
=============

29 May 2018: :git:`1.4.0 <../1.4.0>`

* Corresponding ASE release: ASE-3.16.0.

* Improved parallelization of operations with localized functions in
  PW mode.  This solves the current size bottleneck in PW mode.

* Added QNA XC functional.

* Major refactoring of the LCAOTDDFT code and added Kohn--Sham decomposition
  analysis within LCAOTDDFT, see :ref:`the documentation <lcaotddft>`.

* New ``experimental`` keyword, ``GPAW(experimental={...})`` to enable
  features that are still being tested.

* Experimental support for calculations with non-collinear spins
  (plane-wave mode only).
  Use ``GPAW(experimental={'magmoms': magmoms})``, where ``magmoms``
  is an array of magnetic moment vectors of shape ``(len(atoms), 3)``.

* Number of bands no longer needs to be divisible by band parallelization
  group size.  Number of bands will no longer be automatically adjusted
  to fit parallelization.

* Major code refactoring to facilitate work with parallel arrays.  See new
  module: :mod:`gpaw.matrix`.

* Better reuse of wavefunctions when atoms are displaced.  This can
  improve performance of optimizations and dynamics in FD and PW mode.
  Use ``GPAW(experimental={'reuse_wfs_method': name})`` where name is
  ``'paw'`` or ``'lcao'``.  This will move the projections of the
  wavefunctions upon the PAW projectors or LCAO basis set along with
  the atoms.  The latter is best when used with ``dzp``.
  This feature has no effect for LCAO mode where the basis functions
  automatically follow the atoms.

* Broadcast imports (Python3 only): Master process broadcasts most module
  files at import time to reduce file system overhead in parallel
  calculations.

* Command-line arguments for BLACS/ScaLAPACK
  have been
  removed in favour of the :ref:`parallel keyword
  <manual_parallelization_types>`.  For example instead of running
  ``gpaw-python --sl_diagonalize=4,4,64``, set the parallelization
  within the script using
  ``GPAW(parallel={'sl_diagonalize': (4, 4, 64)})``.

* When run through the ordinary Python interpreter, GPAW will now only
  intercept and use command-line options of the form ``--gpaw
  key1=value1,key2=value2,...`` or ``--gpaw=key1=value1,key2=value2,...``.

* ``gpaw-python`` now takes :ref:`command line options` directly
  instead of stealing them from ``sys.argv``, passing the remaining
  ones to the script:
  Example: ``gpaw-python --gpaw=debug=True myscript.py myscript_arguments``.
  See also ``gpaw-python --help``.

* Two new parameters for specifying the Pulay stress. Directly like this::

      GPAW(mode=PW(ecut, pulay_stress=...), ...)

  or indirectly::

      GPAW(mode=PW(ecut, dedecut=...), ...)

  via the formula `\sigma_P=(2/3)E_{\text{cut}}dE/dE_{\text{cut}}/V`.  Use
  ``dedecut='estimate'`` to use an estimate from the kinetic energy of an
  isolated atom.

* New utility function: :func:`gpaw.utilities.ibz2bz.ibz2bz`.


Version 1.3.0
=============

2 October 2017: :git:`1.3.0 <../1.3.0>`

* Corresponding ASE release: ASE-3.15.0.

* :ref:`command line options` ``--dry-run`` and ``--debug`` have been removed.
  Please use ``--gpaw dry-run=N`` and ``--gpaw debug=True`` instead
  (or ``--gpaw dry-run=N,debug=True`` for both).

* The :meth:`ase.Atoms.get_magnetic_moments` method will no longer be
  scaled to sum up to the total magnetic moment.  Instead, the magnetic
  moments integrated inside the atomic PAW spheres will be returned.

* New *sbatch* sub-command for GPAW's :ref:`cli`.

* Support added for ASE's new *band-structure* :ref:`ase:cli`::

  $ ase band-structure xxx.gpw -p GKLM

* Added :ref:`tetrahedron method <tetrahedron>` for calculation the density
  response function.

* Long-range cutoff for :mod:`~ase.calculators.qmmm` calculations can now be
  per molecule instead of only per point charge.

* Python 2.6 no longer supported.

* There is now a web-page documenting the use of the in development version
  of GPAW: https://wiki.fysik.dtu.dk/gpaw/dev/.

* :ref:`BSE <bse tutorial>` calculations for spin-polarized systems.

* Calculation of :ref:`magnetic anisotropy <magnetic anisotropy>`.

* Calculation of vectorial magnetic moments inside PAW spheres based on
  spin-orbit spinors.

* Added a simple :func:`gpaw.occupations.occupation_numbers` function for
  calculating occupation numbers, fermi-level, magnetic moment, and entropy
  from eigenvalues and k-point weights.

* Deprecated calculator-keyword ``dtype``.  If you need to force the datatype
  of the wave functions to be complex, then use something like::

      calc = GPAW(mode=PW(ecut=500, force_complex_dtype=True))

* Norm-conserving potentials (HGH and SG15) now subtract the Hartree
  energies of the compensation charges.
  The total energy of an isolated pseudoatom stripped of all valence electrons
  will now be zero.

* HGH and SG15 pseudopotentials are now Fourier-filtered at runtime
  as appropriate for the given grid spacing.  Using them now requires scipy.

* The ``gpaw dos`` sub-command of the :ref:`cli` can now show projected DOS.
  Also, one can now use linear tetrahedron interpolation for the calculation
  of the (P)DOS.

* The :class:`gpaw.utilities.ps2ae.PS2AE` tool can now also calculate the
  all-electron electrostatic potential.


Version 1.2.0
=============

7 February 2017: :git:`1.2.0 <../1.2.0>`.

* Corresponding ASE release: ASE-3.13.0.

* New file-format for gpw-files.  Reading of old files should still work.
  Look inside the new files with::

      $ python3 -m ase.io.ulm abc.gpw

* Simple syntax for specifying BZ paths introduced:
  ``kpts={'path': 'GXK', 'npoints': 50}``.

* Calculations with ``fixdensity=True`` no longer update the Fermi level.

* The GPAW calculator object has a new
  :meth:`~ase.calculators.calculator.Calculator.band_structure`
  method that returns an :class:`ase.dft.band_structure.BandStructure`
  object.  This makes it very easy to create band-structure plots as shown
  in section 9 of this awesome Psi-k *Scientfic Highlight Of The Month*:
  http://psi-k.net/download/highlights/Highlight_134.pdf.

* Dipole-layer corrections for slab calculations can now be done in PW-mode
  also.  See :ref:`dipole`.

* New :meth:`~gpaw.paw.PAW.get_electrostatic_potential` method.

* When setting the default PAW-datasets or basis-sets using a dict, we
  must now use ``'default'`` as the key instead of ``None``:

  >>> calc = GPAW(basis={'default': 'dzp', 'H': 'sz(dzp)'})

  and not:

  >>> calc = GPAW(basis={None: 'dzp', 'H': 'sz(dzp)'})

  (will still work, but you will get a warning).

* New feature added to the GW code to be used with 2D systems. This lowers
  the required k-point grid necessary for convergence. See this tutorial
  :ref:`gw-2D`.

* It is now possible to carry out GW calculations with eigenvalue self-
  consistency in G. See this tutorial :ref:`gw-GW0`.

* XC objects can now be specified as dictionaries, allowing GGAs and MGGAs
  with custom stencils: ``GPAW(xc={'name': 'PBE', 'stencil': 2})``

* Support for spin-polarized vdW-DF functionals (svdW-DF) with libvdwxc.


Version 1.1.0
=============

22 June 2016: :git:`1.1.0 <../1.1.0>`.

* Corresponding ASE release: ASE-3.11.0.

* There was a **BUG** in the recently added spin-orbit module.  Should now
  be fixed.

* The default Davidson eigensolver can now parallelize over bands.

* There is a new PAW-dataset file available:
  :ref:`gpaw-setup-0.9.20000.tar.gz <datasets>`.
  It's identical to the previous
  one except for one new data-file which is needed for doing vdW-DF
  calculations with Python 3.

* Jellium calculations can now be done in plane-wave mode and there is a new
  ``background_charge`` keyword (see the :ref:`Jellium tutorial <jellium>`).

* New band structure unfolding tool and :ref:`tutorial <unfolding tutorial>`.

* The :meth:`~gpaw.calculator.GPAW.get_pseudo_wave_function` method
  has a new keyword:  Use ``periodic=True`` to get the periodic part of the
  wave function.

* New tool for interpolating the pseudo wave functions to a fine real-space
  grids and for adding PAW-corrections in order to obtain all-electron wave
  functions.  See this tutorial: :ref:`ps2ae`.

* New and improved dataset pages (see :ref:`periodic table`).  Now shows
  convergence of absolute and relative energies with respect to plane-wave
  cut-off.

* :ref:`wannier90 interface`.

* Updated MacOSX installation guide for :ref:`homebrew` users.

* topological index


Version 1.0.0
=============

17 March 2016: :git:`1.0.0 <../1.0.0>`.

* Corresponding ASE release: ASE-3.10.0.

* A **BUG** related to use of time-reversal symmetry was found in the
  `G_0W_0` code that was introduced in version 0.11.  This has been `fixed
  now`_ --- *please run your calculations again*.

* New :mod:`gpaw.external` module.

* The gradients of the cavity and the dielectric in the continuum
  solvent model are now calculated analytically for the case of the
  effective potential method. This improves the accuracy of the forces
  in solution compared to the gradient calculated by finite
  differences. The solvation energies are expected to change slightly
  within the accuracy of the model.

* New `f_{\text{xc}}` kernels for correlation energy calculations.  See this
  updated :ref:`tutorial <rapbe_tut>`.

* Correlation energies within the range-separated RPA.  See this
  :ref:`tutorial <rangerpa_tut>`.

* Experimental interface to the libvdwxc_ library
  for efficient van der Waals density functionals.

* It's now possible to use Davidson and CG eigensolvers for MGGA calculations.

* The functional name "M06L" is now deprecated.  Use "M06-L" from now on.


.. _fixed now: https://gitlab.com/gpaw/gpaw/commit/c72e02cd789
.. _libvdwxc: https://gitlab.com/libvdwxc/libvdwxc


Version 0.11.0
==============

22 July 2015: :git:`0.11.0 <../0.11.0>`.

* Corresponding ASE release: ASE-3.9.1.

* When searching for basis sets, the setup name if any is now
  prepended automatically to the basis name.  Thus if
  :file:`setups='<setupname>'` and :file:`basis='<basisname>'`, GPAW
  will search for :file:`<symbol>.<setupname>.<basisname>.basis`.

* :ref:`Time-propagation TDDFT with LCAO <lcaotddft>`.

* Improved distribution and load balance when calculating atomic XC
  corrections, and in LCAO when calculating atomic corrections to the
  Hamiltonian and overlap.

* Norm-conserving :ref:`SG15 pseudopotentials <manual_setups>` and
  parser for several dialects of the UPF format.

* Non-selfconsistent spin-orbit coupling have been added. See :ref:`tutorial
  <spinorbit>` for examples of band structure calculations with spin-orbit
  coupling.

* Text output from ground-state calculations now list the symmetries found
  and the **k**-points used.  Eigenvalues and occupation numbers are now
  also printed for systems with **k**-points.

* :ref:`GW <gw exercise>`, :ref:`rpa`, and :ref:`response function
  calculation <df_tutorial>` has been rewritten to take advantage of
  symmetry and fast matrix-matrix multiplication (BLAS).

* New :ref:`symmetry <manual_symmetry>` keyword.  Replaces ``usesymm``.

* Use non-symmorphic symmetries: combining fractional translations with
  rotations, reflections and inversion.  Use
  ``symmetry={'symmorphic': False}`` to turn this feature on.

* New :ref:`forces <manual_convergence>` keyword in convergence.  Can
  be used to calculate forces to a given precision.

* Fixed bug in printing work functions for calculations with a
  dipole-correction `<http://listserv.fysik.dtu.dk/pipermail/
  gpaw-users/2015-February/003226.html>`_.

* A :ref:`continuum solvent model <continuum_solvent_model>` was added.

* A :ref:`orbital-free DFT <ofdft>` with PAW transformation is available.

* GPAW can now perform :ref:`electrodynamics` simulations using the
  quasistatic finite-difference time-domain (QSFDTD) method.

* BEEF-vdW, mBEEF and mBEEF-vdW functionals added.

* Support for Python 3.


Version 0.10.0
==============

8 April 2014: :git:`0.10.0 <../0.10.0>`.

* Corresponding ASE release: ASE-3.8.1

* Default eigensolver is now the Davidson solver.

* Default density mixer parameters have been changed for calculations
  with periodic boundary conditions.  Parameters for that case:
  ``Mixer(0.05, 5, 50)`` (or ``MixerSum(0.05, 5, 50)`` for spin-paired
  calculations.  Old parameters: ``0.1, 3, 50``.

* Default is now ``occupations=FermiDirac(0.1)`` if a
  calculation is periodic in at least one direction,
  and ``FermiDirac(0.0)`` otherwise (before it was 0.1 eV for anything
  with **k**-points, and 0 otherwise).

* Calculations with a plane-wave basis set are now officially supported.

* :ref:`One-shot GW calculations <gw_theory>` with full frequency
  integration or plasmon-pole approximation.

* Beyond RPA-correlation: `using renormalized LDA and PBE
  <https://trac.fysik.dtu.dk/projects/gpaw/browser/branches/sprint2013/doc/tutorials/fxc_correlation>`_.

* :ref:`bse theory`.

* Improved RMM-DIIS eigensolver.

* Support for new libxc 2.0.1.  libxc must now be built separately from GPAW.

* MGGA calculations can be done in plane-wave mode.

* Calculation of the stress tensor has been implemented for plane-wave
  based calculation (except MGGA).

* MGGA: number of neighbor grid points to use for FD stencil for
  wave function gradient changed from 1 to 3.

* New setups: Y, Sb, Xe, Hf, Re, Hg, Tl, Rn

* Non self-consistent calculations with screened hybrid functionals
  (HSE03 and HSE06) can be done in plane-wave mode.

* Modified setups:

  .. note::

     Most of the new semicore setups currently require
     :ref:`eigensolver <manual_eigensolver>` ``dav``, ``cg``
     eigensolvers or ``rmm-diis`` eigensolver with a couple of iterations.

  - improved eggbox: N, O, K, S, Ca, Sc, Zn, Sr, Zr, Cd, In, Sn, Pb, Bi

  - semicore states included: Na, Mg, V, Mn, Ni,
    Nb, Mo, Ru (seems to solve the Ru problem :git:`gpaw/test/big/Ru001/`),
    Rh, Pd, Ag, Ta, W, Os, Ir, Pt

  - semicore states removed: Te

  - elements removed: La (energetics was wrong: errors ~1eV per unit cell
    for PBE formation energy of La2O3 wrt. PBE benchmark results)

  .. note::

     For some of the setups one has now a choice of different
     number of valence electrons, e.g.::

       setups={'Ag': '11'}

     See :ref:`manual_setups` and list the contents of :envvar:`GPAW_SETUP_PATH`
     for available setups.

* new ``dzp`` basis set generated for all the new setups, see
  https://trac.fysik.dtu.dk/projects/gpaw/ticket/241


Version 0.9.0
=============

7 March 2012: :git:`0.9.0 <../0.9.0>`.

* Corresponding ASE release: ase-3.6

* Convergence criteria for eigenstates changed: The missing volume per
  grid-point factor is now included and the units are now eV**2. The
  new default value is 4.0e-8 eV**2 which is equivalent to the old
  default for a grid spacing of 0.2 Å.

* GPAW should now work also with NumPy 1.6.

* Much improved :ref:`cli` now based on the `new tool`_ in ASE.


.. _new tool: https://wiki.fysik.dtu.dk/ase/ase/cmdline.html


Version 0.8.0
=============

25 May 2011: :git:`0.8.0 <../0.8.0>`.

* Corresponding ASE release: ase-3.5.1
* Energy convergence criterion changed from 1 meV/atom to 0.5
  meV/electron.  This was changed in order to allow having no atoms like
  for jellium calculations.
* Linear :ref:`dielectric response <df_theory>` of an extended system
  (RPA and ALDA kernels) can now be calculated.
* :ref:`rpa`.
* Non-selfconsistent calculations with k-points for hybrid functionals.
* Methfessel-Paxton distribution added.
* Text output now shows the distance between planes of grid-points as
  this is what will be close to the grid-spacing parameter *h* also for
  non-orthorhombic cells.
* Exchange-correlation code restructured.  Naming convention for
  explicitely specifying libxc functionals has changed: :ref:`manual_xc`.
* New PAW setups for Rb, Ti, Ba, La, Sr, K, Sc, Ca, Zr and Cs.


Version 0.7.2
=============

13 August 2010: :git:`0.7.2 <../0.7.2>`.

* Corresponding ASE release: ase-3.4.1
* For version 0.7, the default Poisson solver was changed to
  ``PoissonSolver(nn=3)``.  Now, also the Poisson solver's default
  value for ``nn`` has been changed from ``'M'`` to ``3``.


Version 0.7
===========

23 April 2010: :git:`0.7 <../0.7>`.

* Corresponding ASE release: ase-3.4.0
* Better and much more efficient handling of non-orthorhombic unit
  cells.  It may actually work now!
* Much better use of ScaLAPACK and BLACS.  All large matrices can now
  be distributed.
* New test coverage pages for all files.
* New default value for Poisson solver stencil: ``PoissonSolver(nn=3)``.
* Much improved MPI module (:ref:`communicators`).
* Self-consistent Meta GGA.
* New :ref:`PAW setup tar-file <setups>` now contains revPBE setups and
  also dzp basis functions.
* New ``$HOME/.gpaw/rc.py`` configuration file.
* License is now GPLv3+.
* New HDF IO-format.
* :ref:`Advanced GPAW Test System <big-test>` Introduced.


Version 0.6
===========

9 October 2009: :git:`0.6 <../0.6>`.

* Corresponding ASE release: ase-3.2.0
* Much improved default parameters.
* Using higher order finite-difference stencil for kinetic energy.
* Many many other improvements like: better parallelization, fewer bugs and
  smaller memory footprint.


Version 0.5
===========

1 April 2009: :git:`0.5 <../0.5>`.

* Corresponding ASE release: ase-3.1.0
* `new setups added Bi, Br, I, In, Os, Sc, Te; changed Rb setup <https://trac.fysik.dtu.dk/projects/gpaw/changeset/3612>`_.
* `memory estimate feature is back <https://trac.fysik.dtu.dk/projects/gpaw/changeset/3575>`_


Version 0.4
===========

13 November 2008: :git:`0.4 <../0.4>`.

* Corresponding ASE release: ase-3.0.0
* Now using ASE-3 and numpy.
* TPSS non self-consistent implementation.
* LCAO mode.
* VdW-functional now coded in C.
* Added atomic orbital basis generation scripts.
* Added an Overlap object, and moved apply_overlap and apply_hamiltonian
  from Kpoint to Overlap and Hamiltonian classes.

* Wannier code much improved.
* Experimental LDA+U code added.
* Now using libxc.
* Many more setups.
* Delta scf calculations.

* Using localized functions will now no longer use MPI group
  communicators and blocking calls to MPI_Reduce and MPI_Bcast.
  Instead non-blocking sends/receives/waits are used.  This will
  reduce synchronization time for large parallel calculations.

* More work on LB94.
* Using LCAO code forinitial guess for grid calculations.
* TDDFT.
* Moved documentation to Sphinx.
* Improved metric for Pulay mixing.
* Porting and optimization for BlueGene/P.
* Experimental Hartwigsen-Goedecker-Hutter pseudopotentials added.
* Transport calculations with LCAO.


Version 0.3
===========

19 December 2007: :git:`0.3 <../0.3>`.
