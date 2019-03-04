.. _lrtddft2:

=========================================
Linear response TDDFT 2 - indexed version
=========================================

Ground state
============

The linear response TDDFT calculation needs a converged ground state
calculation with a set of unoccupied states. It is safer to use 'dav' or 'cg'
eigensolver instead of the default 'rmm-diis' eigensolver to converge
unoccupied states. However, 'dav' and 'cg' are often too expensive for large
systems. In this case, you should use 'rmm-diis' with tens or hundreds of
extra states in addition to the unoccupied states you wish to converge.

.. literalinclude:: r-methyl-oxirane.xyz

.. literalinclude:: Oxirane_lrtddft2_unocc.py


Calculating response matrix and spectrum
========================================

The next step is to calculate the response matrix:

.. literalinclude:: Oxirane_lrtddft2_lr.py

Note: Unfortunately, spin is not implemented yet. For now, use 'lrtddft'.


Restarting, recalculating and analyzing spectrum
================================================

.. literalinclude:: Oxirane_lrtddft2_lr2.py


Quick reference
===============

Parameters for LrCommunicators:

================  =================  ===================  ========================================
keyword           type               default value        description
================  =================  ===================  ========================================
``world``          ``Communicator``  None                 parent communicator 
                                                          (usually gpaw.mpi.world)
``dd_size``        `int``            None                 Number of domains for domain 
                                                          decomposition
``eh_size``        `int``            None                 Number of groups for parallelization
                                                          over e-h -pairs
================  =================  ===================  ========================================

Note: world.size = dd_size x eh_size


Parameters for LrTDDFT2:

====================  ==================  ===================  ========================================
keyword               type                default value        description
====================  ==================  ===================  ========================================
``basefilename``      ``string``                               Prefix for all files created by LRTDDFT2
                                                               (e.g. ``dir/lr``)
``gs_calc``           ``GPAW``                                 Ground-state calculator, which has been 
                                                               loaded from a file with 
                                                               communicator=lr_communicators
                                                               calculation
``fxc``               ``string``          None                 Exchange-correlation kernel
``min_occ``           ``int``             None                 Index of the first occupied state 
                                                               (inclusive)
``max_occ``           ``int``             None                 Index of the last occupied state 
                                                               (inclusive)
``min_unocc``         ``int``             None                 Index of the first unoccupied state 
                                                               (inclusive)
``max_unocc``         ``int``             None                 Index of the last unoccupied state 
                                                               (inclusive)
``max_energy_diff``   ``float``           None                 Maximum Kohn-Sham eigenenergy difference
``recalculate``       ``string``          None                 What should be recalculated. Usually 
                                                               nothing.
``lr_communicators``  ``LrCommuncators``  None                 
``txt``               ``string``          None                 Output
====================  ==================  ===================  ========================================
