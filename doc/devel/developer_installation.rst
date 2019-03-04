.. _developer installation:

======================
Developer installation
======================

Start by :ref:`forking and cloning as it is done for ASE develoment
<ase:contribute>`.  Let ``<repo>`` be the folder where you have cloned to
(could be ``~/gpaw``). Then do::

    $ cd <repo>
    $ python3 setup.py build_ext

This will build two things:

1) :file:`_gpaw.so`:  A shared library for serial calculations containing
   GPAW's C-extension module.  The module will be in
   :file:`<repo>/build/lib.<platform>-X.Y/`.

   For example ``<platform>`` could be *linux-x86_64*, and
   ``X.Y`` could be *2.7*.

2) :file:`gpaw-python`: A special Python interpreter for parallel
   calculations.  The interpreter has GPAW's C-code built in.  The
   :file:`gpaw-python` executable is located
   in :file:`<repo>/build/bin.<platform>-X.Y/`.

   .. note::

       The :file:`gpaw-python` interpreter will be built only if
       :git:`setup.py` finds an ``mpicc`` compiler.

Prepend :file:`<repo>` and :file:`<repo>/build/lib.<platform>-X.Y/`
onto your :envvar:`PYTHONPATH` and
:file:`<repo>/tools:<repo>/build/bin.<platform>-X.Y` onto
:envvar:`PATH` as described here: :ref:`envvars`.
