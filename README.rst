GPAW
====

GPAW is a density-functional theory (DFT) Python_ code based on the
projector-augmented wave (PAW) method and the atomic simulation environment
(ASE_). It uses plane-waves, atom-centered basis-functions or real-space
uniform grids combined with multigrid methods.

Webpage: http://wiki.fysik.dtu.dk/gpaw


Requirements
------------

* Python_ 2.7, 3.4-
* ASE_ (atomic simulation environment)
* NumPy_ (base N-dimensional array package)
* SciPy_ (library for scientific computing)
* LibXC
* BLAS
* LAPACK

Optional:

* MPI
* ScaLAPACK


Installation
------------

Do this::

    $ python setup.py install --user

and make sure you have ``~/.local/bin`` in your $PATH.

For more details, please see:

    https://wiki.fysik.dtu.dk/gpaw/install.html


Testing
-------

Please run the tests::

    $ gpaw test -j 4  # takes 1 hour!

and send us the output if there are failing tests.


Contact
-------

* Mailing list: gpaw-users_
* IRC_: #gpaw on freenode.net
* Bug reports and development: gitlab-issues_
* Old development mailing list (archive): gpaw-developers_

Please send us bug-reports, patches, code, ideas and questions.


Example
-------

Geometry optimization of hydrogen molecule:

>>> from ase import Atoms
>>> from ase.optimize import BFGS
>>> from ase.io import write
>>> from gpaw import GPAW, PW
>>> h2 = Atoms('H2',
               positions=[[0, 0, 0],
                          [0, 0, 0.7]])
>>> h2.center(vacuum=2.5)
>>> h2.set_calculator(GPAW(xc='PBE',
                           mode=PW(300),
                           txt='h2.txt'))
>>> opt = BFGS(h2, trajectory='h2.traj')
>>> opt.run(fmax=0.02)
BFGS:   0  09:08:09       -6.566505       2.2970
BFGS:   1  09:08:11       -6.629859       0.1871
BFGS:   2  09:08:12       -6.630410       0.0350
BFGS:   3  09:08:13       -6.630429       0.0003
>>> write('H2.xyz', h2)
>>> h2.get_potential_energy()  # ASE's units are eV and Å
-6.6304292169392784


Getting started
---------------

Once you have familiarized yourself with ASE_ and NumPy_, you should take a
look at the GPAW exercises_ and tutorials_.


.. _Python: http://www.python.org/
.. _ASE: http://wiki.fysik.dtu.dk/ase
.. _NumPy: http://docs.scipy.org/doc/numpy/reference/
.. _SciPy: http://docs.scipy.org/doc/scipy/reference/
.. _Matplotlib: http://matplotlib.org/
.. _pygtk: http://www.pygtk.org/
.. _gpaw-users: https://listserv.fysik.dtu.dk/mailman/listinfo/gpaw-users
.. _gpaw-developers: https://listserv.fysik.dtu.dk/mailman/listinfo/gpaw-developers
.. _IRC: http://webchat.freenode.net/?randomnick=0&channels=gpaw
.. _gitlab-issues: https://gitlab.com/gpaw/gpaw/issues
.. _exercises: https://wiki.fysik.dtu.dk/gpaw/exercises/exercises.html
.. _tutorials: https://wiki.fysik.dtu.dk/gpaw/tutorials/tutorials.html
