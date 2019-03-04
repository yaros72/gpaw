=======================
Bugs in the latest GPAW
=======================

See here: :ref:`bugs`


------------------
Handling segfaults
------------------

Segmentation faults are probably the hardest type of runtime error to track
down, but they are also quite common during the *unstable* part of the
release cycle. As a rule of thumb, if you get a segfault, start by checking
that all array arguments passed from Python to C functions have the correct
shapes and types.

Apart from appending ``--debug`` to the command line arguments when running
``python`` or ``gpaw-python``, please familiarize yourself with the
:ref:`debugging tools <debugging>` for the Python and C code.

If you experience segfaults or unexplained MPI crashes when running GPAW
in parallel, it is recommended to try a :ref:`custom installation
<customizing installation>` with a debugging flag in ``customize.py``::

    define_macros += [('GPAW_MPI_DEBUG', 1)]


----------------------
Common sources of bugs
----------------------

* General:


  - Elements of NumPy arrays are C ordered, BLAS and LAPACK routines expect
    Fortran ordering.

* Python:

  - Always give contiguous arrays to C functions. If ``x`` is contiguous with
    ``dtype=complex``, then ``x.real`` is non-contiguous of ``dtype=float``.

  - Giving array arguments to a function is a *carte blanche* to alter the data::

      def double(a):
          a *= 2
          return a

      x = np.ones(5)
      print(double(x))  # x[:] is now 2.

  - Forgetting a ``n += 1`` statement in a for loop::

      n = 0
      for thing in things:
          thing.do_stuff(n)
          n += 1

    Use this instead::

      for n, thing in enumerate(things):
          thing.do_stuff(n)

  - Indentation errors like this one::

     if ok:
         x = 1.0
     else:
         x = 0.5
         do_stuff(x)

    where ``do_stuff(x)`` should have been reached in both cases.
    Emacs: always use ``C-c >`` and ``C-c <`` for shifting in and out
    blocks of code (mark the block first).

  - Don't use mutables as default values::

     class A:
         def __init__(self, a=[]):
             self.a = a # All instances get the same list!

  - There are subtle differences between ``x == y`` and ``x is y``.

  - If ``H`` is a numeric array, then ``H - x`` will subtract ``x``
    from *all* elements - not only the diagonal, as in Matlab!


* C:

  - Try building GPAW from scratch.
  - Typos like ``if (x = 0)`` which should have been ``if (x == 0)``.
  - Remember ``break`` in switch-case statements.
  - Check ``malloc-free`` pairs. Test for :ref:`memory leaks <memory_leaks>`
    by repeating the call many times.
  - Remember to update reference counts of Python objects.
  - *Never* put function calls inside ``assert``'s.  Compiling with
    ``-DNDEBUG`` will remove the call.
