.. _newrelease:

===========
New release
===========

* Update ``__version__`` in :git:`gpaw/__init__.py`.

* If a new ase release is required to pass the tests
  modify ``__ase_version_required__`` in :git:`gpaw/__init__.py`.

* Upload to PyPI::

      $ python3 setup.py sdist
      $ twine upload dist/*

* Push and make a tag.

* Update :ref:`news`, :ref:`releasenotes` and :ref:`download` pages.

* Increase the version number and push.

* Send announcement email to the ``gpaw-users`` mailing list.
