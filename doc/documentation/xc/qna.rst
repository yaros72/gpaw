.. module:: gpaw.xc.qna
.. _qna:

=================================================
Quasi-non-local exchange correlation approxmation
=================================================

Rationale

---------
Using QNA
---------

Code example::

  from gpaw import GPAW, PW
  from ase.lattice.compounds import L1_2

  QNA = {'alpha': 2.0,
         'name': 'QNA',
         'orbital_dependent': False,
         'parameters': {'Au': (0.125, 0.1), 'Cu': (0.0795, 0.005)},
         'setup_name': 'PBE',
         'type': 'qna-gga'}

  atoms = L1_2(['Au','Cu'],latticeconstant=3.74)
  calc = GPAW(mode=PW(300),
              xc = QNA,
              kpts=kpts,
              txt='AuCu3_QNA.txt')
  atoms.get_potential_energy()


.. autoclass:: gpaw.xc.qna.QNA
   :members:
