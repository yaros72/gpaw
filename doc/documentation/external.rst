.. module:: gpaw.external

External potential
==================

Examples
--------
    
>>> # 2.5 eV/Ang along z:
>>> from gpaw.external import ConstantElectricField
>>> calc = GPAW(external=ConstantElectricField(2.5, [0, 0, 1]), ...)

.. autoclass:: ConstantElectricField

>>> # Two point-charges:
>>> from gpaw.external import PointChargePotential
>>> pc = PointChargePotential([-1, 1], [[4.0, 4.0, 0.0], [4.0, 4.0, 10.0]])
>>> calc = GPAW(external=pc, ...)

.. autoclass:: PointChargePotential


Your own potential
------------------

See an example here: :git:`gpaw/test/ext_potential/harmonic.py`.
