from gpaw import GPAW
from gpaw.borncharges import borncharges

calc = GPAW('BaTiO3.gpw', txt=None)
borncharges(calc)
