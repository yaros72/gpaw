import ase.db
from ase.calculators.emt import EMT
from ase.optimize.test.test import (test_optimizer, all_optimizers,
                                    get_optimizer)


db1 = ase.db.connect('systems.db')
db = ase.db.connect('results-emt.db', serial=True)

systems = [(row.name, row.toatoms())
           for row in db1.select() if row.formula != 'C5H12']

for opt in all_optimizers:
    optimizer = get_optimizer(opt)
    test_optimizer(systems, optimizer, EMT, 'emt-', db)
