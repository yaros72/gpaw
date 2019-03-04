# Creates: emt-iterations.csv, lcao-time.csv, systems.csv
import ase.db
from ase.optimize.test.analyze import analyze

analyze('results-emt.db', 'emt')
analyze('results-lcao.db', 'lcao')

db1 = ase.db.connect('systems.db')

with open('systems.csv', 'w') as f:
    print('test-name,description', file=f)
    for row in db1.select():
        print('{},{}'.format(row.name, row.description), file=f)
