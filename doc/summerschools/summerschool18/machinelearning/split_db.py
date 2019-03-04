# creates: organometal.db
from ase.db import connect
import random

in_db = connect('organometal.master.db')
out_db = connect('organometal.db')

preserved_names = ['MAPbI3', 'FASnBr2Cl', 'FASnBr3', 'FASnCl3']
rows = [row for row in in_db.select(project='organometal')]
names = set([row.name for row in rows])
for name in names:
    n = random.randint(1, 4)
    sub_rows = [x for x in rows if x.name == name]
    if name not in preserved_names:
        sub_rows = list(random.sample(sub_rows, n))
    for row in sub_rows:
        out_db.write(row, **row.key_value_pairs)

rows = [row for row in in_db.select(subproject='references')]
for row in rows:
    print(row)
    print(row.key_value_pairs)
    out_db.write(row, **row.key_value_pairs)
print()
