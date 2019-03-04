import ase.db
from ase.io.jsonio import encode

from gpaw.atom.generator import Generator
from gpaw.atom.configurations import parameters, parameters_extra
from gpaw.atom.check import check, summary, all_names

con = ase.db.connect('datasets.db')

for name in all_names:
    check(con, name)

data = {}
for name in all_names:
    if '.' in name:
        symbol, e = name.split('.')
        params = parameters_extra[symbol]
        assert params['name'] == e
    else:
        symbol = name
        e = 'default'
        params = parameters[symbol]
        data[symbol] = []
        
    gen = Generator(symbol, 'PBE', scalarrel=True, txt=None)
    gen.run(write_xml=False, **params)
    nlfer = []
    for j in range(gen.njcore):
        nlfer.append((gen.n_j[j], gen.l_j[j], gen.f_j[j], gen.e_j[j], 0.0))
    for n, l, f, eps in zip(gen.vn_j, gen.vl_j, gen.vf_j, gen.ve_j):
        nlfer.append((n, l, f, eps, gen.rcut_l[l]))

    energies = summary(con, name)

    data[symbol].append((e, nlfer, energies))

with open('datasets.json', 'w') as fd:
    fd.write(encode(data))
