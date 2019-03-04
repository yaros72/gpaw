import ase.db
from ase.build import bulk

from gpaw import GPAW, PW, FermiDirac
from gpaw.response.g0w0 import G0W0


data = {
    'C': ['diamond', 3.567],
    'SiC': ['zincblende', 4.358],
    'Si': ['diamond', 5.431],
    'Ge': ['diamond', 5.658],
    'BN': ['zincblende', 3.616],
    'AlP': ['zincblende', 5.463],
    'AlAs': ['zincblende', 5.661],
    'AlSb': ['zincblende', 6.136],
    'GaN': ['zincblende', 4.50],
    # 'GaN': ['zincblende', ],
    'GaP': ['zincblende', 5.451],
    'GaAs': ['zincblende', 5.654],
    'GaSb': ['zincblende', 6.096],
    'InP': ['zincblende', 5.869],
    'InAs': ['zincblende', 6.058],
    'InSb': ['zincblende', 6.479],
    'MgO': ['zincblende', 4.53],
    'ZnO': ['zincblende', 4.60],
    # 'ZnO': ['zincblende', ],
    'ZnS': ['zincblende', 5.409],
    'ZnSe': ['zincblende', 5.669],
    'ZnTe': ['zincblende', 6.103],
    'CdO': ['zincblende', 5.148],
    'CdS': ['zincblende', 5.818],
    'CdSe': ['zincblende', 6.077],
    'CdTe': ['zincblende', 6.481]}

c = ase.db.connect('gw.db')
tag = 'a4'

for name in data:
    id = c.reserve(name=name, tag=tag)
    if id is None:
        continue
        
    x, a = data[name]
    atoms = bulk(name, x, a=a)
    atoms.calc = GPAW(mode=PW(600),
                      xc='LDA',
                      parallel={'band': 1},
                      occupations=FermiDirac(0.02),
                      kpts={'size': (4, 4, 4), 'gamma': True},
                      txt='%s.txt' % name)
    atoms.get_potential_energy()
    atoms.calc.diagonalize_full_hamiltonian()
    atoms.calc.write(name, mode='all')
    n = int(atoms.calc.get_number_of_electrons()) // 2
    gw = G0W0(name, name + 'gw',
              nblocks=4,
              kpts=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)],
              ecut=100,
              wstc=True,
              domega0=0.2,
              omega2=10,
              eta=0.2,
              bands=(0, n + 2))
    results = gw.calculate()
    c.write(atoms, name=name, tag=tag, data=results)
    del c[id]
