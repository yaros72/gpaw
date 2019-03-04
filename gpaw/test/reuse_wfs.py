from ase.build import bulk
from gpaw import GPAW, Mixer, PW

atoms0 = bulk('Si')
atoms0.rattle(stdev=0.01, seed=17)  # Break symmetry

results = []

for mode in ['fd', 'pw']:
    for method in [None, 'paw', 'lcao']:
        atoms = atoms0.copy()

        kwargs = {}
        if mode == 'pw':
            if method == 'lcao':
                continue  # Not implemented yet
            kwargs.update(mode=PW(400.0))

        calc = GPAW(mixer=Mixer(0.4, 5, 20.0),
                    basis='dzp' if method == 'lcao' else {},
                    experimental={'reuse_wfs_method': method},
                    xc='oldLDA',
                    kpts=[2, 2, 2],
                    **kwargs)
        atoms.calc = calc
        E1 = atoms.get_potential_energy()
        niter1 = calc.scf.niter
        atoms.rattle(stdev=0.05)
        E2 = atoms.get_potential_energy()
        niter2 = calc.scf.niter
        results.append([mode, method, E1, E2, niter1, niter2])


if calc.wfs.world.rank == 0:
    for result in results:
        print(' '.join(str(x) for x in result))
