from gpaw.atom.generator2 import generate
from gpaw.mpi import world


def test():
    G = generate('Li', '2s,2p,s', [2.1, 2.1], 2.0, 2, 'PBE', True)
    assert G.check_all()
    basis = G.create_basis_set()
    basis.write_xml()
    setup = G.make_paw_setup('test')
    setup.write_xml()


if world.size == 1:
    test()
