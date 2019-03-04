from __future__ import print_function
from functools import partial

import numpy as np
from gpaw.utilities.grid_redistribute import general_redistribute
from gpaw.utilities.partition import AtomPartition, AtomicMatrixDistributor


class GridRedistributor:
    def __init__(self, comm, broadcast_comm, gd, aux_gd):
        self.comm = comm
        self.broadcast_comm = broadcast_comm
        self.gd = gd
        self.aux_gd = aux_gd
        self.enabled = np.any(gd.parsize_c != aux_gd.parsize_c)

        assert gd.comm.size * broadcast_comm.size == comm.size
        if self.enabled:
            assert comm.compare(aux_gd.comm) in ['ident', 'congruent']
        else:
            assert gd.comm.compare(aux_gd.comm) in ['ident', 'congruent']

        if aux_gd.comm.rank == 0:
            aux_ranks = gd.comm.translate_ranks(aux_gd.comm,
                                                np.arange(gd.comm.size))
        else:
            aux_ranks = np.empty(gd.comm.size, dtype=int)
        aux_gd.comm.broadcast(aux_ranks, 0)

        auxrank2rank = dict(zip(aux_ranks, np.arange(gd.comm.size)))

        def rank2parpos1(rank):
            if rank in auxrank2rank:
                return gd.get_processor_position_from_rank(auxrank2rank[rank])
            else:
                return None

        rank2parpos2 = aux_gd.get_processor_position_from_rank

        try:
            gd.n_cp
        except AttributeError:  # AtomPAW
            self._distribute = self._collect = lambda x: None
            return  # XXX

        self._distribute = partial(general_redistribute, aux_gd.comm,
                                   gd.n_cp, aux_gd.n_cp,
                                   rank2parpos1, rank2parpos2)
        self._collect = partial(general_redistribute, aux_gd.comm,
                                aux_gd.n_cp, gd.n_cp,
                                rank2parpos2, rank2parpos1)

    def distribute(self, src_xg, dst_xg=None):
        if not self.enabled:
            assert src_xg is dst_xg or dst_xg is None
            return src_xg
        if dst_xg is None:
            dst_xg = self.aux_gd.empty(src_xg.shape[:-3], dtype=src_xg.dtype)
        self._distribute(src_xg, dst_xg)
        return dst_xg

    def collect(self, src_xg, dst_xg=None):
        if not self.enabled:
            assert src_xg is dst_xg or dst_xg is None
            return src_xg
        if dst_xg is None:
            dst_xg = self.gd.empty(src_xg.shape[:-3], src_xg.dtype)
        self._collect(src_xg, dst_xg)
        self.broadcast_comm.broadcast(dst_xg, 0)
        return dst_xg

    def get_atom_distributions(self, spos_ac):
        return AtomDistributions(self.comm, self.broadcast_comm,
                                 self.gd, self.aux_gd, spos_ac)


class AtomDistributions:
    def __init__(self, comm, broadcast_comm, gd, aux_gd, spos_ac):
        self.comm = comm
        self.broadcast_comm = broadcast_comm
        self.gd = gd
        self.aux_gd = aux_gd

        rank_a = gd.get_ranks_from_positions(spos_ac)
        aux_rank_a = aux_gd.get_ranks_from_positions(spos_ac)
        self.partition = AtomPartition(gd.comm, rank_a, name='gd')

        if gd is aux_gd:
            name = 'aux-unextended'
        else:
            name = 'aux-extended'
        self.aux_partition = AtomPartition(aux_gd.comm, aux_rank_a, name=name)

        self.work_partition = AtomPartition(comm, np.zeros(len(spos_ac)),
                                            name='work').as_even_partition()

        if gd is aux_gd:
            aux_broadcast_comm = gd.comm.new_communicator([gd.comm.rank])
        else:
            aux_broadcast_comm = broadcast_comm

        self.aux_dist = AtomicMatrixDistributor(self.partition,
                                                aux_broadcast_comm,
                                                self.aux_partition)
        self.work_dist = AtomicMatrixDistributor(self.partition,
                                                 broadcast_comm,
                                                 self.work_partition)

    def to_aux(self, arraydict):
        if self.gd is self.aux_gd:
            return arraydict.copy()
        return self.aux_dist.distribute(arraydict)

    def from_aux(self, arraydict):
        if self.gd is self.aux_gd:
            return arraydict.copy()
        return self.aux_dist.collect(arraydict)

    def to_work(self, arraydict):
        return self.work_dist.distribute(arraydict)

    def from_work(self, arraydict):
        return self.work_dist.collect(arraydict)


def grid2grid(comm, gd1, gd2, src_g, dst_g, offset1_c=None, offset2_c=None):
    assert np.all(src_g.shape == gd1.n_c)
    assert np.all(dst_g.shape == gd2.n_c)

    #master1_rank = gd1.comm.translate_ranks(comm, [0])[0]
    #master2_rank = gd2.comm.translate_ranks(comm, [0])[0]

    ranks1 = gd1.comm.translate_ranks(comm, np.arange(gd1.comm.size))
    ranks2 = gd2.comm.translate_ranks(comm, np.arange(gd2.comm.size))
    assert (ranks1 >= 0).all(), 'comm not parent of gd1.comm'
    assert (ranks2 >= 0).all(), 'comm not parent of gd2.comm'

    def rank2parpos(gd, rank):
        gdrank = comm.translate_ranks(gd.comm, np.array([rank]))[0]
        # XXXXXXXXXXXXX segfault when not passing array!!
        if gdrank == -1:
            return None
        return gd.get_processor_position_from_rank(gdrank)
    rank2parpos1 = partial(rank2parpos, gd1)
    rank2parpos2 = partial(rank2parpos, gd2)


    def add_offset(n_cp, offset_c):
        n_cp = [n_p.copy() for n_p in n_cp]
        for c in range(3):
            n_cp[c] += offset_c[c]
        return n_cp

    n1_cp = gd1.n_cp
    if offset1_c is not None:
        n1_cp = add_offset(n1_cp, offset1_c)

    n2_cp = gd2.n_cp
    if offset2_c is not None:
        n2_cp = add_offset(n2_cp, offset2_c)

    general_redistribute(comm,
                         n1_cp, n2_cp,
                         rank2parpos1, rank2parpos2,
                         src_g, dst_g)

def main():
    from gpaw.grid_descriptor import GridDescriptor
    from gpaw.mpi import world

    serial = world.new_communicator([world.rank])

    # Generator which must run on all ranks
    gen = np.random.RandomState(0)

    # This one is just used by master
    gen_serial = np.random.RandomState(17)

    maxsize = 5
    for i in range(1):
        N1_c = gen.randint(1, maxsize, 3)
        N2_c = gen.randint(1, maxsize, 3)

        gd1 = GridDescriptor(N1_c, N1_c)
        gd2 = GridDescriptor(N2_c, N2_c)
        serial_gd1 = gd1.new_descriptor(comm=serial)
        # serial_gd2 = gd2.new_descriptor(comm=serial)

        a1_serial = serial_gd1.empty()
        a1_serial.flat[:] = gen_serial.rand(a1_serial.size)

        if world.rank == 0:
            print('r0: a1 serial', a1_serial.ravel())

        a1 = gd1.empty()
        a1[:] = -1

        grid2grid(world, serial_gd1, gd1, a1_serial, a1)

        print(world.rank, 'a1 distributed', a1.ravel())
        world.barrier()

        a2 = gd2.zeros()
        a2[:] = -2
        grid2grid(world, gd1, gd2, a1, a2)
        print(world.rank, 'a2 distributed', a2.ravel())
        world.barrier()

        #grid2grid(world, gd2, gd2_serial

        gd1 = GridDescriptor(N1_c, N1_c * 0.2)
        #serialgd = gd2.new_descriptor(

        a1 = gd1.empty()
        a1.flat[:] = gen.rand(a1.size)

        #print a1
        grid2grid(world, gd1, gd2, a1, a2)

        #print a2

if __name__ == '__main__':
    main()
