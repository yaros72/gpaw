# -*- coding: utf-8 -*-
from __future__ import print_function
import itertools
import numpy as np
from gpaw.grid_descriptor import GridDescriptor


class AlignedGridRedistributor:
    """Perform redistributions between two grids.

    See the redistribute function."""
    def __init__(self, gd, distribute_dir, reduce_dir):
        self.gd = gd
        self.distribute_dir = distribute_dir
        self.reduce_dir = reduce_dir
        self.gd2 = get_compatible_grid_descriptor(gd, distribute_dir,
                                                  reduce_dir)

    def _redist(self, src, op):
        return redistribute(self.gd, self.gd2, src, self.distribute_dir,
                            self.reduce_dir, operation=op)

    def forth(self, src):
        return self._redist(src, 'forth')

    def back(self, src):
        return self._redist(src, 'back')


def redistribute(gd, gd2, src, distribute_dir, reduce_dir, operation='forth'):
    """Perform certain simple redistributions among two grid descriptors.

    Redistribute src from gd with decomposition X x Y x Z to gd2 with
    decomposition X x YZ x 1, or some variation of this.  We say that
    we "reduce" along Z while we "distribute" along Y.  The
    redistribution is one-to-one.

             ____________                           ____________
    i       /     /     /|          r              /  /  /  /  /|
    n      /_____/_____/ |         i              /  /  /  /  / |
    d     /     /     /| |        d              /  /  /  /  /  |
    e    /_____/_____/ | j           forth      /__/__/__/__/   j
    p    |     |     | |/|      e    ------->   |  |  |  |  |  /|
    e    |     |     | ł |     c    <-------    |  |  |  |  | / |
    n    |_____|_____|/| j    u       back      |__|__|__|__|/  j
    d    |     |     | |/    d                  |  |  |  |  |  /
    e    |     |     | /    e                   |  |  |  |  | /
    n    |_____|_____|/    r                    |__|__|__|__|/
    t

         d i s t r i b u t e   d i r

    Directions are specified as 0, 1, or 2.  gd2 must be serial along
    the axis of reduction and must parallelize enough over the
    distribution axis to match the size of gd.comm.

    Returns the redistributed array which is compatible with gd2.

    Note: The communicator of gd2 must in general be a special
    permutation of that of gd in order for the redistribution axes to
    align with domain rank assignment.  Use the helper function
    get_compatible_grid_descriptor to obtain a grid descriptor which
    uses a compatible communicator."""

    assert reduce_dir != distribute_dir
    assert gd.comm.size == gd2.comm.size
    # Actually: The two communicators should be equal!!
    for c in [reduce_dir, distribute_dir]:
        assert 0 <= c and c < 3

    # Determine the direction in which nothing happens.
    for c in range(3):
        if c != reduce_dir and c != distribute_dir:
            independent_dir = c
            break
    assert np.all(gd.N_c == gd2.N_c)
    assert np.all(gd.pbc_c == gd2.pbc_c)
    assert gd.n_c[independent_dir] == gd2.n_c[independent_dir]
    assert gd.parsize_c[independent_dir] == gd2.parsize_c[independent_dir]
    assert gd2.parsize_c[reduce_dir] == 1
    assert gd2.parsize_c[distribute_dir] == gd.parsize_c[reduce_dir] \
        * gd.parsize_c[distribute_dir]
    assert operation == 'forth' or operation == 'back'
    forward = (operation == 'forth')
    if forward:
        assert np.all(src.shape == gd.n_c)
    else:
        assert np.all(src.shape == gd2.n_c)
    assert gd.comm.compare(gd2.comm) != 'unequal'

    # We want this to work no matter which direction is distribute and
    # reduce.  But that is tricky to code.  So we use a standard order
    # of the three directions.
    #
    # Thus we have to always transpose the src/dst arrays consistently
    # when interacting with the contiguous MPI send/recv buffers.  An
    # alternative is to use np.take, but that sometimes produces
    # copies where slicing does not, and we want to write back into
    # slices.
    #
    # We only support some of them though...
    dirs = (independent_dir, distribute_dir, reduce_dir)
    src = src.transpose(*dirs)

    # Construct a communicator consisting of all those processes that
    # participate in domain decomposition along the reduction
    # direction.
    #
    # All necessary communication can be done within that
    # subcommunicator using MPI alltoallv.
    #
    # We also construct the "same" communicator from gd2.comm, but with the
    # sole purpose of testing that the ranks are consistent between the two.
    # If they are not, the two grid descriptors are incompatible.
    pos_c = gd.parpos_c.copy()
    pos2_c = gd2.parpos_c.copy()
    positions2_offset = pos_c[distribute_dir] * gd.parsize_c[reduce_dir]
    peer_ranks = []
    peer_ranks2 = []
    for i in range(gd.parsize_c[reduce_dir]):
        pos_c[reduce_dir] = i
        pos2_c[distribute_dir] = i + positions2_offset
        peer_ranks.append(gd.get_rank_from_processor_position(pos_c))
        peer_ranks2.append(gd2.get_rank_from_processor_position(pos2_c))
    peer_comm = gd.comm.new_communicator(peer_ranks)
    test_peer_comm2 = gd2.comm.new_communicator(peer_ranks2)
    if test_peer_comm2.compare(peer_comm) != 'congruent':
        raise ValueError('Grids are not compatible.  '
                         'Use get_compatible_grid_descriptor to construct '
                         'a compatible grid.')
    #assert peer_comm2 is not None
    assert peer_comm.compare(gd2.comm.new_communicator(peer_ranks2)) == 'congruent'
    #print('COMPARE', peer_ranks, peer_ranks2, peer_comm.compare(peer_comm2))

    # Now check that peer_comm encompasses the same physical processes
    # on the communicators of the two grid descriptors.
    #test1 = peer_comm.translate_ranks(gd.comm, np.arange(peer_comm.size))
    #test2 = peer_comm.translate_ranks(gd.comm, np.arange(peer_comm.size))
    #print(peer_comm)

    members = peer_comm.get_members()

    mynpts1_rdir = gd.n_c[reduce_dir]
    mynpts2_ddir = gd2.n_c[distribute_dir]
    mynpts_idir = gd.n_c[independent_dir]
    assert mynpts_idir == gd2.n_c[independent_dir]

    offsets1_rdir_p = gd.n_cp[reduce_dir]
    offsets2_ddir_p = gd2.n_cp[distribute_dir]

    npts1_rdir_p = offsets1_rdir_p[1:] - offsets1_rdir_p[:-1]
    npts2_ddir_p = offsets2_ddir_p[1:] - offsets2_ddir_p[:-1]

    # We have the sendbuffer, and it is contiguous.  But the parts
    # that we are going to send to each CPU are not contiguous!  We
    # need to loop over all the little chunks that we want to send,
    # and put them into a contiguous buffer for MPI.  Moreover, the
    # received data will unsurprisingly be in that very same order.
    # Therefore, we need to know how to unpack those data and put them
    # into the return array too.
    #
    # The following loop builds the send buffer, and manages the logic
    # for the receive buffer.  However since we have not received the
    # data, we obviously cannot copy anything out of the receive
    # buffer yet.  Therefore we create a list of ChunkCopiers that
    # contain all the information that they need to later copy things
    # into the appropriate places of the return array.

    if forward:
        dst = gd2.zeros(dtype=src.dtype)
    else:
        dst = gd.zeros(dtype=src.dtype)
    recvbuf = np.empty(dst.size, dtype=src.dtype)
    dst[:] = -2
    recvbuf[:] = -3

    sendchunks = []
    recvchunks = []
    recv_chunk_copiers = []

    class ChunkCopier:
        def __init__(self, src_chunk, dst_chunk):
            self.src_chunk = src_chunk
            self.dst_chunk = dst_chunk

        def copy(self):
            self.dst_chunk.flat[:] = self.src_chunk

    # Convert from peer_comm
    ranks1to2 = gd.comm.translate_ranks(gd2.comm, np.arange(gd.comm.size))
    assert (ranks1to2 != -1).all()

    recvchunk_start = 0
    for i in range(peer_comm.size):
        parent_rank = members[i]
        parent_rank2 = ranks1to2[parent_rank]

        parent_coord1 = \
            gd.get_processor_position_from_rank(parent_rank)[reduce_dir]
        parent_coord2 = \
            gd2.get_processor_position_from_rank(parent_rank2)[distribute_dir]

        # Warning: Many sendXXX and recvXXX variables are badly named
        # because they change roles when operation='back'.
        sendstart_ddir = offsets2_ddir_p[parent_coord2] \
            - gd.beg_c[distribute_dir]
        sendstop_ddir = sendstart_ddir + npts2_ddir_p[parent_coord2]
        sendnpts_ddir = sendstop_ddir - sendstart_ddir

        # Compensate for the infinitely annoying convention that enumeration
        # of points starts at 1 in non-periodic directions.
        #
        # Also, if we want to handle more general redistributions, the
        # below buffers must have something subtracted to get a proper
        # local index.
        recvstart_rdir = offsets1_rdir_p[parent_coord1] \
            - 1 + gd.pbc_c[reduce_dir]
        recvstop_rdir = recvstart_rdir + npts1_rdir_p[parent_coord1]
        recvnpts_rdir = recvstop_rdir - recvstart_rdir

        # Grab subarray that is going to be sent to process i.
        if forward:
            assert 0 <= sendstart_ddir
            assert sendstop_ddir <= src.shape[1]
            sendchunk = src[:, sendstart_ddir:sendstop_ddir, :]
            assert sendchunk.size == mynpts1_rdir * sendnpts_ddir * mynpts_idir, (sendchunk.shape, (mynpts_idir, sendnpts_ddir, mynpts1_rdir))
        else:
            sendchunk = src[:, :, recvstart_rdir:recvstop_rdir]
            assert sendchunk.size == recvnpts_rdir * mynpts2_ddir * mynpts_idir
        sendchunks.append(sendchunk)

        if forward:
            recvchunksize = recvnpts_rdir * mynpts2_ddir * mynpts_idir
        else:
            recvchunksize = mynpts1_rdir * sendnpts_ddir * mynpts_idir
        recvchunk = recvbuf[recvchunk_start:recvchunk_start + recvchunksize]
        recvchunks.append(recvchunk)
        recvchunk_start += recvchunksize

        if forward:
            dstchunk = dst.transpose(*dirs)[:, :, recvstart_rdir:recvstop_rdir]
        else:
            dstchunk = dst.transpose(*dirs)[:, sendstart_ddir:sendstop_ddir, :]
        copier = ChunkCopier(recvchunk, dstchunk)
        recv_chunk_copiers.append(copier)

    sendcounts = np.array([chunk.size for chunk in sendchunks], dtype=int)
    recvcounts = np.array([chunk.size for chunk in recvchunks], dtype=int)

    assert sendcounts.sum() == src.size
    assert recvcounts.sum() == dst.size
    senddispls = np.array([0] + list(np.cumsum(sendcounts))[:-1], dtype=int)
    recvdispls = np.array([0] + list(np.cumsum(recvcounts))[:-1], dtype=int)

    sendbuf = np.concatenate([sendchunk.ravel() for sendchunk in sendchunks])

    peer_comm.alltoallv(sendbuf, sendcounts, senddispls,
                        recvbuf, recvcounts, recvdispls)

    # Copy contiguous blocks of receive buffer back into precoded slices:
    for chunk_copier in recv_chunk_copiers:
        chunk_copier.copy()
    return dst


def get_compatible_grid_descriptor(gd, distribute_dir, reduce_dir):
    parsize2_c = list(gd.parsize_c)
    parsize2_c[reduce_dir] = 1
    parsize2_c[distribute_dir] = gd.parsize_c[reduce_dir] \
        * gd.parsize_c[distribute_dir]

    # Because of the way in which domains are assigned to ranks, some
    # redistributions cannot be represented on any grid descriptor
    # that uses the same communicator.  However we can create a
    # different one which assigns ranks in a manner corresponding to
    # a permutation of the axes, and there always exists a compatible
    # such communicator.

    # Probably there are two: a left-handed and a right-handed one
    # (i.e., positive or negative permutation of the axes).  It would
    # probably be logical to always choose a right-handed one.  Right
    # now the numbers correspond to whatever first was made to work
    # though!
    t = {(0, 1): (0, 1, 2),
         (0, 2): (0, 2, 1),
         (1, 0): (1, 0, 2),
         (1, 2): (0, 1, 2),
         (2, 1): (0, 2, 1),
         (2, 0): (1, 2, 0)}[(distribute_dir, reduce_dir)]

    ranks = np.arange(gd.comm.size).reshape(gd.parsize_c).transpose(*t).ravel()
    comm2 = gd.comm.new_communicator(ranks)
    gd2 = gd.new_descriptor(comm=comm2, parsize_c=parsize2_c)
    return gd2


class Domains:
    def __init__(self, domains_cp):
        self.domains_cp = domains_cp
        self.parsize_c = tuple(len(domains_cp[c]) - 1 for c in range(3))

    def get_global_shape(self):
        return tuple(self.domains_cp[c][-1]
                     - self.domains_cp[c][0] for c in range(3))

    def get_offset(self, parpos_c):
        offset_c = [self.domains_cp[c][parpos_c[c]] for c in range(3)]
        return np.array(offset_c)

    def get_box(self, parpos_c):
        parpos_c = np.array(parpos_c)
        offset_c = self.get_offset(parpos_c)
        nextoffset_c = self.get_offset(parpos_c + 1)
        return offset_c, nextoffset_c - offset_c

    def as_serial(self):
        return Domains([[self.domains_cp[c][0], self.domains_cp[c][-1]]
                        for c in range(3)])


def random_subcomm(comm, gen, size):
    allranks = np.arange(comm.size)
    wranks = gen.choice(allranks, size=size, replace=False)
    subcomm = comm.new_communicator(wranks)
    return wranks, subcomm


# For testing
class RandomDistribution:
    def __init__(self, comm, domains, gen):
        self.world = comm
        self.domains = domains
        size = np.prod(domains.parsize_c)
        wranks, subcomm = random_subcomm(comm, gen, size)
        self.comm = subcomm

        # We really need to know this information globally, even where
        # subcomm is None.
        comm.broadcast(wranks, 0)
        self.wranks = wranks
        self.subranks = {}
        for i, wrank in enumerate(wranks):
            self.subranks[wrank] = i

        domain_order = np.arange(size)
        gen.shuffle(domain_order)
        self.ranks3d = domain_order.reshape(domains.parsize_c)

        coords_iter = itertools.product(*[range(domains.parsize_c[c])
                                          for c in range(3)])
        self.coords_by_rank = list(coords_iter)

    def parpos2rank(self, parpos_c):
        return self.wranks[self.ranks3d[parpos_c]]

    def rank2parpos(self, rank):
        subrank = self.subranks.get(rank, None)
        if subrank is None:
            return None
        return self.coords_by_rank[subrank]

    def get_test_array(self, dtype=int):
        if self.comm is None:
            return np.zeros((0, 0, 0), dtype=dtype)

        parpos_c = self.rank2parpos(self.world.rank)
        offset_c, size_c = self.domains.get_box(parpos_c)
        arr = np.empty(size_c, dtype=dtype)
        arr[:] = self.comm.rank
        return arr


def general_redistribute(comm, domains1, domains2, rank2parpos1, rank2parpos2,
                         src_xg, dst_xg, behavior='overwrite'):
    """Redistribute array arbitrarily.

    Generally, this function redistributes part of an array into part
    of another array.  It is thus not necessarily one-to-one, but can
    be used to perform a redistribution to or from a larger, padded
    array, for example.

    """
    assert src_xg.dtype == dst_xg.dtype
    # Reshaping as arr.reshape(-1, *shape) fails when some dimensions are zero.
    # Also, make sure datatype is correct even for 0-length tuples:
    nx = np.prod(src_xg.shape[:-3], dtype=int)
    src_xg = src_xg.reshape(nx, *src_xg.shape[-3:])
    dst_xg = dst_xg.reshape(nx, *dst_xg.shape[-3:])
    assert src_xg.shape[0] == dst_xg.shape[0]
    assert behavior in ['overwrite', 'add']

    if not isinstance(domains1, Domains):
        domains1 = Domains(domains1)
    if not isinstance(domains2, Domains):
        domains2 = Domains(domains2)

    # Get global coords for local slice
    myparpos1_c = rank2parpos1(comm.rank)
    if myparpos1_c is not None:
        myoffset1_c, mysize1_c = domains1.get_box(myparpos1_c)
        assert np.all(mysize1_c == src_xg.shape[-3:]), \
            (mysize1_c, src_xg.shape[-3:])
    myparpos2_c = rank2parpos2(comm.rank)
    if myparpos2_c is not None:
        myoffset2_c, mysize2_c = domains2.get_box(myparpos2_c)
        assert np.all(mysize2_c == dst_xg.shape[-3:]), \
            (mysize2_c, dst_xg.shape[-3:])

    sendranks = []
    recvranks = []
    sendchunks = []
    recvchunks = []

    sendcounts, recvcounts, senddispls, recvdispls = np.zeros((4, comm.size),
                                                              dtype=int)

    # The plan is to loop over all ranks, then figure out:
    #
    #   1) What do we have to send to that rank
    #   2) What are we going to receive from that rank
    #
    # Some ranks may not hold any data before, or after, or both.

    def _intersection(myoffset_c, mysize_c, offset_c, size_c, arr_g):
        # Get intersection of two rectangles, given as offset and size
        # in global coordinates.  Returns None if no intersection,
        # else the appropriate slice of the local array
        start_c = np.max([myoffset_c, offset_c], axis=0)
        stop_c = np.min([myoffset_c + mysize_c, offset_c + size_c], axis=0)
        if (stop_c - start_c > 0).all():
            # Reduce to local array coordinates:
            start_c -= myoffset_c
            stop_c -= myoffset_c
            return arr_g[:, start_c[0]:stop_c[0], start_c[1]:stop_c[1],
                            start_c[2]:stop_c[2]]
        else:
            return None

    def get_sendchunk(offset_c, size_c):
        return _intersection(myoffset1_c, mysize1_c, offset_c, size_c, src_xg)

    def get_recvchunk(offset_c, size_c):
        return _intersection(myoffset2_c, mysize2_c, offset_c, size_c, dst_xg)

    nsendtotal = 0
    nrecvtotal = 0
    for rank in range(comm.size):
        # Proceed only if we have something to send
        if myparpos1_c is not None:
            parpos2_c = rank2parpos2(rank)
            # Proceed only if other rank is going to receive something
            if parpos2_c is not None:
                offset2_c, size2_c = domains2.get_box(parpos2_c)
                sendchunk = get_sendchunk(offset2_c, size2_c)
                if sendchunk is not None:
                    sendcounts[rank] = sendchunk.size
                    senddispls[rank] = nsendtotal
                    nsendtotal += sendchunk.size
                    sendchunks.append(sendchunk)
                    sendranks.append(rank)

        # Proceed only if we are going to receive something
        if myparpos2_c is not None:
            parpos1_c = rank2parpos1(rank)
            # Proceed only if other rank has something to send
            if parpos1_c is not None:
                offset1_c, size1_c = domains1.get_box(parpos1_c)
                recvchunk = get_recvchunk(offset1_c, size1_c)
                if recvchunk is not None:
                    recvcounts[rank] = recvchunk.size
                    recvdispls[rank] = nrecvtotal
                    nrecvtotal += recvchunk.size
                    recvchunks.append(recvchunk)
                    recvranks.append(rank)

    # MPI wants contiguous buffers; who are we to argue:
    sendbuf = np.empty(nsendtotal, src_xg.dtype)
    recvbuf = np.empty(nrecvtotal, src_xg.dtype)

    # Copy non-contiguous slices into contiguous sendbuffer:
    for sendrank, sendchunk in zip(sendranks, sendchunks):
        nstart = senddispls[sendrank]
        nstop = nstart + sendcounts[sendrank]
        sendbuf[nstart:nstop] = sendchunk.ravel()

    # Finally!
    comm.alltoallv(sendbuf, sendcounts, senddispls,
                   recvbuf, recvcounts, recvdispls)

    # Now copy from the recvbuffer into the actual destination array:
    for recvrank, recvchunk in zip(recvranks, recvchunks):
        nstart = recvdispls[recvrank]
        nstop = nstart + recvcounts[recvrank]
        buf = recvbuf[nstart:nstop]
        if behavior == 'overwrite':
            recvchunk.flat[:] = buf
        elif behavior == 'add':
            recvchunk.flat[:] += buf

def test_general_redistribute():
    from gpaw.mpi import world

    domains1 = Domains([[0, 1],
                        [1, 3, 5, 6],
                        [0, 5, 9]])
    domains2 = Domains([[0, 1],
                        [0, 2, 4, 7, 10],
                        [2, 4, 6, 7]])

    serial = domains2.as_serial()

    gen = np.random.RandomState(42)

    dist1 = RandomDistribution(world, domains1, gen)
    dist2 = RandomDistribution(world, domains2, gen)

    arr1 = dist1.get_test_array()
    arr2 = dist2.get_test_array()
    print('shapes', arr1.shape, arr2.shape)
    arr2[:] = -1

    general_redistribute(world, domains1, domains2,
                         dist1.rank2parpos, dist2.rank2parpos,
                         arr1, arr2)

    dist_serial = RandomDistribution(world, serial, gen)
    arr3 = dist_serial.get_test_array()
    general_redistribute(world, domains2, serial, dist2.rank2parpos,
                         dist_serial.rank2parpos, arr2, arr3)
    print(arr3)


#if __name__ == '__main__':
#    main()


def playground():
    np.set_printoptions(linewidth=176)
    #N_c = [4, 7, 9]
    N_c = [4, 4, 2]

    pbc_c = (1, 1, 1)

    # 210
    distribute_dir = 1
    reduce_dir = 0

    parsize_c = (2, 2, 2)
    parsize2_c = list(parsize_c)
    parsize2_c[reduce_dir] = 1
    parsize2_c[distribute_dir] *= parsize_c[reduce_dir]
    assert np.prod(parsize2_c) == np.prod(parsize_c)

    gd = GridDescriptor(N_c=N_c, pbc_c=pbc_c, cell_cv=0.2 * np.array(N_c),
                        parsize_c=parsize_c)

    gd2 = get_compatible_grid_descriptor(gd, distribute_dir, reduce_dir)

    src = gd.zeros(dtype=complex)
    src[:] = gd.comm.rank

    src_global = gd.collect(src)
    if gd.comm.rank == 0:
        ind = np.indices(src_global.shape)
        src_global += 1j * (ind[0] / 10. + ind[1] / 100. + ind[2] / 1000.)
        #src_global[1] += 0.5j
        print('GLOBAL ARRAY', src_global.shape)
        print(src_global.squeeze())
    gd.distribute(src_global, src)
    goal = gd2.zeros(dtype=float)
    goal[:] = gd.comm.rank # get_members()[gd2.comm.rank]
    goal_global = gd2.collect(goal)
    if gd.comm.rank == 0:
        print('GOAL GLOBAL')
        print(goal_global.squeeze())
    gd.comm.barrier()
    #return

    recvbuf = redistribute(gd, gd2, src, distribute_dir, reduce_dir,
                           operation='forth')
    recvbuf_master = gd2.collect(recvbuf)
    if gd2.comm.rank == 0:
        print('RECV')
        print(recvbuf_master)
        err = src_global - recvbuf_master
        print('MAXERR', np.abs(err).max())

    hopefully_orig = redistribute(gd, gd2, recvbuf, distribute_dir, reduce_dir,
                                  operation='back')
    tmp = gd.collect(hopefully_orig)
    if gd.comm.rank == 0:
        print('FINALLY')
        print(tmp)
        err2 = src_global - tmp
        print('MAXERR', np.abs(err2).max())


def test(N_c, gd, gd2, reduce_dir, distribute_dir, verbose=True):
    src = gd.zeros(dtype=complex)
    src[:] = gd.comm.rank

    # if gd.comm.rank == 0:
    #     print(gd)
    #     print('hmmm', gd, gd2)

    src_global = gd.collect(src)
    if gd.comm.rank == 0:
        ind = np.indices(src_global.shape)
        src_global += 1j * (ind[0] / 10. + ind[1] / 100. + ind[2] / 1000.)
        #src_global[1] += 0.5j
        if verbose:
            print('GLOBAL ARRAY', src_global.shape)
            print(src_global)
    gd.distribute(src_global, src)
    goal = gd2.zeros(dtype=float)
    goal[:] = gd2.comm.rank
    goal_global = gd2.collect(goal)
    if gd.comm.rank == 0 and verbose:
        print('GOAL GLOBAL')
        print(goal_global)
    gd.comm.barrier()

    recvbuf = redistribute(gd, gd2, src, distribute_dir, reduce_dir,
                           operation='forth')
    recvbuf_master = gd2.collect(recvbuf)
    #if np.all(N_c == [10, 16, 24]):
    #    recvbuf_master[5,8,3] = 7
    maxerr = 0.0
    if gd2.comm.rank == 0:
        #if N_c[0] == 7:
        #    recvbuf_master[5, 0, 0] = 7
        #recvbuf_master[0,0,0] = 7
        err = src_global - recvbuf_master
        maxerr = np.abs(err).max()
        if verbose:
            print('RECV FORTH')
            print(recvbuf_master)
            print('MAXERR', maxerr)
    maxerr = gd.comm.sum(maxerr)
    assert maxerr == 0.0, 'bad values after distribute "forth"'

    recvbuf2 = redistribute(gd, gd2, recvbuf, distribute_dir, reduce_dir,
                            operation='back')

    final_err = gd.comm.sum(np.abs(src - recvbuf2).max())
    assert final_err == 0.0, 'bad values after distribute "back"'


def rigorous_testing():
    from itertools import product, permutations, cycle
    from gpaw.mpi import world
    gridpointcounts = [1, 2, 10, 21]
    cpucounts = np.arange(1, world.size + 1)
    pbc = cycle(product([0, 1], [0, 1], [0, 1]))

    # This yields all possible parallelizations!
    for parsize_c in product(cpucounts, cpucounts, cpucounts):
        if np.prod(parsize_c) != world.size:
            continue

        # All possible grid point counts
        for N_c in product(gridpointcounts, gridpointcounts, gridpointcounts):

            # We simply can't be bothered to also do all possible
            # combinations with PBCs.  Trying every possible set of
            # boundary conditions at least ones should be quite fine
            # enough.
            pbc_c = next(pbc)
            for dirs in permutations([0, 1, 2]):
                independent_dir, distribute_dir, reduce_dir = dirs

                parsize2_c = list(parsize_c)
                parsize2_c[reduce_dir] = 1
                parsize2_c[distribute_dir] *= parsize_c[reduce_dir]
                parsize2_c = tuple(parsize2_c)
                assert np.prod(parsize2_c) == np.prod(parsize_c)

                try:
                    gd = GridDescriptor(N_c=N_c, pbc_c=pbc_c,
                                        cell_cv=0.2 * np.array(N_c),
                                        parsize_c=parsize_c)
                    gd2 = get_compatible_grid_descriptor(gd, distribute_dir,
                                                         reduce_dir)

                    #gd2 = gd.new_descriptor(parsize_c=parsize2_c)
                except ValueError:  # Skip illegal distributions
                    continue

                if gd.comm.rank == 1:
                    #print(gd, gd2)
                    print('N_c=%s[%s] redist %s -> %s [ind=%d dist=%d red=%d]'
                          % (N_c, pbc_c, parsize_c, parsize2_c,
                             independent_dir, distribute_dir, reduce_dir))
                gd.comm.barrier()
                test(N_c, gd, gd2, reduce_dir, distribute_dir,
                     verbose=False)


if __name__ == '__main__':
    test_general_redistribute()
    #playground()
    #rigorous_testing()
