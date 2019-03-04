import numpy as np
from gpaw.arraydict import ArrayDict


def to_parent_comm(partition):
    # XXX assume communicator is strided, i.e. regular.
    # This actually imposes implicit limitations on things, but is not
    # "likely" to cause trouble with the usual communicators, i.e.
    # for gd/kd/bd.
    parent = partition.comm.parent
    if parent is None:
        # This should not ordinarily be necessary, but when running with
        # AtomPAW, it is.  So let's stay out of trouble.
        return partition

    members = partition.comm.get_members()
    parent_rank_a = members[partition.rank_a]

    # XXXX we hope and pray that our communicator is "equivalent" to
    # that which includes parent's rank0.
    assert min(members) == members[0]
    parent_rank_a -= members[0]  # yuckkk
    return AtomPartition(parent, parent_rank_a,
                         name='parent-%s' % partition.name)


class AtomicMatrixDistributor:
    """Class to distribute atomic dictionaries like dH_asp and D_asp."""
    def __init__(self, atom_partition, broadcast_comm,
                 work_partition=None):
        # Assumptions on communicators are as follows.
        #
        # atom_partition represents standard domain decomposition, and
        # broadcast_comm are the corresponding kpt/band communicators
        # together encompassing wfs.world.
        #
        # Initially, dH_asp are distributed over domains according to the
        # physical location of each atom, but duplicated across band
        # and k-point communicators.
        #
        # The idea is to transfer dH_asp so they are distributed equally
        # among all ranks on wfs.world, and back, when necessary.
        self.broadcast_comm = broadcast_comm

        self.grid_partition = atom_partition
        self.grid_unique_partition = to_parent_comm(self.grid_partition)

        # This represents a full distribution across grid, kpt, and band.
        if work_partition is None:
            work_partition = self.grid_unique_partition.as_even_partition()
        self.work_partition = work_partition

    def distribute(self, D_asp):
        # Right now the D are duplicated across the band/kpt comms.
        # Here we pick out a set of unique D.  With duplicates out,
        # we can redistribute one-to-one to the larger work_partition.
        # assert D_asp.partition == self.grid_partition

        Ddist_asp = ArrayDict(self.grid_unique_partition, D_asp.shapes_a,
                              dtype=D_asp.dtype)

        if self.broadcast_comm.rank != 0:
            assert len(Ddist_asp) == 0
        for a in Ddist_asp:
            Ddist_asp[a] = D_asp[a]
        Ddist_asp.redistribute(self.work_partition)
        return Ddist_asp

    def collect(self, dHdist_asp):
        # We have an array on work_partition.  We want first to
        # collect it on grid_unique_partition, the broadcast it
        # to grid_partition.

        # First receive one-to-one from everywhere.
        # assert dHdist_asp.partition == self.work_partition
        dHdist_asp = dHdist_asp.copy()
        dHdist_asp.redistribute(self.grid_unique_partition)

        dH_asp = ArrayDict(self.grid_partition, dHdist_asp.shapes_a,
                           dtype=dHdist_asp.dtype)
        if self.broadcast_comm.rank == 0:
            buf = dHdist_asp.toarray()
            assert not np.isnan(buf).any()
        else:
            buf = dH_asp.toarray()
            buf[:] = np.nan  # Let's be careful for now like --debug mode
        self.broadcast_comm.broadcast(buf, 0)
        assert not np.isnan(buf).any()
        dH_asp.fromarray(buf)
        return dH_asp


class EvenPartitioning:
    """Represents an even partitioning of N elements over a communicator.

    For example N=17 and comm.size=5 will result in this distribution:

     * rank 0 has 3 local elements: 0, 1, 2
     * rank 1 has 3 local elements: 3, 4, 5
     * rank 2 has 3 local elements: 6, 7, 8
     * rank 3 has 4 local elements: 9, 10, 11, 12
     * rank 4 has 4 local elements: 13, 14, 15, 16

    This class uses only the 'rank' and 'size' communicator attributes."""
    def __init__(self, comm, N):
        # Conventions:
        #  n, N: local/global size
        #  i, I: local/global index
        self.comm = comm
        self.N = N
        self.nlong = -(-N // comm.size)  # size of a long slice
        self.nshort = N // comm.size  # size of a short slice
        self.longcount = N % comm.size  # number of ranks with a long slice
        self.shortcount = comm.size - self.longcount  # ranks with short slice

    def nlocal(self, rank=None):
        """Get the number of locally stored elements."""
        if rank is None:
            rank = self.comm.rank
        if rank < self.shortcount:
            return self.nshort
        else:
            return self.nlong

    def minmax(self, rank=None):
        """Get the minimum and maximum index of elements stored locally."""
        if rank is None:
            rank = self.comm.rank
        I1 = self.nshort * rank
        if rank < self.shortcount:
            I2 = I1 + self.nshort
        else:
            I1 += rank - self.shortcount
            I2 = I1 + self.nlong
        return I1, I2

    def slice(self, rank=None):
        """Get the list of indices of locally stored elements."""
        I1, I2 = self.minmax(rank=rank)
        return np.arange(I1, I2)

    def global2local(self, I):
        """Get a tuple (rank, local index) from global index I."""
        nIshort = self.nshort * self.shortcount
        if I < nIshort:
            return I // self.nshort, I % self.nshort
        else:
            Ioffset = I - nIshort
            return (self.shortcount + Ioffset // self.nlong,
                    Ioffset % self.nlong)

    def local2global(self, i, rank=None):
        """Get global index I corresponding to local index i on rank."""
        if rank is None:
            rank = self.comm.rank
        return rank * self.nshort + max(rank - self.shortcount, 0) + i

    def as_atom_partition(self, strided=False, name='unnamed-even'):
        rank_a = [self.global2local(i)[0] for i in range(self.N)]
        if strided:
            rank_a = np.arange(self.comm.size).repeat(self.nlong)
            rank_a = rank_a.reshape(self.comm.size, -1).T.ravel()
            rank_a = rank_a[self.shortcount:].copy()
        return AtomPartition(self.comm, rank_a, name=name)

    def get_description(self):
        lines = []
        for a in range(self.comm.size):
            elements = ', '.join(map(str, self.slice(a)))
            line = 'rank %d has %d local elements: %s' % (a, self.nlocal(a),
                                                          elements)
            lines.append(line)
        return '\n'.join(lines)


# Interface for things that can be redistributed with general_redistribute
class Redistributable:
    def get_recvbuffer(self, a):
        raise NotImplementedError

    def get_sendbuffer(self, a):
        raise NotImplementedError

    def assign(self, a):
        raise NotImplementedError


# Let's keep this as an independent function for now in case we change the
# classes 5 times, like we do
def general_redistribute(comm, src_rank_a, dst_rank_a, redistributable):
    # To do: it should be possible to specify duplication to several ranks
    # But how is this done best?
    requests = []
    flags = (src_rank_a != dst_rank_a)
    my_incoming_atom_indices = np.argwhere(np.bitwise_and(flags, \
        dst_rank_a == comm.rank)).ravel()
    my_outgoing_atom_indices = np.argwhere(np.bitwise_and(flags, \
        src_rank_a == comm.rank)).ravel()

    for a in my_incoming_atom_indices:
        # Get matrix from old domain:
        buf = redistributable.get_recvbuffer(a)
        requests.append(comm.receive(buf, src_rank_a[a], tag=a, block=False))
        # These arrays are not supposed to pointers into a larger,
        # contiguous buffer, so we should make a copy - except we
        # must wait until we have completed the send/receiving
        # into them, so we will do it a few lines down.
        redistributable.assign(a, buf)

    for a in my_outgoing_atom_indices:
        # Send matrix to new domain:
        buf = redistributable.get_sendbuffer(a)
        requests.append(comm.send(buf, dst_rank_a[a], tag=a, block=False))

    comm.waitall(requests)


class AtomPartition:
    """Represents atoms distributed on a standard grid descriptor."""
    def __init__(self, comm, rank_a, name='unnamed'):
        self.comm = comm
        self.rank_a = np.array(rank_a)
        self.my_indices = self.get_indices(comm.rank)
        self.natoms = len(rank_a)
        self.name = name

    def __eq__(self, other):
        try:
            comm = other.comm
            rank_a = other.rank_a
        except AttributeError:
            return False
        return (self.comm.compare(comm) in ['ident', 'congruent']
                and np.array_equal(self.rank_a, rank_a))

    def __ne__(self, other):
        return not self == other

    def as_serial(self):
        return AtomPartition(self.comm, np.zeros(self.natoms, int),
                             name='%s-serial' % self.name)

    def get_indices(self, rank):
        return np.where(self.rank_a == rank)[0]

    def as_even_partition(self):
        even_part = EvenPartitioning(self.comm, len(self.rank_a))
        return even_part.as_atom_partition()

    def redistribute(self, new_partition, atomdict_ax, get_empty):
        # XXX we the two communicators to be equal according to
        # some proper criterion like MPI_Comm_compare -> MPI_IDENT.
        # But that is not implemented, so we don't.
        if self.comm.compare(new_partition.comm) not in ['ident',
                                                         'congruent']:
            msg = ('Incompatible partitions %s --> %s.  '
                   'Communicators must be at least congruent'
                   % (self, new_partition))
            raise ValueError(msg)

        # atomdict_ax may be a dictionary or a list of dictionaries

        has_many = not hasattr(atomdict_ax, 'items')
        if has_many:
            class Redist:
                def get_recvbuffer(self, a):
                    return get_empty(a)

                def assign(self, a, b_x):
                    for u, d_ax in enumerate(atomdict_ax):
                        assert a not in d_ax
                        atomdict_ax[u][a] = b_x[u]

                def get_sendbuffer(self, a):
                    return np.array([d_ax.data.pop(a) for d_ax in atomdict_ax])
        else:
            class Redist:
                def get_recvbuffer(self, a):
                    return get_empty(a)

                def assign(self, a, b_x):
                    assert a not in atomdict_ax
                    atomdict_ax[a] = b_x

                def get_sendbuffer(self, a):
                    return atomdict_ax.data.pop(a)

        try:
            general_redistribute(self.comm, self.rank_a,
                                 new_partition.rank_a, Redist())
        except ValueError as err:
            raise ValueError('redistribute %s --> %s: %s'
                             % (self, new_partition, err))
        if isinstance(atomdict_ax, ArrayDict):
            atomdict_ax.partition = new_partition # XXX
            atomdict_ax.check_consistency()

    def __repr__(self):
        indextext = ', '.join(map(str, self.my_indices))
        return ('%s %s@rank%d/%d (%d/%d): [%s]'
                % (self.__class__.__name__, self.name, self.comm.rank,
                   self.comm.size, len(self.my_indices), self.natoms,
                   indextext))

    def arraydict(self, shapes, dtype=float):
        return ArrayDict(self, shapes, dtype)
