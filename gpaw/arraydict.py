import numpy as np


class DefaultKeyMapping:
    def a2key(self, a):
        return a

    def key2a(self, key):
        return key


class TableKeyMapping:
    def __init__(self, a2key_a, key2a_k):
        self.a2key_a = a2key_a
        self.key2a_k = key2a_k

    def key2a(self, key):
        return self.key2a_k[key]

    def a2key(self, a):
        return self.a2key_a[a]


class ArrayDict:
    """Distributed dictionary of fixed-size, fixed-dtype arrays.

    Implements a map [0, ..., N] -> [A0, ..., AN]

    Elements are initialized as empty numpy arrays.

    Unlike a normal dictionary, this class implements a strict loop ordering
    which is consistent with that of the underlying atom partition."""
    def __init__(self, partition, shapes_a, dtype=float, d=None,
                 keymap=None):
        self.partition = partition
        if callable(shapes_a):
            shapes_a = [shapes_a(a) for a in range(self.partition.natoms)]
        self.shapes_a = shapes_a  # global
        assert len(shapes_a) == partition.natoms
        self.dtype = dtype
        self.data = {}

        # This will be a terrible hack to make it easier to customize
        # the "keys".  Normally keys are 0...N and correspond to
        # rank_a of the AtomPartition.  But the keymap allows the user
        # to pass around a one-to-one mapping between the "physical"
        # keys 0...N and some other objects that the user may fancy.
        if keymap is None:
            keymap = DefaultKeyMapping()
        self.keymap = keymap

        if d is None:
            for a in partition.my_indices:
                self[a] = np.empty(self.shapes_a[a], dtype=dtype)
        else:
            self.update(d)
        self.check_consistency()

    # copy() is dangerous since redistributions write back
    # into arrays, and redistribution of a copy could lead to bugs
    # if the other copy changes.
    #
    # Let's just use deepcopy to stay out of trouble.  Then the
    # dictionary copy() is not invoked confusingly.
    #
    #  -askhl
    def copy(self):
        copy = ArrayDict(self.partition, self.shapes_a, dtype=self.dtype,
                         keymap=self.keymap)
        for a in self:
            copy[a][:] = self[a]
        return copy

    def update(self, d):
        self.data.update(d)
        self.check_consistency()

    def __getitem__(self, a):
        value = self.data[a]
        assert value.shape == self.shapes_a[a]
        assert value.dtype == self.dtype
        return value

    def __setitem__(self, a, value):
        assert value.shape == self.shapes_a[a], \
            'defined shape %s vs new %s' % (self.shapes_a[a], value.shape)
        assert value.dtype == self.dtype
        self.data[a] = value

    def redistribute(self, partition):
        """Redistribute according to specified partition."""
        def get_empty(a):
            return np.empty(self.shapes_a[a], self.dtype)

        self.partition.redistribute(partition, self, get_empty)
        self.partition = partition  # Better with immutable partition?
        self.check_consistency()

    def check_consistency(self):
        k1 = set(self.partition.my_indices)
        k2 = set(self.data.keys())
        assert k1 == k2, 'Required keys %s different from actual %s' % (k1, k2)
        for a, array in self.items():
            assert array.dtype == self.dtype
            assert array.shape == self.shapes_a[a], \
                'array shape %s vs specified shape %s' % (array.shape,
                                                          self.shapes_a[a])

    def toarray(self, axis=None):
        # We could also implement it as a contiguous buffer.
        if len(self) == 0:
            # XXXXXX how should we deal with globally or locally empty arrays?
            # This will probably lead to bugs unless we get all the
            # dimensions right.
            return np.empty(0, self.dtype)
        if axis is None:
            return np.concatenate([self[a].ravel()
                                   for a in self.partition.my_indices])
        else:
            # XXX self[a].shape must all be consistent except along axis
            return np.concatenate([self[a] for a in self.partition.my_indices],
                                  axis=axis)

    def fromarray(self, data):
        assert data.dtype == self.dtype
        M1 = 0
        for a in self.partition.my_indices:
            M2 = M1 + np.prod(self.shapes_a[a])
            self[a].ravel()[:] = data[M1:M2]
            M1 = M2

    def redistribute_and_broadcast(self, dist_comm, dup_comm):
        # Data exists on self which is a "nice" distribution but now
        # we want it on sub_partition which has a smaller communicator
        # whose parent is self.comm.
        #
        # We want our own data replicated on each

        # XXX direct comparison of communicators are unsafe as we do not use
        # MPI_Comm_compare

        #assert subpartition.comm.parent == self.partition.comm
        from gpaw.utilities.partition import AtomPartition

        newrank_a = self.partition.rank_a % dist_comm.size
        masters_only_partition = AtomPartition(self.partition.comm, newrank_a)
        dst_partition = AtomPartition(dist_comm, newrank_a)
        copy = self.copy()
        copy.redistribute(masters_only_partition)

        dst = ArrayDict(dst_partition, self.shapes_a, dtype=self.dtype,
                        keymap=self.keymap)
        data = dst.toarray()
        if dup_comm.rank == 0:
            data0 = copy.toarray()
            data[:] = data0
        dup_comm.broadcast(data, 0)
        dst.fromarray(data)
        return dst

    # These functions enforce the same ordering as self.partition
    # when looping.
    def keys(self):
        return self.partition.my_indices

    def __iter__(self):
        for key in self.partition.my_indices:
            yield key

    def values(self):
        return [self[key] for key in self]

    def items(self):
        for key in self:
            yield key, self[key]

    def __repr__(self):
        tokens = []
        for key in sorted(self.keys()):
            shapestr = 'x'.join(map(str, self.shapes_a[key]))
            tokens.append('%s:%s' % (self.keymap.a2key(key), shapestr))
        text = ', '.join(tokens)
        return '%s@rank%d/%d {%s}' % (self.__class__.__name__,
                                      self.partition.comm.rank,
                                      self.partition.comm.size,
                                      text)

    def get(self, a):
        return self.data.get(a)

    def __len__(self):
        return len(self.data)
