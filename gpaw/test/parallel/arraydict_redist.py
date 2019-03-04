import numpy as np
from gpaw.mpi import world
from gpaw.utilities.partition import AtomPartition
from gpaw.arraydict import ArrayDict

gen = np.random.RandomState(0)


def shape(a):
    return (a, a // 2)  # Shapes: (0, 0), (1, 0), (2, 1), ...

natoms = 33

if world.size == 1:
    rank_a = np.zeros(natoms, int)
else:
    # When on more than 2 cores, make sure that at least one core
    # (rank=0) has zero entries:
    lower = 0 if world.size == 2 else 1
    rank_a = gen.randint(lower, world.size, natoms)
assert (rank_a < world.size).all()

serial = AtomPartition(world, np.zeros(natoms, int))
partition = AtomPartition(world, rank_a)
even_partition = partition.as_even_partition()


def check(atomdict, title):
    if world.rank == world.size // 2 or world.rank == 0:
        print('rank %d %s: %s' % (world.rank, title.rjust(10), atomdict))

    # Create a normal, "well-behaved" dict against which to test arraydict.
    ref = dict(atomdict)
    #print atomdict
    assert set(atomdict.keys()) == set(ref.keys())  # check keys()
    for a in atomdict:  # check __iter__, __getitem__
        #print ref[a].shape, atomdict[a].shape #ref[a].shape, atomdict[a].shape
        #print ref[a], atomdict[a]
        #assert 1#ref[a] is atomdict[a]
        assert ref[a] is atomdict[a]
    values = atomdict.values()
    for i, key in enumerate(atomdict):
        # AtomDict guarantees fixed ordering of keys.  Check that
        # values() ordering is consistent with loop ordering:
        assert values[i] is atomdict[key]

    items = list(atomdict.items())

    for i, (key, item) in enumerate(atomdict.items()):
        assert item is atomdict[key]
        assert item is ref[key]
        assert items[i][0] == key
        assert items[i][1] is item

    # Hopefully this should verify all the complicated stuff

ad = ArrayDict(partition, shape, float)
for key in ad:
    ad[key][:] = key
array0 = ad.toarray()

ad0 = dict(ad)
check(ad, 'new')
ad.redistribute(even_partition)
array1 = ad.toarray()
if world.rank > 1:
    assert array1.shape != array0.shape
check(ad, 'even')
ad.redistribute(serial)
check(ad, 'serial')
ad.redistribute(partition)
check(ad, 'back')

array2 = ad.toarray()
assert (array0 == array2).all()
