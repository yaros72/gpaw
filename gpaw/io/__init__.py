def Reader(filename):
    import ase.io.ulm as ulm
    try:
        return ulm.Reader(filename)
    except ulm.InvalidULMFileError:
        pass
    from gpaw.io.old import wrap_old_gpw_reader
    return wrap_old_gpw_reader(filename)


def Writer(filename, world, mode='w', tag='GPAW'):
    import ase.io.ulm as ulm
    if world.rank == 0:
        return ulm.Writer(filename, mode=mode, tag=tag)
    return ulm.DummyWriter()
