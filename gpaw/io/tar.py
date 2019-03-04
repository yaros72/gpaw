import numbers
import tarfile
import xml.sax

import numpy as np

from gpaw.mpi import broadcast as mpi_broadcast
from gpaw.mpi import world

intsize = 4
floatsize = np.array([1], float).itemsize
complexsize = np.array([1], complex).itemsize
itemsizes = {'int': intsize, 'float': floatsize, 'complex': complexsize}

    
class FileReference:
    """Common base class for having reference to a file. The actual I/O
       classes implementing the referencing should be inherited from
       this class."""

    def __init__(self):
        raise NotImplementedError('Should be implemented in derived classes')

    def __len__(self):
        raise NotImplementedError('Should be implemented in derived classes')

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self):
        raise NotImplementedError('Should be implemented in derived classes')

    def __array__(self):
        return self[::]


def open(filename, mode='r', comm=world):
    if not filename.endswith('.gpw'):
        filename += '.gpw'

    assert mode == 'r'
    return Reader(filename, comm)


class Reader(xml.sax.handler.ContentHandler):
    def __init__(self, name, comm=world):
        self.comm = comm  # used for broadcasting replicated data
        self.master = (self.comm.rank == 0)
        self.dims = {}
        self.shapes = {}
        self.dtypes = {}
        self.parameters = {}
        xml.sax.handler.ContentHandler.__init__(self)
        self.tar = tarfile.open(name, 'r')
        f = self.tar.extractfile('info.xml')
        xml.sax.parse(f, self)

    def startElement(self, tag, attrs):
        if tag == 'gpaw_io':
            self.byteswap = ((attrs['endianness'] == 'little') !=
                             np.little_endian)
        elif tag == 'array':
            name = attrs['name']
            self.dtypes[name] = attrs['type']
            self.shapes[name] = []
            self.name = name
        elif tag == 'dimension':
            n = int(attrs['length'])
            self.shapes[self.name].append(n)
            self.dims[attrs['name']] = n
        else:
            assert tag == 'parameter'
            try:
                value = eval(attrs['value'], {})
            except (SyntaxError, NameError):
                value = str(attrs['value'])
            self.parameters[attrs['name']] = value

    def dimension(self, name):
        return self.dims[name]
    
    def __getitem__(self, name):
        return self.parameters[name]

    def has_array(self, name):
        return name in self.shapes
    
    def get(self, name, *indices, **kwargs):
        broadcast = kwargs.pop('broadcast', False)
        if self.master or not broadcast:
            fileobj, shape, size, dtype = self.get_file_object(name, indices)
            array = np.fromstring(fileobj.read(size), dtype)
            if self.byteswap:
                array = array.byteswap()
            if dtype == np.int32:
                array = np.asarray(array, int)
            array.shape = shape
            if shape == ():
                array = array.item()
        else:
            array = None

        if broadcast:
            array = mpi_broadcast(array, 0, self.comm)
        return array
    
    def get_reference(self, name, indices, length=None):
        fileobj, shape, size, dtype = self.get_file_object(name, indices)
        assert dtype != np.int32
        return TarFileReference(fileobj, shape, dtype, self.byteswap, length)
    
    def get_file_object(self, name, indices):
        dtype, type, itemsize = self.get_data_type(name)
        fileobj = self.tar.extractfile(name)
        n = len(indices)
        shape = self.shapes[name]
        size = itemsize * np.prod(shape[n:], dtype=int)
        offset = 0
        stride = size
        for i in range(n - 1, -1, -1):
            offset += indices[i] * stride
            stride *= shape[i]
        fileobj.seek(offset)
        return fileobj, shape[n:], size, dtype

    def get_data_type(self, name):
        type = self.dtypes[name]
        dtype = np.dtype({'int': np.int32,
                          'float': float,
                          'complex': complex}[type])
        return dtype, type, dtype.itemsize

    def get_parameters(self):
        return self.parameters

    def close(self):
        self.tar.close()


class TarFileReference(FileReference):
    def __init__(self, fileobj, shape, dtype, byteswap, length):
        self.fileobj = fileobj
        self.shape = tuple(shape)
        self.dtype = dtype
        self.itemsize = dtype.itemsize
        self.byteswap = byteswap
        self.offset = fileobj.tell()
        self.length = length

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, indices):
        if isinstance(indices, slice):
            start, stop, step = indices.indices(len(self))
            if start != 0 or step != 1 or stop != len(self):
                raise NotImplementedError('You can only slice a TarReference '
                                          'with [:] or [int]')
            else:
                indices = ()
        elif isinstance(indices, numbers.Integral):
            indices = (indices,)
        else:  # Probably tuple or ellipsis
            raise NotImplementedError('You can only slice a TarReference '
                                      'with [:] or [int]')
            
        n = len(indices)

        size = np.prod(self.shape[n:], dtype=int) * self.itemsize
        offset = self.offset
        stride = size
        for i in range(n - 1, -1, -1):
            offset += indices[i] * stride
            stride *= self.shape[i]
        self.fileobj.seek(offset)
        array = np.fromstring(self.fileobj.read(size), self.dtype)
        if self.byteswap:
            array = array.byteswap()
        array.shape = self.shape[n:]
        if self.length:
            array = array[..., :self.length].copy()
        return array
