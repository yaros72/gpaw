import numpy as np

from gpaw.matrix import Matrix, create_distribution


class MatrixInFile:
    def __init__(self, M, N, dtype, data, dist):
        self.shape = (M, N)
        self.dtype = dtype
        self.array = data  # pointer to data in a file
        self.dist = create_distribution(M, N, *dist)


class ArrayWaveFunctions:
    def __init__(self, M, N, dtype, data, dist, collinear):
        self.collinear = collinear
        if not collinear:
            N *= 2
        if data is None or isinstance(data, np.ndarray):
            self.matrix = Matrix(M, N, dtype, data, dist)
            self.in_memory = True
        else:
            self.matrix = MatrixInFile(M, N, dtype, data, dist)
            self.in_memory = False
        self.comm = None
        self.dtype = self.matrix.dtype

    def __len__(self):
        return len(self.matrix)

    def multiply(self, alpha, opa, b, opb, beta, c, symmetric):
        self.matrix.multiply(alpha, opa, b.matrix, opb, beta, c, symmetric)
        if opa == 'N' and self.comm:
            if self.comm.size > 1:
                c.comm = self.comm
                c.state = 'a sum is needed'
            assert opb in 'TC' and b.comm is self.comm

    def matrix_elements(self, other=None, out=None, symmetric=False, cc=False,
                        operator=None, result=None, serial=False):
        if out is None:
            out = Matrix(len(self), len(other or self), dtype=self.dtype,
                         dist=(self.matrix.dist.comm,
                               self.matrix.dist.rows,
                               self.matrix.dist.columns))
        if other is None or isinstance(other, ArrayWaveFunctions):
            assert cc
            if other is None:
                assert symmetric
                operate_and_multiply(self, self.dv, out, operator, result)
            elif not serial:
                assert not symmetric
                operate_and_multiply_not_symmetric(self, self.dv, out,
                                                   other)
            else:
                self.multiply(self.dv, 'N', other, 'C', 0.0, out, symmetric)
        else:
            assert not cc
            P_ani = {a: P_ni for a, P_ni in out.items()}
            other.integrate(self.array, P_ani, self.kpt)
        return out

    def add(self, lfc, coefs):
        lfc.add(self.array, dict(coefs.items()), self.kpt)

    def apply(self, func, out=None):
        out = out or self.new()
        func(self.array, out.array)
        return out

    def __setitem__(self, i, x):
        x.eval(self.matrix)

    def __iadd__(self, other):
        other.eval(self.matrix, 1.0)
        return self

    def eval(self, matrix):
        matrix.array[:] = self.matrix.array

    def read_from_file(self):
        """Read wave functions from file into memory."""
        matrix = Matrix(*self.matrix.shape,
                        dtype=self.dtype, dist=self.matrix.dist)
        # Read band by band to save memory
        rows = matrix.dist.rows
        blocksize = (matrix.shape[0] + rows - 1) // rows
        for myn, psit_G in enumerate(matrix.array):
            n = matrix.dist.comm.rank * blocksize + myn
            if self.comm.rank == 0:
                big_psit_G = self.array[n]
                if big_psit_G.dtype == complex and self.dtype == float:
                    big_psit_G = big_psit_G.view(float)
                elif big_psit_G.dtype == float and self.dtype == complex:
                    big_psit_G = np.asarray(big_psit_G, complex)
            else:
                big_psit_G = None
            self._distribute(big_psit_G, psit_G)
        self.matrix = matrix
        self.in_memory = True


class UniformGridWaveFunctions(ArrayWaveFunctions):
    def __init__(self, nbands, gd, dtype=None, data=None, kpt=None, dist=None,
                 spin=0, collinear=True):
        ngpts = gd.n_c.prod()
        ArrayWaveFunctions.__init__(self, nbands, ngpts, dtype, data, dist,
                                    collinear)

        M = self.matrix

        if data is None:
            M.array = M.array.reshape(-1).reshape(M.dist.shape)

        self.myshape = (M.dist.shape[0],) + tuple(gd.n_c)
        self.gd = gd
        self.dv = gd.dv
        self.kpt = kpt
        self.spin = spin
        self.comm = gd.comm

    @property
    def array(self):
        if self.in_memory:
            return self.matrix.array.reshape(self.myshape)
        else:
            return self.matrix.array

    def _distribute(self, big_psit_R, psit_R):
        self.gd.distribute(big_psit_R, psit_R.reshape(self.gd.n_c))

    def __repr__(self):
        s = ArrayWaveFunctions.__repr__(self).split('(')[1][:-1]
        shape = self.gd.get_size_of_global_array()
        s = 'UniformGridWaveFunctions({}, gpts={}x{}x{})'.format(s, *shape)
        return s

    def new(self, buf=None, dist='inherit', nbands=None):
        if dist == 'inherit':
            dist = self.matrix.dist
        return UniformGridWaveFunctions(nbands or len(self),
                                        self.gd, self.dtype,
                                        buf,
                                        self.kpt, dist,
                                        self.spin)

    def view(self, n1, n2):
        return UniformGridWaveFunctions(n2 - n1, self.gd, self.dtype,
                                        self.array[n1:n2],
                                        self.kpt, None,
                                        self.spin)

    def plot(self):
        import matplotlib.pyplot as plt
        ax = plt.figure().add_subplot(111)
        a, b, c = self.array.shape[1:]
        ax.plot(self.array[0, a // 2, b // 2])
        plt.show()


class PlaneWaveExpansionWaveFunctions(ArrayWaveFunctions):
    def __init__(self, nbands, pd, dtype=None, data=None, kpt=None, dist=None,
                 spin=0, collinear=True):
        ng = ng0 = pd.myng_q[kpt]
        if data is not None:
            assert data.dtype == complex
        if dtype == float:
            ng *= 2
            if isinstance(data, np.ndarray):
                data = data.view(float)

        ArrayWaveFunctions.__init__(self, nbands, ng, dtype, data, dist,
                                    collinear)
        self.pd = pd
        self.gd = pd.gd
        self.comm = pd.gd.comm
        self.dv = pd.gd.dv / pd.gd.N_c.prod()
        self.kpt = kpt
        self.spin = spin
        if collinear:
            self.myshape = (self.matrix.dist.shape[0], ng0)
        else:
            self.myshape = (self.matrix.dist.shape[0], 2, ng0)

    @property
    def array(self):
        if not self.in_memory:
            return self.matrix.array
        elif self.dtype == float:
            return self.matrix.array.view(complex)
        else:
            return self.matrix.array.reshape(self.myshape)

    def _distribute(self, big_psit_G, psit_G):
        if self.collinear:
            if self.dtype == float:
                if big_psit_G is not None:
                    big_psit_G = big_psit_G.view(complex)
                psit_G = psit_G.view(complex)
            psit_G[:] = self.pd.scatter(big_psit_G, self.kpt)
        else:
            psit_sG = psit_G.reshape((2, -1))
            psit_sG[0] = self.pd.scatter(big_psit_G[0], self.kpt)
            psit_sG[1] = self.pd.scatter(big_psit_G[1], self.kpt)

    def matrix_elements(self, other=None, out=None, symmetric=False, cc=False,
                        operator=None, result=None, serial=False):
        if other is None or isinstance(other, ArrayWaveFunctions):
            if out is None:
                out = Matrix(len(self), len(other or self), dtype=self.dtype,
                             dist=(self.matrix.dist.comm,
                                   self.matrix.dist.rows,
                                   self.matrix.dist.columns))
            assert cc
            if other is None:
                assert symmetric
                operate_and_multiply(self, self.dv, out, operator, result)
            elif not serial:
                assert not symmetric
                operate_and_multiply_not_symmetric(self, self.dv, out,
                                                   other)
            elif self.dtype == complex:
                self.matrix.multiply(self.dv, 'N', other.matrix, 'C',
                                     0.0, out, symmetric)
            else:
                self.matrix.multiply(2 * self.dv, 'N', other.matrix, 'T',
                                     0.0, out, symmetric)
                if self.gd.comm.rank == 0:
                    correction = np.outer(self.matrix.array[:, 0],
                                          other.matrix.array[:, 0])
                    if symmetric:
                        out.array -= 0.5 * self.dv * (correction +
                                                      correction.T)
                    else:
                        out.array -= self.dv * correction
        else:
            assert not cc
            P_ani = {a: P_ni for a, P_ni in out.items()}
            other.integrate(self.array, P_ani, self.kpt)
        return out

    def new(self, buf=None, dist='inherit', nbands=None):
        if buf is not None:
            array = self.array
            buf = buf.ravel()[:array.size]
            buf.shape = array.shape
        if dist == 'inherit':
            dist = self.matrix.dist
        return PlaneWaveExpansionWaveFunctions(nbands or len(self),
                                               self.pd, self.dtype,
                                               buf,
                                               self.kpt, dist,
                                               self.spin, self.collinear)

    def view(self, n1, n2):
        return PlaneWaveExpansionWaveFunctions(n2 - n1, self.pd, self.dtype,
                                               self.array[n1:n2],
                                               self.kpt, None,
                                               self.spin, self.collinear)


def operate_and_multiply(psit1, dv, out, operator, psit2):
    if psit1.comm:
        if psit2 is not None:
            assert psit2.comm is psit1.comm
        if psit1.comm.size > 1:
            out.comm = psit1.comm
            out.state = 'a sum is needed'

    comm = psit1.matrix.dist.comm
    N = len(psit1)
    n = (N + comm.size - 1) // comm.size
    mynbands = len(psit1.matrix.array)

    buf1 = psit1.new(nbands=n, dist=None)
    buf2 = psit1.new(nbands=n, dist=None)
    half = comm.size // 2
    psit = psit1.view(0, mynbands)
    if psit2 is not None:
        psit2 = psit2.view(0, mynbands)

    for r in range(half + 1):
        rrequest = None
        srequest = None

        if r < half:
            srank = (comm.rank + r + 1) % comm.size
            rrank = (comm.rank - r - 1) % comm.size
            skip = (comm.size % 2 == 0 and r == half - 1)
            n1 = min(rrank * n, N)
            n2 = min(n1 + n, N)
            if not (skip and comm.rank < half) and n2 > n1:
                rrequest = comm.receive(buf1.array[:n2 - n1], rrank, 11, False)
            if not (skip and comm.rank >= half) and len(psit1.array) > 0:
                srequest = comm.send(psit1.array, srank, 11, False)

        if r == 0:
            if operator:
                operator(psit1.array, psit2.array)
            else:
                psit2 = psit

        if not (comm.size % 2 == 0 and r == half and comm.rank < half):
            m12 = psit2.matrix_elements(psit, symmetric=(r == 0), cc=True,
                                        serial=True)
            n1 = min(((comm.rank - r) % comm.size) * n, N)
            n2 = min(n1 + n, N)
            out.array[:, n1:n2] = m12.array[:, :n2 - n1]

        if rrequest:
            comm.wait(rrequest)
        if srequest:
            comm.wait(srequest)

        psit = buf1
        buf1, buf2 = buf2, buf1

    requests = []
    blocks = []
    nrows = (comm.size - 1) // 2
    for row in range(nrows):
        for column in range(comm.size - nrows + row, comm.size):
            if comm.rank == row:
                n1 = min(column * n, N)
                n2 = min(n1 + n, N)
                if mynbands > 0 and n2 > n1:
                    requests.append(
                        comm.send(out.array[:, n1:n2].T.conj().copy(),
                                  column, 12, False))
            elif comm.rank == column:
                n1 = min(row * n, N)
                n2 = min(n1 + n, N)
                if mynbands > 0 and n2 > n1:
                    block = np.empty((mynbands, n2 - n1), out.dtype)
                    blocks.append((n1, n2, block))
                    requests.append(comm.receive(block, row, 12, False))

    comm.waitall(requests)
    for n1, n2, block in blocks:
        out.array[:, n1:n2] = block


def operate_and_multiply_not_symmetric(psit1, dv, out, psit2):
    if psit1.comm:
        if psit2 is not None:
            assert psit2.comm is psit1.comm
        if psit1.comm.size > 1:
            out.comm = psit1.comm
            out.state = 'a sum is needed'

    comm = psit1.matrix.dist.comm
    N = len(psit1)
    n = (N + comm.size - 1) // comm.size
    mynbands = len(psit1.matrix.array)

    buf1 = psit1.new(nbands=n, dist=None)
    buf2 = psit1.new(nbands=n, dist=None)

    psit1 = psit1.view(0, mynbands)
    psit = psit2.view(0, mynbands)
    for r in range(comm.size):
        rrequest = None
        srequest = None

        if r < comm.size - 1:
            srank = (comm.rank + r + 1) % comm.size
            rrank = (comm.rank - r - 1) % comm.size
            n1 = min(rrank * n, N)
            n2 = min(n1 + n, N)
            if n2 > n1:
                rrequest = comm.receive(buf1.array[:n2 - n1], rrank, 11, False)
            if len(psit1.array) > 0:
                srequest = comm.send(psit2.array, srank, 11, False)

        m12 = psit1.matrix_elements(psit, cc=True, serial=True)
        n1 = min(((comm.rank - r) % comm.size) * n, N)
        n2 = min(n1 + n, N)
        out.array[:, n1:n2] = m12.array[:, :n2 - n1]

        if rrequest:
            comm.wait(rrequest)
        if srequest:
            comm.wait(srequest)

        psit = buf1
        buf1, buf2 = buf2, buf1
