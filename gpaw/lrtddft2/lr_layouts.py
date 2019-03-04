import numpy as np

from gpaw.blacs import BlacsGrid, Redistributor
import _gpaw

class LrDiagonalizeLayout:
    """BLACS layout for distributed Omega matrix in linear response
       time-dependet DFT calculations"""
    def __init__(self, sl_lrtddft, nrows, lr_comms):
        self.mprocs, self.nprocs, self.block_size = tuple(sl_lrtddft)

        self.lr_comms = lr_comms

        # original grid, ie, how matrix is stored
        self.matrix_grid = BlacsGrid( self.lr_comms.parent_comm,
                                      self.lr_comms.dd_comm.size,
                                      self.lr_comms.eh_comm.size )

        # diagonalization grid
        self.diag_grid = BlacsGrid(self.lr_comms.parent_comm, self.mprocs, self.nprocs)


        # -----------------------------------------------------------------
        # for SCALAPACK we need TRANSPOSED MATRIX (and vector)
        #
        # M = rows, N = cols
        M = nrows
        N = nrows
        mb = 1
        nb = 1
        self.matrix_descr = self.matrix_grid.new_descriptor(N, M, nb, mb)

        bs = self.block_size
        self.diag_descr = self.diag_grid.new_descriptor(N, M, bs, bs)

        self.diag_in_redist  = Redistributor( self.lr_comms.parent_comm,
                                              self.matrix_descr,
                                              self.diag_descr )

        self.diag_out_redist = Redistributor( self.lr_comms.parent_comm,
                                              self.diag_descr,
                                              self.matrix_descr )


    def diagonalize(self, eigenvectors, eigenvalues):
        """Diagonalize symmetric distributed Casida matrix using Scalapack.
        Parameters:

        eigenvectors
          distributed Casida matrix on input, distributed eigenvectors on output

        eigenvalues
          zero array on input, eigenvalues on output
        """
        O_diag = self.diag_descr.empty(dtype=float)
        if self.matrix_descr.blacsgrid.is_active():
            O_orig = eigenvectors
        else:
            O_orig = np.empty((0,0), dtype=float)

        self.diag_in_redist.redistribute(O_orig, O_diag)

        #print O_diag

        self.diag_descr.diagonalize_dc(O_diag.copy(), O_diag, eigenvalues, 'L')

        self.diag_out_redist.redistribute(O_diag, O_orig)

        self.lr_comms.parent_comm.broadcast(eigenvalues, 0)



class LrTDDFPTSolveLayout:
    """BLACS layouts for distributed TD-DFPT"""
    def __init__(self, sl_lrtddft, nrows, lr_comms):
        self.mprocs, self.nprocs, self.block_size = tuple(sl_lrtddft)

        self.lr_comms = lr_comms

        # for SCALAPACK we need TRANSPOSED MATRIX (and vector)
        #
        # -----------------------------------------------------------------
        # matrix

        # original grid, ie, how matrix is stored
        self.orig_matrix_grid = BlacsGrid( self.lr_comms.parent_comm,
                                           self.lr_comms.dd_comm.size,
                                           self.lr_comms.eh_comm.size )

        # solve grid
        self.solve_matrix_grid = BlacsGrid(self.lr_comms.parent_comm, self.mprocs, self.nprocs)

        # M = rows, N = cols
        M = nrows*4
        N = nrows*4
        mb = 4
        nb = 4
        self.orig_matrix_descr = self.orig_matrix_grid.new_descriptor(N, M, nb, mb)

        bs = self.block_size
        self.solve_matrix_descr = self.solve_matrix_grid.new_descriptor(N, M, bs, bs)

        self.matrix_in_redist  = Redistributor( self.lr_comms.parent_comm,
                                                self.orig_matrix_descr,
                                                self.solve_matrix_descr )

        # -----------------------------------------------------------------
        # vector

        # original grid, ie, how vector is stored
        self.orig_vector_grid = BlacsGrid( self.lr_comms.parent_comm,
                                           1,
                                           ( self.lr_comms.dd_comm.size *
                                             self.lr_comms.eh_comm.size ) )

        # solve grid
        #self.solve_vector_grid = BlacsGrid(self.lr_comms.parent_comm, self.mprocs, self.nprocs)

        # M = rows, N = cols
        M = nrows*4
        Nrhs = 1
        mb = 4
        nb = 1
        self.orig_vector_descr = self.orig_vector_grid.new_descriptor(Nrhs, M, nb, mb)

        bs = self.block_size
        self.solve_vector_descr = self.solve_matrix_grid.new_descriptor(Nrhs, M, 1, bs)

        self.vector_in_redist  = Redistributor( self.lr_comms.parent_comm,
                                                self.orig_vector_descr,
                                                self.solve_vector_descr )

        self.vector_out_redist = Redistributor( self.lr_comms.parent_comm,
                                                self.solve_vector_descr,
                                                self.orig_vector_descr )


    def solve(self, A_orig, b_orig):
        """Solve TD-DFPT equation using Scalapack.
        """

        A_solve = self.solve_matrix_descr.empty(dtype=float)
        if not self.orig_matrix_descr.blacsgrid.is_active():
            A_orig = np.empty((0,0), dtype=float)

        self.matrix_in_redist.redistribute(A_orig, A_solve)

        b_solve = self.solve_vector_descr.empty(dtype=float)
        if not self.orig_vector_descr.blacsgrid.is_active():
            b_orig = np.empty((0,0), dtype=float)

        self.vector_in_redist.redistribute(b_orig, b_solve)

        #if False:
        #    np.set_printoptions(precision=5, suppress=True)
        #    for i in range(self.lr_comms.parent_comm.size):
        #        if ( self.lr_comms.parent_comm.rank == i ):
        #            print 'rank ', i
        #            print A_orig
        #            print A_solve
        #            print
        #            print b_orig
        #            print b_solve
        #            print
        #            print
        #            print self.solve_matrix_descr.asarray()
        #            print self.solve_vector_descr.asarray()
        #            print
        #            print '---'
        #            print
        #        self.lr_comms.parent_comm.barrier()

        info = 0
        if self.solve_matrix_descr.blacsgrid.is_active():
            _gpaw.scalapack_solve(A_solve, self.solve_matrix_descr.asarray(), b_solve, self.solve_vector_descr.asarray())
            if info != 0:
                raise RuntimeError('scalapack_solve error: %d' % info)


        self.vector_out_redist.redistribute(b_solve, b_orig)


        #if False:
        #    for i in range(self.lr_comms.parent_comm.size):
        #        if ( self.lr_comms.parent_comm.rank == i ):
        #            print 'rank ', i
        #            print A_orig
        #            print A_solve
        #            print
        #            print b_orig
        #            print b_solve
        #            print
        #            print
        #        self.lr_comms.parent_comm.barrier()

        return b_orig


###############################
# BLACS layout for ScaLAPACK
class LrTDDFTLayouts:
    """BLACS layout for distributed Omega matrix in linear response
       time-dependet DFT calculations"""

    def __init__(self, sl_lrtddft, nkq, dd_comm, eh_comm):
        mcpus, ncpus, blocksize = tuple(sl_lrtddft)
        self.world = eh_comm.parent
        self.dd_comm = dd_comm
        if self.world is None:
            self.world = self.dd_comm

        # All the ranks within domain communicator contain the omega matrix
        # construct new communicator only on domain masters
        eh_ranks = np.arange(eh_comm.size) * dd_comm.size
        self.eh_comm2 = self.world.new_communicator(eh_ranks)

        self.eh_grid = BlacsGrid(self.eh_comm2, eh_comm.size, 1)
        self.eh_descr = self.eh_grid.new_descriptor(nkq, nkq, 1, nkq)
        self.diag_grid = BlacsGrid(self.world, mcpus, ncpus)
        self.diag_descr = self.diag_grid.new_descriptor(nkq, nkq,
                                                        blocksize,
                                                        blocksize)

        self.redistributor_in = Redistributor(self.world,
                                              self.eh_descr,
                                              self.diag_descr)
        self.redistributor_out = Redistributor(self.world,
                                               self.diag_descr,
                                               self.eh_descr)

        """
        # -----------------------------------------------------------------
        # for SCALAPACK we need TRANSPOSED MATRIX (and vector)
        # -----------------------------------------------------------------
        # M = rows, N = cols
        M = nkq*4; N = nkq*4; mb = nkq*4; nb = 4; Nrhs = 1
        # Matrix, mp=1, np=eh_comm.size
        self.eh_grid2a = BlacsGrid(self.eh_comm2, eh_comm.size, 1)
        # Vector, mp=eh_comm.size, np=1
        self.eh_grid2b = BlacsGrid(self.eh_comm2, 1, eh_comm.size)
        self.eh_descr2a = self.eh_grid2a.new_descriptor(N,    M,  nb, mb)
        self.eh_descr2b = self.eh_grid2b.new_descriptor(Nrhs, N,   1, nb)

        self.solve_descr2a =self.diag_grid.new_descriptor(N, M,
                                                          blocksize, blocksize)
        self.solve_descr2b =self.diag_grid.new_descriptor(Nrhs, N,
                                                          1, blocksize)

        self.redist_solve_in_2a = Redistributor(self.world,
                                                self.eh_descr2a,
                                                self.solve_descr2a)
        self.redist_solve_in_2b = Redistributor(self.world,
                                                self.eh_descr2b,
                                                self.solve_descr2b)

        self.redist_solve_out_2a = Redistributor(self.world,
                                                 self.solve_descr2a,
                                                 self.eh_descr2a)
        self.redist_solve_out_2b = Redistributor(self.world,
                                                 self.solve_descr2b,
                                                 self.eh_descr2b)
        """


        # -----------------------------------------------------------------
        # for SCALAPACK we need TRANSPOSED MATRIX (and vector)
        # -----------------------------------------------------------------
        # M = rows, N = cols
        M = nkq*4
        N = nkq*4
        mb = 4
        nb = 4
        Nrhs = 1
        # Matrix, mp=1, np=eh_comm.size
        self.eh_grid2a = BlacsGrid(self.world, dd_comm.size, eh_comm.size)
        # Vector, mp=eh_comm.size, np=1
        self.eh_grid2b = BlacsGrid(self.world, 1, dd_comm.size * eh_comm.size)
        self.eh_descr2a = self.eh_grid2a.new_descriptor(N,    M,  nb,   mb)
        self.eh_descr2b = self.eh_grid2b.new_descriptor(Nrhs, N,  Nrhs, nb)
        self.solve_descr2a =self.diag_grid.new_descriptor(N, M,
                                                          blocksize, blocksize)
        self.solve_descr2b =self.diag_grid.new_descriptor(Nrhs, N,
                                                          Nrhs, blocksize)


        self.redist_solve_in_2a = Redistributor(self.world,
                                                self.eh_descr2a,
                                                self.solve_descr2a)
        self.redist_solve_in_2b = Redistributor(self.world,
                                                self.eh_descr2b,
                                                self.solve_descr2b)

        self.redist_solve_out_2a = Redistributor(self.world,
                                                 self.solve_descr2a,
                                                 self.eh_descr2a)
        self.redist_solve_out_2b = Redistributor(self.world,
                                                 self.solve_descr2b,
                                                 self.eh_descr2b)



    def solve(self, A, b):
        #if 0:
        #    print 'edescr2a', rank, self.eh_descr2a.asarray()
        #    print 'edescr2b', rank, self.eh_descr2b.asarray()
        #
        #    sys.stdout.flush()
        #    self.world.barrier()
        #
        #    print 'sdescr2a', rank, self.solve_descr2a.asarray()
        #    print 'sdescr2b', rank, self.solve_descr2b.asarray()
        #
        #    sys.stdout.flush()
        #    self.world.barrier()
        #
        #    print 'A ', rank, A.shape
        #    if b is not None:
        #        print 'b ', rank, b.shape
        #
        #    sys.stdout.flush()
        #    self.world.barrier()

        A_nn = self.solve_descr2a.empty(dtype=float)
        if self.eh_descr2a.blacsgrid.is_active():
            A_Nn = A
        else:
            A_Nn = np.empty((0,0), dtype=float)
        self.redist_solve_in_2a.redistribute(A_Nn, A_nn)

        b_n = self.solve_descr2b.empty(dtype=float)
        if self.eh_descr2b.blacsgrid.is_active():
            b_N = b.reshape(1,len(b))
        else:
            b_N = np.empty((A_Nn.shape[0],0), dtype=float)
        self.redist_solve_in_2b.redistribute(b_N, b_n)


        #if 0:
        #    print 'A_Nn ', rank, A_Nn.shape
        #    print 'b_N  ', rank, b_N.shape
        #    sys.stdout.flush()
        #    self.world.barrier()
        #    print 'A_nn ', rank, A_nn.shape
        #    print 'b_n  ', rank, b_n.shape
        #    sys.stdout.flush()
        #    self.world.barrier()
        #
        #
        #    print 'b_N  ', rank, b_N
        #    sys.stdout.flush()
        #    self.world.barrier()
        #    print 'b_n ', rank, b_n
        #    sys.stdout.flush()
        #    self.world.barrier()
        #
        #    print 'A_Nn  ', rank, A_Nn
        #    sys.stdout.flush()
        #    self.world.barrier()
        #    print 'A_nn ', rank, A_nn
        #    sys.stdout.flush()
        #    self.world.barrier()

        info = 0
        if self.solve_descr2a.blacsgrid.is_active():
            _gpaw.scalapack_solve(A_nn, self.solve_descr2a.asarray(), b_n, self.solve_descr2b.asarray())
            if info != 0:
                raise RuntimeError('scalapack_solve error: %d' % info)
        self.redist_solve_out_2b.redistribute(b_n, b_N)


        if self.eh_descr2b.blacsgrid.is_active():
            b_N = b_N.flatten()
        else:
            b_N = b

        #self.dd_comm.broadcast(b_N, 0)

        b[:] = b_N


    def diagonalize(self, Om, eps_n):

        O_nn = self.diag_descr.empty(dtype=float)
        if self.eh_descr.blacsgrid.is_active():
            O_nN = Om
        else:
            O_nN = np.empty((0,0), dtype=float)

        self.redistributor_in.redistribute(O_nN, O_nn)
        self.diag_descr.diagonalize_dc(O_nn.copy(), O_nn, eps_n, 'L')
        self.redistributor_out.redistribute(O_nn, O_nN)
        self.world.broadcast(eps_n, 0)
        # Broadcast eigenvectors within domains
        if not self.eh_descr.blacsgrid.is_active():
            O_nN = Om
        self.dd_comm.broadcast(O_nN, 0)


###############################################################################
