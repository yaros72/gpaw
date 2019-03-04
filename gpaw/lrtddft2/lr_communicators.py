import gpaw.mpi


class LrCommunicators:
    def __init__(self, world=None, dd_size=None, eh_size=None):
        """Create communicators for LrTDDFT calculation.
        
        Input parameters:
        
        world
          MPI parent communicator (usually gpaw.mpi.world)
        
        dd_size
          Over how many processes is domain distributed.
      
        eh_size
          Over how many processes are electron-hole pairs distributed.
      
        Note
        ----
        Sizes must match, i.e., world.size must be equal to
        dd_size x eh_size, e.g., 1024 = 64*16

        Tip
        ---
        Use enough processes for domain decomposition (dd_size) to fit
        everything (easily) into memory, and use the remaining processes
        for electron-hole pairs as K-matrix build is trivially parallel
        over them.


        Pass lr_comms.dd_comm to ground state calc when reading for LrTDDFT.


        ----------------------------------------------------------------------

        Example (for 8 MPI processes)::

          lr_comms = LrCommunicators(gpaw.mpi.world, 4, 2)
          txt = 'lr_%06d_%06d.txt' % (lr_comms.dd_comm.rank,
                                      lr_comms.eh_comm.rank)
          lr = LrTDDFTindexed(GPAW('unocc.gpw', communicator=lr_comms.dd_comm),
                              lr_communicators=lr_comms, txt=txt)
        
        """

        self.parent_comm = None
        self.dd_comm = None
        self.eh_comm = None
        
        self.world = world
        self.dd_size = dd_size
        self.eh_size = eh_size
        
        if self.world is None:
            return
        if self.dd_size is None:
            return
        
        if self.eh_size is None:
            self.eh_size = self.world.size // self.dd_size
            
        self.parent_comm = self.world
        
        if self.world.size != self.dd_size * self.eh_size:
            raise RuntimeError('Domain decomposition processes (dd_size) '
                               'times electron-hole (eh_size) processes '
                               'does not match with total processes '
                               '(world size != dd_size * eh_size)')
            
        dd_ranks = []
        eh_ranks = []
        for k in range(self.world.size):
            if k // self.dd_size == self.world.rank // self.dd_size:
                dd_ranks.append(k)
            if k % self.dd_size == self.world.rank % self.dd_size:
                eh_ranks.append(k)
        self.dd_comm = self.world.new_communicator(dd_ranks)
        self.eh_comm = self.world.new_communicator(eh_ranks)

    def initialize(self, calc):
        if self.parent_comm is None:
            if calc is not None:
                self.dd_comm = calc.density.gd.comm
                self.parent_comm = self.dd_comm.parent
                if self.parent_comm.size != self.dd_comm.size:
                    raise RuntimeError('Invalid communicators in LrTDDFT2. Ground state calculator domain decomposition communicator and its parent (or actually its parent parent) has different size. Please set up LrCommunicators explicitly to avoid this. Or contact developers if this is intentional.')
                self.eh_comm = gpaw.mpi.serial_comm
            else:
                self.parent_comm = gpaw.mpi.serial_comm
                self.dd_comm = gpaw.mpi.serial_comm
                self.eh_comm = gpaw.mpi.serial_comm
        else:
            # Check that parent_comm is valid
            if self.parent_comm != self.eh_comm.parent:
                raise RuntimeError(
                    'Invalid communicators in LrTDDFT2. LrTDDFT2 parent '
                    'communicator does is not parent of electron-hole '
                    'communicator. Please set up LrCommunicators explicitly '
                    'to avoid this.')
            if self.parent_comm != self.dd_comm.parent:
                raise RuntimeError(
                    'Invalid communicators in LrTDDFT2. LrTDDFT2 parent '
                    'communicator does is not parent of domain decomposition '
                    'communicator. Please set up LrCommunicators explicitly '
                    'to avoid this.')

    # Do not use so slow... unless absolutely necessary
    # def index_of_kss(self,i,p):
    #     for (ind,kss) in enumerate(self.kss_list):
    #         if kss.occ_ind == i and kss.unocc_ind == p:
    #             return ind
    #     return None

    def get_local_eh_index(self, ip):
        if ip % self.eh_comm.size != self.eh_comm.rank:
            return None
        return ip // self.eh_comm.size

    def get_local_dd_index(self, jq):
        if jq % self.dd_comm.size != self.dd_comm.rank:
            return None
        return jq // self.dd_comm.size

    def get_global_eh_index(self, lip):
        return lip * self.eh_comm.size + self.eh_comm.rank

    def get_global_dd_index(self, ljq):
        return ljq * self.dd_comm.size + self.dd_comm.rank

    def get_matrix_elem_proc_and_index(self, ip, jq):
        ehproc = ip % self.eh_comm.size
        ddproc = jq % self.dd_comm.size
        proc = ehproc * self.dd_comm.size + ddproc
        lip = ip // self.eh_comm.size
        ljq = jq // self.eh_comm.size
        return (proc, ehproc, ddproc, lip, ljq)
