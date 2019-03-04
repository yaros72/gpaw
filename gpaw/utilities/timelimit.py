import time
import numpy as np

from gpaw.analyse.observers import Observer


def time_to_seconds(timestr):
    """Convert time to seconds

    Parameters:

    timestr: float or string
        Float in seconds or string in format
        'DD-HH:MM:SS', 'HH:MM:SS', 'MM:SS', or 'SS'.
    """
    try:
        return float(timestr)
    except ValueError:
        pass
    time = 0.0
    d_i = timestr.split('-')
    if len(d_i) > 1:
        assert len(d_i) == 2
        time += int(d_i[0]) * 24 * 60 * 60
        timestr = d_i[1]
    d_i = timestr.split(':')
    mult = 1
    for d in d_i[::-1]:
        time += int(d) * mult
        mult *= 60
    return time


class TimeLimiter(Observer):
    """Class for automatically breaking the loops of GPAW calculation.

    The time estimation is done by a polynomial fit to
    the data `(i, dt)`, where `i` is the iteration index and
    `dt` is the calculation time of that iteration.

    The loop is broken by adjusting paw.maxiter value (or equivalent).
    """

    # Keywords for supported loops
    scf = 'scf'
    tddft = 'tddft'

    def __init__(self, paw, timestart=None, timelimit='10:00',
                 output=None, interval=1):
        """__init__ method

        Parameters:

        paw:
            GPAW calculator
        timestart: float
            The start time defining the "zero time".
            Format: as given by time.time().
        timelimit: float or string
            The allowed run time counted from `timestart`.
            Format: any supported by function `time_to_seconds()`.
        output: str
            The name of the output file for dumping the time estimates.
        """
        Observer.__init__(self, interval)
        self.timelimit = time_to_seconds(timelimit)
        if timestart is None:
            self.time0 = time.time()
        else:
            self.time0 = timestart
        self.comm = paw.world
        self.output = output
        self.do_output = self.output is not None
        if self.comm.rank == 0 and self.do_output:
            self.outf = open(self.output, 'w')
        self.loop = None
        paw.attach(self, interval, paw)

    def reset(self, loop, order=0, min_updates=5):
        """Reset the time estimation.

        Parameters:

        loop: str
            The keyword of the controlled loop.
        order: int
            The polynomial order of the fit used to estimate
            the run time between each update.
        min_updates: int
            The minimum number of updates until time estimates are given.
        """
        if loop not in [self.scf, self.tddft]:
            raise RuntimeError('Unsupported loop type: {}'.format(loop))
        self.loop = loop
        if self.comm.rank == 0:
            self.order = order
            self.min_updates = max(min_updates, order + 1)
            self.time_t = [time.time()]  # Add the initial time
            self.iteridx_t = []

    def update(self, paw):
        """Update time estimate and break calculation if necessary."""
        # Select the iteration index
        if self.loop is None:
            return
        elif self.loop == self.scf:
            iteridx = paw.scf.niter
        elif self.loop == self.tddft:
            iteridx = paw.niter

        # Update the arrays
        if self.comm.rank == 0:
            self.time_t.append(time.time())
            self.iteridx_t.append(iteridx)
            self.p_k = None

            if self.do_output:
                timediff = self.time_t[-1] - self.time_t[-2]
                line = 'update %12d %12.4f' % (iteridx, timediff)
                self.outf.write('%s\n' % line)
                # self.outf.flush()

        # Check if there is time to do the next iteration
        if not self.has_time(iteridx + self.interval):
            # The calling loop is assumed to do "niter += 1"
            # after calling observers
            paw.log('{}: Breaking the loop '
                    'due to the time limit'.format(self.__class__.__name__))
            if self.loop == self.scf:
                paw.scf.maxiter = iteridx
            elif self.loop == self.tddft:
                paw.maxiter = iteridx

    def eta(self, iteridx):
        """Estimate the time required to calculate the iteration of
           the given index `iteridx`."""
        if self.comm.rank == 0:
            if len(self.iteridx_t) < self.min_updates:
                eta = 0.0
            else:
                if self.p_k is None:
                    iteridx_t = np.array(self.iteridx_t)
                    time_t = np.array(self.time_t)
                    timediff_t = time_t[1:] - time_t[:-1]

                    self.p_k = np.polyfit(iteridx_t,
                                          timediff_t,
                                          self.order)
                if type(iteridx) in (int, float):
                    iteridx = [iteridx]
                iteridx_i = np.array(iteridx)
                eta = max(0.0, np.sum(np.polyval(self.p_k, iteridx_i)))

                if self.do_output:
                    line = 'eta    %12s %12.4f' % (iteridx, eta)
                    self.outf.write('%s\n' % line)
            return eta
        else:
            return None

    def has_time(self, iteridx):
        """Check if there is still time to calculate the iteration of
           the given index `iteridx`."""
        if self.timelimit is None:
            return True
        # Calculate eta on master and broadcast to all ranks
        data_i = np.empty(1, dtype=int)
        if self.comm.rank == 0:
            if len(self.iteridx_t) < self.min_updates:
                data_i[0] = True
            else:
                time_required = self.eta(iteridx)
                time_available = self.timelimit - (time.time() - self.time0)
                data_i[0] = time_required < time_available
        self.comm.broadcast(data_i, 0)
        return bool(data_i[0])

    def __del__(self):
        if self.comm.rank == 0 and self.do_output:
            self.outf.close()
