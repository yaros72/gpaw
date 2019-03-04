import _gpaw
from gpaw.fd_operators import FDOperator
import numpy


class WeightedFDOperator(FDOperator):
    def __init__(self, operators):
        """Compound operator A with nonlocal weights.

        A = Sum_i weights_i * operators_i

        Arguments:
        operators -- List of FDOperators.

        The weights have to be set using the set_weights method.
        A is build from operators which are then discarded.
        """
        self.gd = operators[0].gd
        self.shape = tuple(self.gd.n_c)
        self.coef_ps = []
        self.offset_ps = []
        for op in operators:
            assert self.gd == op.gd
            assert operators[0].dtype == op.dtype
            assert not hasattr(op, 'nweights')
            if 0 in op.offset_p:
                assert op.offset_p[0] == 0
                self.offset_ps.append(
                    numpy.ascontiguousarray(op.offset_p.copy())
                )
                self.coef_ps.append(
                    numpy.ascontiguousarray(op.coef_p.copy())
                )
            else:
                self.offset_ps.append(
                    numpy.ascontiguousarray(numpy.hstack(([0], op.offset_p)))
                )
                self.coef_ps.append(
                    numpy.ascontiguousarray(numpy.hstack(([.0], op.coef_p)))
                )
            assert self.offset_ps[-1][0] == 0
            assert len(self.coef_ps[-1]) == len(self.offset_ps[-1])
            assert self.offset_ps[-1].flags.c_contiguous
            assert self.coef_ps[-1].flags.c_contiguous
            assert self.offset_ps[-1].dtype == int
            assert self.coef_ps[-1].dtype == float

        self.nweights = len(operators)
        self.cfd = all([op.cfd for op in operators])
        self.mp = max([op.mp for op in operators])
        self.dtype = operators[0].dtype
        if self.gd.comm.size > 1:
            self.comm = self.gd.comm.get_c_object()
        else:
            self.comm = None
        self.npoints = max([op.npoints for op in operators])
        self.weights = None
        self.operator = None
        self.description = 'Weighted Finite Difference Operator\n      '
        self.description += '\n      '.join([op.description for op in operators])

    def set_weights(self, weights):
        """Set the operator weights.

        weights   -- List of numpy arrays sized gd.n_c
                     (weights are not copied).
        """
        assert len(weights) == self.nweights
        for weight in weights:
            assert weight.shape == self.shape
            assert weight.dtype == float
            assert weight.flags.c_contiguous
        self.weights = weights
        self.operator = _gpaw.WOperator(
            self.nweights, self.weights,
            self.coef_ps, self.offset_ps, self.gd.n_c, self.mp,
            self.gd.neighbor_cd, self.dtype == float, self.comm, self.cfd
        )
