# -*- coding: utf-8 -*-

from __future__ import print_function
import os.path
import numpy as np
from ase import parallel as mpi
from ase.parallel import parprint


class FiniteDifference:

    """
    atoms: Atoms object
        The atoms to work on.
    propertyfunction: function that returns a single number.
        The finite difference calculation is progressed on this value.
        For proper parallel usage the function should either be
        either a property of the atom object
            fd = FiniteDifference(atoms, atoms.property_xyz)
        or an arbitrary function with the keyword "atoms"
            fd = FiniteDifference(atoms, function_xyz)
            xyz = fd.run(atoms=atoms)
    d: float
        Magnitude of displacements.
    save: If true the write statement of the calculator is called
        to save the displacementsteps.
    name: string
        Name for restart data
    ending: string
        File handel for restart data
    parallel: int
        splits the mpi.world into 'parallel' subprocs that calculate
        displacements of different atoms individually.
    """

    def __init__(self, atoms, propertyfunction,
                 save=False, name='fd', ending='',
                 d=0.001, parallel=0, world=None):

        self.atoms = atoms
        self.indices = np.asarray(range(len(atoms)))
        self.propertyfunction = propertyfunction
        self.save = save
        self.name = name
        self.ending = ending
        self.d = d
        self.value = np.empty([len(self.atoms), 3])

        if world is None:
            world = mpi.world
        self.world = world
        if world.size < 2:
            if parallel > 1:
                parprint('#', (self.__class__.__name__ + ':'),
                         'Serial calculation, keyword parallel ignored.')
                parallel = 0
        self.parallel = parallel
        if parallel > 1:
            self.set_parallel()

    def calculate(self, a, i, filename='fd', **kwargs):
        """Evaluate finite difference  along i'th axis on a'th atom.
        This will trigger two calls to propertyfunction(), with atom a moved
        plus/minus d in the i'th axial direction, respectively.
        if save is True the +- states are saved after
        the calculation
        """
        if 'atoms' in kwargs:
            kwargs['atoms'] = self.atoms

        p0 = self.atoms.positions[a, i]
        if self.parallel > 1:
            par = 0
            while (par + 1) * len(self.atoms) / self.parallel <= a:
                par += 1
            if self.world.rank not in self.ranks[par]:
                return

        self.atoms.positions[a, i] += self.d
        eplus = self.propertyfunction(**kwargs)
        if self.save is True:
            savecalc = self.atoms.get_calculator()
            savecalc.write(filename + '+' + self.ending)

        self.atoms.positions[a, i] -= 2 * self.d
        eminus = self.propertyfunction(**kwargs)
        if self.save is True:
            savecalc = self.atoms.get_calculator()
            savecalc.write(filename + '-' + self.ending)
        self.atoms.positions[a, i] = p0

        self.value[a, i] = (eminus - eplus) / (2 * self.d)
        if self.parallel > 1:
            k = a - par * len(self.atoms) // self.parallel
            self.locvalue[k, i] = (eminus - eplus) / (2 * self.d)
            print('# rank', self.world.rank, 'Atom', a,
                  'direction', i, 'FD: ', self.value[a, i])
        else:
            parprint('Atom', a, 'direction', i,
                     '\nFD: ', self.value[a, i], '\n')
        return((eminus - eplus) / (2 * self.d))

    def run(self, **kwargs):
        """Evaluate finite differences for all atoms
        """

        for filename, a, i in self.displacements():
            self.calculate(a, i, filename=filename, **kwargs)

        if self.parallel > 1:
            gathercom = self.world.new_communicator(self.ranks[:, 0])
            if self.world.rank in self.ranks[:, 0]:
                gathercom.all_gather(self.locvalue, self.value)
            self.world.broadcast(self.value, 0)
        return self.value

    def displacements(self):
        for a in self.indices:
            for i in range(3):
                filename = ('{0}_{1}_{2}'.format(self.name, a, 'xyz'[i]))
                yield filename, a, i

    def restart(self, restartfunction, **kwargs):
        """Uses restartfunction to recalculate values
        from the saved files.
        If a file with the corresponding name is found the
        restartfunction is called to get the FD value
        The restartfunction should take a string as input
        parameter like the standart read() function.
        If no file is found, a calculation is initiated.
        Example:
            def re(self, name):
                calc = Calculator(restart=name)
                return calc.get_potential_energy()

            fd = FiniteDifference(atoms, atoms.get_potential_energy)
            fd.restart(re)
        """
        for filename, a, i in self.displacements():

            if (os.path.isfile(filename + '+' + self.ending) and
                    os.path.isfile(filename + '-' + self.ending)):
                eplus = restartfunction(
                    self, filename + '+' + self.ending, **kwargs)
                eminus = restartfunction(
                    self, filename + '-' + self.ending, **kwargs)
                self.value[a, i] = (eminus - eplus) / (2 * self.d)
            else:
                self.calculate(a, i, filename=filename, **kwargs)

        return self.value

    def set_parallel(self):
        assert self.world.size == 1 or self.world.size % self.parallel == 0
        assert len(self.atoms) % self.parallel == 0

        calc = self.atoms.get_calculator()
        calc.write(self.name + '_eq' + self.ending)
        self.locvalue = np.empty([len(self.atoms) // self.parallel, 3])
        ranks = np.array(range(self.world.size), dtype=int)
        self.ranks = ranks.reshape(
            self.parallel, self.world.size // self.parallel)
        self.comm = []
        for i in range(self.parallel):
            self.comm.append(self.world.new_communicator(self.ranks[i]))
            if self.world.rank in self.ranks[i]:
                calc2 = calc.__class__(
                    restart=self.name + '_eq' + self.ending,
                    communicator=self.comm[i], txt=None)
                self.atoms.set_calculator(calc2)
                calc2.calculate(atoms=self.atoms)

        return
