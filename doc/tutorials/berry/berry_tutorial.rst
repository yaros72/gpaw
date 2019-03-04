.. module:: gpaw.berryphase
.. _berry tutorial:

==========================
 Berry phase calculations
==========================

In this tutorial we provide a set of examples of how to calculate various physical properties derived from the `k`-space Berry phases.


Spontaneous polarization of tetragonal BaTiO3
=============================================

As a first example we calculate the spontaneous polarization of the ferroelectric BaTiO3 in the tetragonal phase. The initial step is to generate a ground state calculation with wavefunctions. This can be don e with the script :download:`gs_BaTiO3.py`. We can then run the script

.. literalinclude:: polarization_BaTiO3.py

which calculates the polarization. It will take a few minutes on a single CPU, but can also be parallelized. It generates a .json file that contains the polarization and will read if the script is run again. It is thus possible to submit the polarization script and print the polarization by rerunning the script above in the terminal. The calculation adds the contribution from the electrons and the nucleii, which implies that the result is independent of the positions of the atoms relative to the unit cell. The results should be 0.33 `C/m^2`, which agrees with the values from literature.


Born effective charges of tetragonal BaTiO3
===========================================

The Berry phase module can also be used to calculate Born effective charges. They are defined by

.. math:: Z^a_{ij}=V\frac{\partial P_i}{\partial u_j^a}

where `u_j^a` is the position of atom  `a` in direction `j` relative to the equilibrium position and `P_i` is the polarization in direction `i`. Like the case of phonons all atoms are moved in all possible directions and the induced polarization is calculated for each move. For the BaTiO3 structure calculated above the calculation is performed with the script

.. literalinclude:: born_BaTiO3.py

Again the results are written to a .json file and the Born effective charges can be viewed with the script

.. literalinclude:: get_borncharges.py

Due to symmetry all the tensors are diagonal. Note, however, the large differences between the components for each of the O atoms. The Born effective charges tell us how the atoms are affected by an external electric field.


Topological properties of stanene from parallel transport
=========================================================

As a last example we demonstrate how the `k`-space Berry phases can be applied to extract topological properties of solids. For an isolated band we can calculate the Berry phase

.. math:: \gamma_1(k_2, k_3)=i\int_0^1 dk_1 \langle u(\mathbf{k})|\partial_{k_1}|u(\mathbf{k})\rangle

where `k_i` is the component of crystal momentum corresponding to `\mathbf{b}_i` in reduced coordinates. The Berry phase is only gauge invariant modulo `2\pi` and since `\gamma_1(k_2, k_3)=\gamma_1(k_2+1, k_3)=\gamma_1(k_2, k_3+1)` modulo `2\pi`, we can count how many multiples of the `2\pi` that `\gamma_1` changes while `k_2` of `k_3` is adiabatically cycled through the Brillouin zone. This is the Chern number, which gives rise to a topological `\mathbb{Z}` classification of all two-dimensional insulators. For multiple valence bands the situation is slightly more complicated and one has to introduce the notion of parallel transport to obtain the Berry phases of individual bands. We refer to Ref. [#Olsen]_ for details.

For materials with time-reversal symmetry the Chern number vanishes. Instead, any insulator in two dimensions can be classified according to a `\mathbb{Z}_2` index `\nu` that counts the number of times berry phases acquire a particular value in half the Brillouin zone modulo two. Below we give the example of stanene, which is referred to as a quantum spin Hall insulator due to the non-trivial topology reflected by `\nu=1`. The ground state can be set up with the script :download:`gs_Sn.py`. Afterwards the Berry phases of all occupied bands are calculated with

.. literalinclude:: Sn_parallel_transport.py

Finally the berry phase spectrum can be plottet with :download:`plot_phase.py` and the result is shown below.

.. image:: phases.png
    :height: 400 px

Note the degeneracy of all phases at the time-reversal invariant points `\Gamma` and `M`. Also note that any horizontal line is transversed by an odd number of phases in half the Brillouin zone (for example the `\Gamma-M` line). We also display the expectation value of `S_z` according to color. This is possible because the individual phases correspond to the first moments of hybrid Wannier functions localized along the `x`-direction and these functions have a spinorial structure with a well-defined value of `\langle S_z\rangle`. [#Olsen]_

.. [#Olsen] T. Olsen, E. Andersen, T. Okugawa, D. Torelli, T. Deilmann, K. S. Thygesen
            arXiv:1812.06666

