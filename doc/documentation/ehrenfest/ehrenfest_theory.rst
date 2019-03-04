.. _ehrenfest_theory:

======================================
Ehrenfest dynamics (TDDFT/MD) - Theory
======================================

Molecular dynamics (MD) simulations usually rely on the Born-Oppenheimer
approximation, where the electronic system is assumed to react so much
faster than the ionic system that it reaches its ground state at each timestep.
Thus, forces for the dynamics are calculated from the DFT groundstate density.
While this approximation is sufficently valid in most situations, there are
cases where the explicit dynamics of the electronic system can affect the
molecular dynamics, or the movement of the atoms can affect averaged spectral
or other properties. These cases can be handled using so-called Ehrenfest
dynamics, ie. time-dependent density functional theory molecular dynamics
(TDDFT/MD).

This guide describes the basics of the Ehrenfest dynamics implementation
in GPAW from a theoretical and point of view. For examples, see :ref:`Ehrenfest dynamics <ehrenfest>`.

The original implementation by Ari Ojanpera is described in Ref. \ [#Ojanpera2012]_.

Time-dependent DFT in the PAW formalism
=======================================

In the past decades, time-dependent DFT has become a popular method for calculating
materials properties related excited electronic states as well as for simulating processes,
in which nonadiabatic electron-ion dynamics plays a significant role. There are two main
realizations of TDDFT: the time-propagation scheme and the linear-response method. The
most general realization of TDDFT is the former scheme, in which the time-dependent Kohn-Sham
equations are integrated over the time domain.

The starting point of time-propagation TDDFT is the all-electron time-dependent Kohn-Sham
(TDKS) equation,

.. math::

   \begin{equation}
   i \frac{\partial \psi_n ({\bf r}, t)}{\partial t} = \hat {H} (t) \psi_n ({\bf r},t),
   \end{equation}

where $\psi_n$ is the Kohn-Sham wavefunction of electronic state $n$, and $\hat{H}$ is the electronic
Hamiltonian. Using the PAW approximation `\psi_n ({\bf r}, t) = \hat{\cal T} \tilde{\psi}_n ({\bf r}, t)`
and operating from the left with the adjoint of the PAW operator `\hat{\cal T}^{\dagger}`, we obtain the following equation:

.. math::
   :label: pawt_tdks

   \begin{equation}
   i \hat{\cal T}^{\dagger} \hat{\cal T} \frac{\partial \tilde{\psi}_n ({\bf r}, t)}{\partial t}
   = [\hat{\cal T}^{\dagger} \hat{H} \hat{\cal T}  -i \hat{\cal T}^{\dagger} \frac{\partial \hat{\cal T}}
   {\partial t}]\tilde{\psi}_n ({\bf r}, t).
   \end{equation}

Next, we define the PAW Hamiltonian `\tilde{H}`, the PAW overlap operator and the `\tilde{P}` term, which
corrects for the movement of the atoms in the TDKS equation, in the following manner:

.. math::

   \begin{align}
   \tilde{S} &= \hat{\cal T}^{\dagger} \hat{\cal T},\\ \tilde{H} &= \hat{\cal T}^{\dagger} \hat{H}
   \hat{\cal T},  \\ \tilde{P} &= -i \hat{\cal T}^{\dagger} \frac{\partial \hat{\cal T}}{\partial t}.
   \end{align}

Using these definitions, the PAW-transformed TDKS equation (Eq. :eq:`pawt_tdks`) is reduced to a more
compact form,

.. math::
   :label: tdks_paw

   \begin{equation}
   i \tilde{S} \frac{\partial \tilde{\psi}_n}{\partial t} = [\tilde{H} + \tilde{P}] \tilde{\psi}_n ({\bf r}, t)
   \end{equation}

In order to solve Eq. :eq:`tdks_paw` in practice, a method called semi-implicit Crank-Nicholson (SICN) is used in GPAW.
The Crank-Nicholson propagator is often used in time-propagation TDDFT calculations, since it is both unitary and time-reversible.
Semi-implicit means that a predictor-corrector scheme is used for accounting for the non-linearity of the Hamiltonian. At each time step,
one first assumes the Hamiltonian to be constant during the time step, and solves the predictor equation to obtain the predicted
future wavefunctions,

.. math::

   \begin{equation}
   [\tilde{S} + i \frac{\Delta t}{2} (\tilde{H} (t) + \tilde{P})] \tilde{\psi}^{\text{pred}} (t + \Delta t) = [\tilde{S}
   - i \frac{\Delta t}{2}(\tilde{H} (t) + \tilde{P}) \tilde{\psi}(t),
   \end{equation}

where the position-dependence of `\psi_n` is dropped for convenience. Note that `\tilde{S}` and `\tilde{P}` do not depend
explicitly on time, but instead through the atomic positions and velocities. The predicted Hamiltonian
`\tilde{H}^{\text{pred}}` is calculated from the predicted wavefunctions. The Hamiltonian in the middle of the time step
is obtained by taking the average

.. math::

   \begin{equation}
   \tilde{H}(t + \Delta t/2) = \frac{1}{2} [\tilde{H}(t) + \tilde{H}^{\text{pred}} (t + \Delta t)].
   \end{equation}

Finally, the propagated wavefunctions are obtained from the corrector equation,

.. math::

   \begin{equation}
   [\tilde{S} + i \frac{\Delta t}{2} (\tilde{H} (t + \Delta t/2) + \tilde{P})] \tilde{\psi} (t + \Delta t) = [\tilde{S}
   - i \frac{\Delta t}{2}(\tilde{H} (t + \Delta t/2) + \tilde{P}) \tilde{\psi}(t).
   \end{equation}

Time-propagation of the electron-ion system
===========================================

However, this only covers the propagation of the electronic subsystem. In order to propagate the coupled electron-ion system,
the following splitting of electronic and nuclear propagation is employed,

.. math::
   :label: uen

   \begin{equation}
   \hat{U}_{N,e} = \hat{U}_N (t, t + \Delta t/2) \hat{U}_e (t + \Delta t) \hat{U}_N (t + \Delta t/2, t + \Delta t),
   \end{equation}

where the propagator for the nuclei (`U_N`) is the Velocity Verlet algorithm. In practice, Eq. (:eq:`uen`) means that the nuclei
are first propagated forward by `\Delta t/2`, while the electronic subsystem is kept unchanged. Then, the positions of the nuclei remain
fixed, while the electronic subsystem is propagated by `\Delta t`. Finally, the nuclei are propagated by `\Delta t/2`. The following five-step
scheme describes the propagation of electrons and nuclei in the GPAW implementation of Ehrenfest dynamics:

.. math::

   \begin{align}
   \ddot{\bf R}(t) &=
   \frac{\mathbf{F}(\mathbf{R}(t), n (t))}{M} \\
   \mathbf{R} (t + \Delta t/2) &= \mathbf{R}(t) + \dot{\bf R} (t) \frac{\Delta t}{2} + \frac{1}{2}
   \ddot{\bf R}(t) \left(\frac{\Delta t}{2}\right)^2 \\
   \dot{\bf R}(t+ \Delta t/4) &= \dot{\bf R}(t) +
   \frac{1}{2} \ddot{\bf R}(t) \frac{\Delta t}{2}
   \end{align}

|

.. math::

   \begin{align}
   \ddot{\bf R} (t + \Delta t/2) &= \frac{\mathbf{F} (\mathbf{R}(t+ \Delta t /2), n(t))}{M} \\
   \dot{\bf R} (t + \Delta t/2) &= \dot{\bf R} (t + \Delta t /4) + \frac{1}{2} \ddot{\bf R} (t +
   \Delta t/2) \frac{\Delta t}{2}
   \end{align}

|

.. math::

   \begin{align}
   \tilde{\psi}_n(t + \Delta t; {\bf R} (t+ \Delta t/2)) = \hat{U}^{\text{SICN}} (t, t+\Delta t)
   \tilde{\psi}_n (t; {\bf R} (t+ \Delta t/2))
   \end{align}

|

.. math::

   \begin{align}
   \ddot{\bf R}(t + \Delta t/2) &= \frac{\mathbf{F}( \mathbf{R}(t+\Delta t/2), n(t+\Delta t))}{M} \\
   \mathbf{R}(t + \Delta t) = \mathbf{R}(t+\Delta t/2) &+ \dot{\bf R}(t + \Delta t/2) \frac{\Delta t}{2}
   + \frac{1}{2} \ddot{\bf R}(t+\Delta t/2) \left( \frac{\Delta t}{2}\right)^2\\
   \dot{\bf R}(t+ 3\Delta t/4) &= \dot{\bf R}(t + \Delta t/2) + \frac{1}{2} \ddot{\bf R}(\Delta t/2)
   \frac{\Delta t}{2}
   \end{align}

.. math::

   \begin{align}
   &\dot{\bf R} ( t+ \Delta t) = \dot{\bf R}(t+ 3\Delta t/4) + \frac{1}{2} \ddot{\bf R}(t+ \Delta t)
   \frac{\Delta t}{2} \\
   &\text{update } n (t + \Delta t, {\bf R} (t + \Delta t/2)) \rightarrow n (t
   + \Delta t, {\bf R}(t + \Delta t)),
   \end{align}

where `{\bf R}`, `M` and `{\bf F}` denote the positions of the nuclei, atomic masses and atomic forces, respectively, and `n` denotes the
electron density. Calculation of the atomic forces is tricky in PAW-based Ehrenfest dynamics due to the atomic position-dependent
PAW transformation. In the GPAW program the force is derived on the grounds the the total energy of the quantum-classical
system is conserved.

The atomic forces in Ehrenfest dynamics are thoroughly analysed and explained
in Ref. [#Ojanpera2012]_.

References
==========

.. [#Ojanpera2012] A. Ojanpera, V. Havu, L. Lehtovaara, M. Puska,
                   "Nonadiabatic Ehrenfest molecular dynamics within the projector augmented-wave method",
                   *J. Chem. Phys.* **136**, 144103 (2012).
