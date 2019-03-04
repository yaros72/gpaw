.. _defects_theory:

=======================================================================
Localised electrostatic charges in non-uniform dielectric media: Theory
=======================================================================

For examples, see :ref:`defects_tutorial`.

Introduction
============

The purpose of this section is to enable us to calculate the formation energy of
charged defects. In the Zhang-Northrup formula, this is given by

.. math::
  E^f[X^q] = E[X^q] - E_0 - \sum_i\mu_in_i + q (\epsilon_v + \epsilon_F)

In this formula, `X` labels the type of defect and `q` its charge state, i.e.
the net charge contained in some volume surrounding the defect. `q` is defined
such that `q=-1` for an electron. `E[X^q]` is the total energy of the sample
with the defect, and `E_0` the energy of the pristine (bulklike) sample.

These quantities are usually calculated in a supercell approach with periodic
boundary conditions. That is, we create the defect we are interested in, and
place it in an environment containing many repetitions of the pristine unit
cell. In the infinite supercell limit, we can describe the properties of the
isolated defect.

When we employ periodic boundary conditions, our calculation includes spurious
electrostatic interactions between localised charge distribution of the defect
state, and all its periodically repeated images. These long-ranged interactions
mean that the convergence with respect to supercell size is slow.

To accelerate this convergence, we employ an electrostatic correction scheme,
and write

.. math::
  E^f[X^q]_{\mathrm{corrected}} = E^f[X^q]_{\mathrm{uncorrected}} - E_{\mathrm{periodic}} + E_{\mathrm{isolated}} + q\Delta V

This assumes that DFT describes the bonding and energies of the system well, but
contains errors in the electrostatics. The correction thus consists in
subtracting the spurious interactions between the periodic images (the term
included in the DFT calculation) and adding in the energy of an isolated charge
distribution in the given dielectric environment. Finally, the potentials
between the charged and neutral states must be aligned to the same reference.

Implementing this scheme consists of the following steps, which will be
described in further detail in the sections below. First, we determine the
`z`-dependent dielectric function of the 2D layers. Knowing this, we use a model
charge distribution to emulate the behaviour of the defect charge state, and
find the potential associated with this model distribution by solving the
Poisson equation. We do this twice: first with periodic boundary conditions, and
then with zero boundary conditions. The electrostatic energy associated with the
charge distribution in a given potential can then be found from the usual
formula,

.. math::
  U = \frac{1}{2}\int_{\Omega} \rho V.

Defining the dielectric response of 2D layers
=============================================

In two dimensions, the bulk dielectric response is poorly defined, and some
model must be used.

The approach used here, following Ref [#Komsa]_ is to assume that the dielectric
function of the isolated layer is isotropic in plane, and varies only in the `z`
direction. Additionally, the screening is assumed to follow the density
distribution of the system so that we can write

.. math::
 \varepsilon^{i}(z) = k^i\cdot n(z) + 1,

Where `i` varies over "in-plane" and "out-of-plane", and `n` is the in-plane
averaged density of the system. The normalization constants, `k^i`, are chosen such that

.. math::

  \frac{1}{L} \int \mathrm{d} z\, \varepsilon^{\parallel}(z) &= \varepsilon^{\parallel}_{\mathrm{DFT}} \\
  \frac{1}{L} \int \mathrm{d} z\, \left(\varepsilon^{\perp}(z)\right)^{-1} &= \left(\varepsilon^{\perp}_{\mathrm{DFT}}\right)^{-1}



Calculating the energy of the periodic images
=============================================

Once we have the dielectric response, we proceed by solving the poisson equation

.. math::

  \nabla \cdot \boldsymbol{\varepsilon}(z) \odot \nabla \phi(\mathbf r) =
  -\rho(\mathbf r).

Here `\odot` is the elementwise (Hadamard) product, and `\boldsymbol{\varepsilon} =
(\varepsilon^{\parallel}, \varepsilon^{\parallel}, \varepsilon^{\perp})`.
`\rho(\mathbf r)` is a model charge distribution that we use to describe the
charge distribution of the defect state. For convenience, a Gaussian is often
chosen, so that

.. math::

  \rho(\mathbf r) = \frac{1}{\left(\sqrt{2\pi}\sigma\right)^3} e^{-(\mathbf r -
  \mathbf r_0)^2/(2\sigma^2)}.

In terms of the coordinate axes, the poisson equation reduces to

.. math::

  \varepsilon^{\parallel}(z)\frac{\partial^2 V}{\partial x^2} +
  \varepsilon^{\parallel}(z)\frac{\partial^2 V}{\partial y^2} +
  \varepsilon^{\perp}(z)\frac{\partial^2 V}{\partial z^2} + \frac{\partial
  \varepsilon^{\perp}}{\partial z} \frac{\partial V}{\partial
  z} = -\rho(\mathbf r)

Since we wish to solve this equation with periodic boundary conditions, we
Fourier transform the above equation, giving

.. math::

  \varepsilon^{\parallel}(G_z) * \left[(G_x^2 + G_y^2)V(\mathbf G)\right] +
  \varepsilon^{\perp}(G_z) * \left[G_z^2 V(\mathbf G)\right] + \left[G_z
  \varepsilon^{\perp}(G_z)\right] * \left[ G_z V\mathbf(G)\right] = \rho(\mathbf
  G),

where `*` denotes a convolution along the `z` axis. Writing out these
convolutions, we finally arrive at the expression

.. math::

  \rho_{G_x,G_y,G_z} = \sum_{G_z'} \left[\varepsilon^\parallel_{G_z -
  G_z'}\left(G_x^2 + G_y^2\right) + \varepsilon^{\perp}_{G_z - G_z'}G_zG_z'\right] V_{G_x,
  G_y, G_z'}.

For each value of `(G_x, G_y)`, we can thus calculate the corresponding potential `V_{G_z}` through a matrix inversion, and use that to calculate the energy of the model charge distribution.

We can also use the potential `V_{G_z}` to calculate the alignment term, `\Delta V`. We can Fourier transform this to get real-space potential of the model charge distribution. If we have described the electrostatics of the system well, this potential should be similar to the true potential of the defect charge distribution, up to a constant shift. Defining

.. math::
  \Delta V(\vec{r}) = V(\vec{r}) - [V^{X^q}_\mathrm{el}(\vec{r}) - V^{0}_\mathrm{el}(\vec{r}) ],

We set 

Calculating the energy of the isolated system
=============================================

We start as before, with the Poisson equation, but since we would like to
describe the energy of the isolated defect, we do not impose periodic boundary
conditions and Fourier transform. Instead, following Ref. [#Ping]_ we can
exploit the in-plane symmetry of the problem and expand `\phi` using cylindrical
Bessel functions.

.. math::

  \phi(\mathbf r) = \int_0^\infty \mathrm{d}k'\, 2qe^{-k'^2\sigma^2/2}
  \varphi_{k'}(z) J_0(\rho k')\right

Inserting this into the above equation and using the orthogonality relation
`\int \rho\mathrm{d}\rho J_0(\rho k)J_0(\rho k') = \delta(k - k') / k` we find
that `\varphi_k` must obey the Poisson equation

.. math::

  -\frac{\partial}{\partial z}\left(\varepsilon^{\perp}(z) \frac{\partial
  \varphi_k(z)}{\partial z}\right) + k^2\varepsilon^{\parallel}(z) \varphi_k(z) =
  \frac{1}{\sqrt{2\pi}\sigma}e^{-\left(z - z_0\right)^2/\left(2\sigma^2\right)},

where `z_0` is the center of the gaussian density along the `z` direction. The normalization of `\varphi_k` defined above was chosen precisely so that the right hand side of this equation is a normalized gaussian along the `z` direction.

We solve this equation by separating the response into two components: The bulk response,
describing the screening far away from the material, and the remaining
`z` -dependent response close to the system. We thus define `\Delta \varepsilon^i(z) = \varepsilon^i(z) - \varepsilon^{i}_{\mathrm{bulk}}` and the Green's function of the bulk response `\hat K = (-\varepsilon^{\perp}_{\mathrm{bulk}} \frac{\partial^2}{\partial z^2} + k^2\varepsilon^{\parallel}_{\mathrm{bulk}})^{-1}`. As an implementation detail, we note that for 2D materials, the bulk response is generally 1. 

The equation for `\varphi_k` can then be written as

.. math::

  \hat{K}^{-1} \varphi_k - \frac{\partial}{\partial
  z}\left(\Delta\varepsilon^{\perp} \frac{\partial \varphi_k}{\partial
  z}\right) + k^2\Delta\varepsilon^{\parallel}\varphi_k =
  \frac{1}{\sqrt{2\pi}\sigma}e^{-\left(z - z_0\right)^2/\left(2\sigma^2\right)}.

Only the first term on the left hand side is affected by the boundary conditions on `\varphi_k`. We can solve this by Fourier transforming along the `z` axis and wigner-seitz truncating the Green's function, which yields the following equation

.. math::
  \sum_{G_z} D_{G_zG_z'} \left(\varphi_k\right)_{G_z} = e^{-i G_z'z_0 - G_z'^2\sigma^2/2},

with the matrix `D` given by

.. math::

   \frac{1}{L}D_{G_z'G_z} = \frac{\varepsilon^{\parallel}_{\mathrm{b}}k^2 +
   \varepsilon^{\perp}_{\mathrm{b}}G_z^2}{1 -
   e^{-kL/2}\cos(G_zL/2)}\delta_{G_zG_z'} + \Delta\varepsilon^{\parallel}_{G_z -
   G_z'}k^2 + \varepsilon^{\perp}_{G_z - G_z'}G_zG_z'.

Finding `\varphi_k` is thus just a simple matrix inversion. Once we have solved the poisson equation, we calculate the total energy.

.. math::
  U &=  \frac{1}{2} \int_{\Omega} \rho(\mathbf r) \phi(\mathbf r) \\
    &= q^2 \int k \mathrm{d}k e^{-k^2 \sigma^2} U_k,

with

.. math::

  U_k \int \mathrm{d}z\, \varphi_k(z)
  \frac{1}{\sqrt{2\pi}\sigma}e^{-\left(z - z_0\right)^2/\left(2\sigma^2\right)}

Using the solution to the Poisson equation, this reduces to

.. math::

  U_k = \sum_{G_z,G_z'} e^{i(G_z - G_z')z_0 - (G_z^2 + G_z'^2)\sigma^2 / 2}
  \left(D^{-1}\right)_{G_zG_z'},

With `D` defined as above.

References
==========
.. [#Komsa] H.-P. Komsa, T. T. Rantala and A. Pasquarello
              *Phys. Rev. B* **86**, 045112 (2012)

.. [#Ping] R. Sundararaman and Y. Ping
		   *J. Chem. Phys.* **146**, 104109 (2017)
