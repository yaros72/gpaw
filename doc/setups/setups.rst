.. _setups:
.. _datasets:
    
=================
Atomic PAW Setups
=================

A setup is to the PAW method what a pseudo-potential is to the
pseudo-potential method.  All available setups are contained in this
tar-file: gpaw-setups-0.9.11271.tar.gz_.  There are setups for the LDA,
PBE, revPBE, RPBE and GLLBSC functionals.  Install them as described
in the :ref:`installation of paw datasets` section.  The setups are stored as
compressed :ref:`pawxml` files.


Setup releases
==============

===========  =============================
Date         Tarfile
===========  =============================
Mar 22 2016  gpaw-setups-0.9.20000.tar.gz_
Mar 27 2014  gpaw-setups-0.9.11271.tar.gz_
Oct 26 2012  gpaw-setups-0.9.9672.tar.gz_
Apr 13 2011  gpaw-setups-0.8.7929.tar.gz_
Apr 19 2010  gpaw-setups-0.6.6300.tar.gz_
Jul 22 2009  gpaw-setups-0.5.3574.tar.gz_
===========  =============================


.. _periodic table:
    
Periodic table
==============

=== === === === === === === === === === === === === === === === === ===
H_                                                                  He_
Li_ Be_                                         B_  C_  N_  O_  F_  Ne_
Na_ Mg_                                         Al_ Si_ P_  S_  Cl_ Ar_
K_  Ca_ Sc_ Ti_ V_  Cr_ Mn_ Fe_ Co_ Ni_ Cu_ Zn_ Ga_ Ge_ As_ Se_ Br_ Kr_
Rb_ Sr_ Y_  Zr_ Nb_ Mo_ Tc  Ru_ Rh_ Pd_ Ag_ Cd_ In_ Sn_ Sb_ Te_ I_  Xe_
Cs_ Ba_ La_ Hf_ Ta_ W_  Re_ Os_ Ir_ Pt_ Au_ Hg_ Tl_ Pb_ Bi_ Po  At  Rn_
=== === === === === === === === === === === === === === === === === ===


.. _installation of paw datasets:
    
Installation of PAW datasets
============================

The PAW datasets can be installed automatically or manually.

To install them automatically, run :command:`gpaw install-data
{<dir>}`.  This downloads and unpacks the newest package into
:file:`{<dir>}/gpaw-setups-{<version>}`.  When prompted, answer
yes (y) to register the path in the GPAW configuration file.

To manually install the setups, do as follows:

1) Get the tar file :file:`gpaw-setups-{<version>}.tar.gz`
   of the <version> of PAW datasets from the :ref:`setups` page
   and unpack it somewhere, preferably in ``$HOME``
   (``cd; tar -xf gpaw-setups-<version>.tar.gz``) - it could
   also be somewhere global where
   many users can access it like in :file:`/usr/share/gpaw-setups/`.
   There will now be a subdirectory :file:`gpaw-setups-{<version>}/`
   containing all the atomic data for the most commonly used functionals.

2) Set the environment variable :envvar:`GPAW_SETUP_PATH`
   to point to the directory
   :file:`gpaw-setups-{<version>}/`, e.g. for bash users, you would put into
   :file:`~/.bashrc`::

       export GPAW_SETUP_PATH=~/gpaw-setups-<version>

   Refer to :ref:`using_your_own_setups` for alternative way of
   setting the location of PAW datasets.

   .. note::

     In case of several locations of PAW datasets the first found setup
     file is used.

See also `NIST Atomic Reference Data`_.

.. _NIST Atomic Reference Data: http://physics.nist.gov/PhysRefData/DFTdata/Tables/ptable.html


.. toctree::
   :maxdepth: 2

   molecule_tests
   bulk_tests
   g2_1
   dcdft
   generation_of_setups
   pawxml


.. toctree::
   :glob:
   :hidden:

   [A-Z]*


.. _gpaw-setups-0.9.20000.tar.gz:
    https://wiki.fysik.dtu.dk/gpaw-files/gpaw-setups-0.9.20000.tar.gz
.. _gpaw-setups-0.9.11271.tar.gz:
    https://wiki.fysik.dtu.dk/gpaw-files/gpaw-setups-0.9.11271.tar.gz
.. _gpaw-setups-0.9.9672.tar.gz:
    https://wiki.fysik.dtu.dk/gpaw-files/gpaw-setups-0.9.9672.tar.gz
.. _gpaw-setups-0.8.7929.tar.gz:
    https://wiki.fysik.dtu.dk/gpaw-files/gpaw-setups-0.8.7929.tar.gz
.. _gpaw-setups-0.6.6300.tar.gz:
    https://wiki.fysik.dtu.dk/gpaw-files/gpaw-setups-0.6.6300.tar.gz
.. _gpaw-setups-0.5.3574.tar.gz:
    https://wiki.fysik.dtu.dk/gpaw-files/gpaw-setups-0.5.3574.tar.gz

..  _H:  H.html
.. _He: He.html
.. _Li: Li.html
.. _Be: Be.html
..  _B:  B.html
..  _C:  C.html
..  _N:  N.html
..  _O:  O.html
..  _F:  F.html
.. _Ne: Ne.html
.. _Na: Na.html
.. _Mg: Mg.html
.. _Al: Al.html
.. _Si: Si.html
..  _P:  P.html
..  _S:  S.html
.. _Cl: Cl.html
.. _Ar: Ar.html
..  _K:  K.html
.. _Ca: Ca.html
.. _Sc: Sc.html
.. _Ti: Ti.html
..  _V:  V.html
.. _Cr: Cr.html
.. _Mn: Mn.html
.. _Fe: Fe.html
.. _Co: Co.html
.. _Ni: Ni.html
.. _Cu: Cu.html
.. _Zn: Zn.html
.. _Ga: Ga.html
.. _Ge: Ge.html
.. _As: As.html
.. _Se: Se.html
.. _Br: Br.html
.. _Kr: Kr.html
.. _Rb: Rb.html
.. _Sr: Sr.html
..  _Y: Y.html
.. _Zr: Zr.html
.. _Nb: Nb.html
.. _Mo: Mo.html
.. _Ru: Ru.html
.. _Rh: Rh.html
.. _Pd: Pd.html
.. _Ag: Ag.html
.. _Cd: Cd.html
.. _In: In.html
.. _Sn: Sn.html
.. _Sb: Sb.html
.. _Te: Te.html
..  _I:  I.html
.. _Xe: Xe.html
.. _Cs: Cs.html
.. _Ba: Ba.html
.. _La: La.html
.. _Hf: Hf.html
.. _Ta: Ta.html
..  _W:  W.html
.. _Re: Re.html
.. _Os: Os.html
.. _Ir: Ir.html
.. _Pt: Pt.html
.. _Au: Au.html
.. _Hg: Hg.html
.. _Tl: Tl.html
.. _Pb: Pb.html
.. _Bi: Bi.html
.. _Rn: Rn.html
