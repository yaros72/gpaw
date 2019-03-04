.. _literature:

----------
Literature
----------


Links to guides and manual pages
--------------------------------

* The GPAW calculator :ref:`manual`

* The :ref:`devel` pages

* The :ref:`guide for developers <developersguide>`

* The code :ref:`overview`

* The :ref:`features and algorithms` in GPAW


.. _literature_reports_presentations_and_theses:

Reports, presentations, and theses using gpaw
---------------------------------------------

* Summer-school 2014 talk about PAW, GPAW and ASE: :download:`ss14.pdf`

* A short note on the basics of PAW: :download:`paw_note.pdf`

* A master thesis on the inclusion of non-local exact exchange in the
  PAW formalism, and the implementation in gpaw:
  :download:`rostgaard_master.pdf`

* A master thesis on the inclusion of a localized basis in the PAW
  formalism, plus implementation and test results in GPAW:
  :download:`marco_master.pdf`

* A master thesis on the inclusion of localized basis sets in the PAW
  formalism, focusing on basis set generation and force calculations:
  :download:`askhl_master.pdf`

* A course report on a project involving the optimization of the
  setups (equivalent of pseudopotentials) in gpaw:
  :download:`askhl_10302_report.pdf`

* Slides from a talk about PAW: :download:`mortensen_paw.pdf`

* Slides from a talk about GPAW development:
  :download:`mortensen_gpaw-dev.pdf`

* Slides from a mini symposium during early development stage:
  :download:`mortensen_mini2003talk.pdf`


.. _paw_papers:

Articles on the PAW formalism
-----------------------------

The original article introducing the PAW formalism:
   | P. E. Blöchl
   | `Projector augmented-wave method`__
   | Physical Review B, Vol. **50**, 17953, 1994

   __ http://dx.doi.org/10.1103/PhysRevB.50.17953

A different formulation of PAW by Kresse and Joubert designed to make the transition from USPP to PAW easy.
  | G. Kresse and D. Joubert
  | `From ultrasoft pseudopotentials to the projector augmented-wave method`__
  | Physical Review B, Vol. **59**, 1758, 1999

  __ http://dx.doi.org/10.1103/PhysRevB.59.1758

A second, more pedagogical, article on PAW by Blöchl and co-workers.
  | P. E. Blöchl, C. J. Först, and J. Schimpl
  | `Projector Augmented Wave Method: ab-initio molecular dynamics with full wave functions`__
  | Bulletin of Materials Science, Vol. **26**, 33, 2003

  __ http://www.ias.ac.in/matersci/


.. _gpaw_publications:

Citations of the GPAW method papers
-----------------------------------

.. image:: citations.png
   :width: 750

(updated on May 18, 2013)

The total number of citations above is the number of publications
citing at least one of the other papers, not the sum of all citation
counts.

The six method papers are:

gpaw1:
    \J. J. Mortensen, L. B. Hansen, and K. W. Jacobsen

    `Real-space grid implementation of the projector augmented
    wave method`__

    Physical Review B, Vol. **71**, 035109 (2005)

    __ http://dx.doi.org/10.1103/PhysRevB.71.035109

tddft:
    \M. Walter, H. Häkkinen, L. Lehtovaara, M. Puska, J. Enkovaara,
    C. Rostgaard, and J. J. Mortensen

    `Time-dependent density-functional theory in the projector
    augmented-wave method`__

    Journal of Chemical Physics, Vol. **128**, 244101 (2008)

    __ http://dx.doi.org/10.1063/1.2943138

lcao:
    \A. H. Larsen, M. Vanin, J. J. Mortensen, K. S. Thygesen, and
    K. W. Jacobsen

    `Localized atomic basis set in the projector augmented wave method`__

    Physical Review B, Vol. **80**,  195112 (2009)

    __ http://dx.doi.org/10.1103/PhysRevB.80.195112

gpaw2:
    \J. Enkovaara, C. Rostgaard, J. J. Mortensen, J. Chen, M. Dulak,
    L. Ferrighi, J. Gavnholt, C. Glinsvad, V. Haikola, H. A. Hansen,
    H. H. Kristoffersen, M. Kuisma, A. H. Larsen, L. Lehtovaara,
    M. Ljungberg, O. Lopez-Acevedo, P. G. Moses, J. Ojanen, T. Olsen,
    V. Petzold, N. A. Romero, J. Stausholm, M. Strange, G. A. Tritsaris,
    M. Vanin, M. Walter, B. Hammer, H. Häkkinen, G. K. H. Madsen,
    R. M. Nieminen, J. K. Nørskov, M. Puska, T. T. Rantala,
    J. Schiøtz, K. S. Thygesen, and K. W. Jacobsen

    `Electronic structure calculations with GPAW: a real-space
    implementation of the projector augmented-wave method`__

    \J. Phys.: Condens. Matter **22**, 253202 (2010)

    __ http://stacks.iop.org/0953-8984/22/253202

response:
    Jun Yan, Jens. J. Mortensen, Karsten W. Jacobsen, and Kristian S. Thygesen

    `Linear density response function in the projector augmented wave method:
    Applications to solids, surfaces, and interfaces`__

    Phys. Rev. B **83**, 245122 (2011)

    __ http://prb.aps.org/abstract/PRB/v83/i24/e245122

csm:
    \A. Held and M. Walter

    `Simplified continuum solvent model with a smooth cavity based on
    volumetric data`__

    \J. Chem. Phys. **141**, 174108 (2014)

    __ http://dx.doi.org/10.1063/1.4900838


All citing articles:

.. csv-table::
   :file: citations.csv
   :header: #, title
   :widths: 1, 15
