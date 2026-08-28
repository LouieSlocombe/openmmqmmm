openmmqmmm
==========

Electrostatically embedded QM/MM for biomolecular systems, combining the
`ORCA <https://www.faccts.de/orca/>`_ quantum chemistry program with the
`OpenMM <https://openmm.org>`_ molecular-mechanics library.

The package is derived from the `ASH <https://github.com/RagnarB83/ash>`_ multiscale
modelling program, reduced to the ORCA + OpenMM QM/MM stack and given a PEP 8 Python API.
It covers the whole path from a raw PDB file to a QM/MM trajectory: system preparation and
solvation, minimization and equilibration, single points, geometry optimization,
frequencies, classical and QM/MM molecular dynamics, and ring-polymer dynamics for nuclear
quantum effects.

Installation
------------

.. code-block:: bash

   bash build_tools/conda_install.sh

One command: it creates the ``openmmqmmm`` conda environment, compiles PLUMED and the
OpenMM-PLUMED plugin into it, and installs this package and forcefill. ORCA is installed
separately. See :doc:`install` for the pip-only route, the cluster route, and what each one
leaves out.

A QM/MM calculation
-------------------

.. code-block:: python

   from openmmqmmm import (
       Fragment,
       OpenMMTheory,
       ORCATheory,
       QMMMTheory,
       configure_logging,
       openmm_md,
       optimize_geometry,
   )

   configure_logging()

   fragment = Fragment(pdbfile="system.pdb")

   qm = ORCATheory(orcasimpleinput="! r2SCAN-3c tightscf", numcores=8)
   mm = OpenMMTheory(
       xmlfiles=["charmm36.xml", "charmm36/water.xml", "specialresidue.xml"],
       pdbfile="system.pdb",
       periodic=True,
   )

   qmatoms = [93, 94, 95, 96, 97, 133, 134, 135, 2001, 2002]
   qm_mm = QMMMTheory(
       qm_theory=qm, mm_theory=mm, fragment=fragment, qm_charge=-1, qm_mult=6, qmatoms=qmatoms
   )

   # Optimize the QM region ...
   optimize_geometry(theory=qm_mm, fragment=fragment, actatoms=qmatoms)
   # ... or run QM/MM molecular dynamics (ps).
   openmm_md(fragment=fragment, theory=qm_mm, timestep=0.001, simulation_time=2)

Where to go next
----------------

* :doc:`install` — the three installation routes, and configuring ORCA.
* :doc:`quickstart` — a first gas-phase calculation, start to finish.
* :doc:`guide/fragments` — building the ``Fragment`` every job function takes.
* :doc:`guide/qmmm` — QM region, link atoms, embedding and the active region.
* :doc:`guide/system_setup` — going from a raw PDB file to a solvated, parameterized system.
* :doc:`guide/dynamics` and :doc:`guide/rpmd` — classical, QM/MM and ring-polymer dynamics.
* :doc:`api/index` — every name in the public API.

.. toctree::
   :maxdepth: 1
   :caption: Getting started
   :hidden:

   install
   quickstart

.. toctree::
   :maxdepth: 1
   :caption: User guide
   :hidden:

   guide/fragments
   guide/theories
   guide/qmmm
   guide/system_setup
   guide/jobs
   guide/dynamics
   guide/rpmd
   guide/nqe_interop
   guide/ase
   guide/output

.. toctree::
   :maxdepth: 2
   :caption: Reference
   :hidden:

   api/index
   conventions
   examples

.. toctree::
   :maxdepth: 1
   :caption: Project
   :hidden:

   development
   citing

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
