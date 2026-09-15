OpenMM setup and dynamics
=========================

Building a solvated, parameterized system, equilibrating it, and running dynamics on it.

See :doc:`../guide/system_setup`, :doc:`../guide/dynamics` and :doc:`../guide/rpmd` for the
task-oriented versions of this page.

System preparation
------------------

``openmm_modeller`` is the usual entry point: it repairs a PDB file, adds hydrogens,
solvates, and writes out a system ready for ``OpenMMTheory``.

.. currentmodule:: openmmqmmm

.. autosummary::

   openmm_modeller
   openmm_minimize
   solvate_small_molecule
   merge_pdb_files

.. autofunction:: openmm_modeller

.. autofunction:: openmm_minimize

.. autofunction:: solvate_small_molecule

.. autofunction:: merge_pdb_files

Equilibration
-------------

``openmm_box_equilibration`` settles the box under NPT; ``gentle_warmup_md`` raises
the temperature in stages, which a freshly built system usually needs.

.. currentmodule:: openmmqmmm

.. autosummary::

   openmm_box_equilibration
   gentle_warmup_md

.. autofunction:: openmm_box_equilibration

.. autofunction:: gentle_warmup_md

Molecular dynamics
------------------

``openmm_md`` is a thin wrapper around ``MolecularDynamicsEngine``; drive the engine
directly when you need ``extra_reporters`` or ``pre_dynamics_hook``.

.. currentmodule:: openmmqmmm

.. autosummary::

   openmm_md
   MolecularDynamicsEngine
   openmm_md_plumed
   check_gradient_for_bad_atoms

.. autofunction:: openmm_md

.. autoclass:: MolecularDynamicsEngine

.. autofunction:: openmm_md_plumed

.. autofunction:: check_gradient_for_bad_atoms

Handing the potential to another driver
---------------------------------------

The seam with openmmnqe and anything else that drives OpenMM itself; see
:doc:`../guide/nqe_interop`.

.. currentmodule:: openmmqmmm

.. autosummary::

   export_rpmd_potential
   RPMDPotentialExport
   modeller_from_topology

.. autofunction:: export_rpmd_potential

.. autoclass:: RPMDPotentialExport

.. autofunction:: modeller_from_topology

Lower-level helpers
-------------------

Exported from ``openmmqmmm.openmm`` only, not from the package root: the alternate-location
report, the nonbonded-parameter dump and the system-size summary.

.. currentmodule:: openmmqmmm.openmm

.. autosummary::

   find_alternate_locations_residues
   write_xmlfile_nonbonded
   print_systemsize

.. autofunction:: find_alternate_locations_residues

.. autofunction:: write_xmlfile_nonbonded

.. autofunction:: print_systemsize
