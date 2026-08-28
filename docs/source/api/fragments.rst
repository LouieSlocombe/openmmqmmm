Fragments and coordinates
=========================

A :class:`~openmmqmmm.Fragment` is the molecular system every job function takes:
elements, coordinates in Angstrom, charge, multiplicity, connectivity and -- when it
was built from a PDB file -- an OpenMM topology. The functions below build, measure,
align and write those coordinates.

See :doc:`../guide/fragments` for the task-oriented version of this page.

Classes
-------

.. currentmodule:: openmmqmmm

.. autosummary::

   Fragment
   Reaction

.. autoclass:: Fragment

.. autoclass:: Reaction

Reading and writing coordinates
-------------------------------

``read_*`` return element/coordinate pairs rather than fragments; pass them to
``Fragment(elems=..., coords=...)`` when a fragment is what you want.

.. currentmodule:: openmmqmmm

.. autosummary::

   read_xyzfile
   read_xyzfiles
   split_multimolxyzfile
   read_ambercoordinates
   read_gromacsfile
   write_xyzfile
   write_pdbfile

.. autofunction:: read_xyzfile

.. autofunction:: read_xyzfiles

.. autofunction:: split_multimolxyzfile

.. autofunction:: read_ambercoordinates

.. autofunction:: read_gromacsfile

.. autofunction:: write_xyzfile

.. autofunction:: write_pdbfile

Measuring geometry
------------------

Distances in Angstrom, angles and dihedrals in degrees.

.. currentmodule:: openmmqmmm

.. autosummary::

   distance_between_atoms
   angle_between_atoms
   dihedral_between_atoms
   print_internal_coordinate_table
   calculate_rmsd
   nuc_nuc_repulsion

.. autofunction:: distance_between_atoms

.. autofunction:: angle_between_atoms

.. autofunction:: dihedral_between_atoms

.. autofunction:: print_internal_coordinate_table

.. autofunction:: calculate_rmsd

.. autofunction:: nuc_nuc_repulsion

Aligning and combining structures
---------------------------------

.. currentmodule:: openmmqmmm

.. autosummary::

   flexible_align
   flexible_align_pdb
   flexible_align_xyz
   insert_solute_into_solvent
   get_molecules_from_trajectory

.. autofunction:: flexible_align

.. autofunction:: flexible_align_pdb

.. autofunction:: flexible_align_xyz

.. autofunction:: insert_solute_into_solvent

.. autofunction:: get_molecules_from_trajectory

Constraint helpers
------------------

Constraint lists in the form the optimizer and the MD engine accept.

.. currentmodule:: openmmqmmm

.. autosummary::

   define_xh_constraints
   get_water_constraints
   simple_get_water_constraints

.. autofunction:: define_xh_constraints

.. autofunction:: get_water_constraints

.. autofunction:: simple_get_water_constraints
