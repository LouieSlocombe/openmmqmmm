Job functions
=============

Each job function takes a theory and a fragment, runs a calculation, returns a
:class:`~openmmqmmm.Results` object and writes it to a ``results_*.json`` file.

See :doc:`../guide/jobs` for what the results contain and how the files are named.

Single-point energies
---------------------

.. currentmodule:: openmmqmmm

.. autosummary::

   single_point
   single_point_theories
   single_point_fragments
   single_point_fragments_and_theories
   single_point_reaction
   reaction_energy

.. autofunction:: single_point

.. autofunction:: single_point_theories

.. autofunction:: single_point_fragments

.. autofunction:: single_point_fragments_and_theories

.. autofunction:: single_point_reaction

.. autofunction:: reaction_energy

Geometry optimization
---------------------

``optimize_geometry`` drives geomeTRIC; ``orca_external_optimizer`` hands the
optimization to ORCA instead.

.. currentmodule:: openmmqmmm

.. autosummary::

   optimize_geometry
   GeometricOptimizer
   orca_external_optimizer

.. autofunction:: optimize_geometry

.. autoclass:: GeometricOptimizer

.. autofunction:: orca_external_optimizer

Frequencies and Hessians
------------------------

Hessian files are the plain-text format ``read_hessian`` and ``write_hessian`` share.

.. currentmodule:: openmmqmmm

.. autosummary::

   numerical_frequencies
   analytic_frequencies
   read_hessian
   write_hessian
   approximate_full_hessian_from_smaller
   calc_rotational_constants

.. autofunction:: numerical_frequencies

.. autofunction:: analytic_frequencies

.. autofunction:: read_hessian

.. autofunction:: write_hessian

.. autofunction:: approximate_full_hessian_from_smaller

.. autofunction:: calc_rotational_constants

Running jobs in parallel
------------------------

.. currentmodule:: openmmqmmm

.. autosummary::

   job_parallel

.. autofunction:: job_parallel
