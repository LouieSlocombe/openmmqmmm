Results and trajectories
========================

The job functions return a :class:`~openmmqmmm.Results` object, and most also write it as JSON.
The trajectory helpers post-process what an MD run leaves behind.

Results
-------

.. currentmodule:: openmmqmmm

.. autosummary::

   Results
   read_results_from_file

.. autoclass:: Results

.. autofunction:: read_results_from_file

Trajectory processing
---------------------

Thin wrappers over mdtraj, for trajectories written by the MD engine.

.. currentmodule:: openmmqmmm

.. autosummary::

   mdtraj_image_trajectory
   mdtraj_rmsf

.. autofunction:: mdtraj_image_trajectory

.. autofunction:: mdtraj_rmsf
