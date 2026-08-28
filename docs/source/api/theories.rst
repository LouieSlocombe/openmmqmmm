Theory objects
==============

A theory object knows how to turn coordinates into an energy and a gradient. Every job
function in :doc:`jobs` takes one. ``ORCATheory`` is the QM side, ``OpenMMTheory`` the MM
side, and ``QMMMTheory`` couples them by electrostatic embedding.

See :doc:`../guide/theories` and :doc:`../guide/qmmm` for how to configure them.

Quantum mechanics
-----------------

.. currentmodule:: openmmqmmm

.. autosummary::

   ORCATheory

.. autoclass:: ORCATheory

Molecular mechanics
-------------------

.. currentmodule:: openmmqmmm

.. autosummary::

   OpenMMTheory

.. autoclass:: OpenMMTheory

QM/MM
-----

``QMMMTheory`` is itself a theory, so it goes anywhere a plain theory does.

.. currentmodule:: openmmqmmm

.. autosummary::

   QMMMTheory
   define_active_region
   expand_qm_region
   expand_qm_pc_region
   compute_decomposed_qm_mm_energy
   read_charges_from_psf

.. autoclass:: QMMMTheory

.. autofunction:: define_active_region

.. autofunction:: expand_qm_region

.. autofunction:: expand_qm_pc_region

.. autofunction:: compute_decomposed_qm_mm_energy

.. autofunction:: read_charges_from_psf

Numerical gradients and test theories
-------------------------------------

``NumGrad`` wraps a theory that has no analytic gradient; ``ZeroTheory`` returns
zero energy and gradient, for exercising a workflow without a calculation.

.. currentmodule:: openmmqmmm

.. autosummary::

   NumGrad
   ZeroTheory

.. autoclass:: NumGrad

.. autoclass:: ZeroTheory

ASE
---

Exposes a configured ``QMMMTheory`` to the Atomic Simulation Environment.

.. currentmodule:: openmmqmmm

.. autosummary::

   OpenMMQMMMCalculator

.. autoclass:: OpenMMQMMMCalculator
