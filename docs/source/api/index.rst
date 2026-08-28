API reference
=============

``openmmqmmm.__all__`` is the public API: 76 names, re-exported at the package root from
the module that defines them. ``from openmmqmmm import single_point`` and
``from openmmqmmm.singlepoint import single_point`` are equivalent, and the root import is
the documented one.

The pages below cover every one of those names, grouped by what they are for rather than
by which module they live in. A handful of lower-level OpenMM helpers are exported from
``openmmqmmm.openmm`` only; those carry their full path in the pages that list them.

.. note::

   Anything not listed here is internal, whatever its name looks like. The package ships
   inline type annotations and a ``py.typed`` marker, so a type checker will follow the
   signatures on these pages into your own code.

.. toctree::
   :maxdepth: 2

   fragments
   theories
   jobs
   openmm
   results
   errors

Package metadata
----------------

.. autodata:: openmmqmmm.__version__
   :no-value:

   Taken from the installed package's metadata; ``"0.0.0+unknown"`` when running from a
   source tree that was never installed.
