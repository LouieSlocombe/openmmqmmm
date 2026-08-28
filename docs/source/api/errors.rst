Errors and logging
==================

Every error the package raises derives from :exc:`~openmmqmmm.OpenMMQMMMError`, and each
subclass also inherits the closest builtin -- so ``except ValueError`` keeps catching an
:exc:`~openmmqmmm.InputError`. Importing the package configures no logging handler; call
:func:`~openmmqmmm.configure_logging` to get output.

See :doc:`../guide/output`.

Exceptions
----------

.. currentmodule:: openmmqmmm

.. autosummary::

   OpenMMQMMMError
   InputError
   MissingDependencyError
   ExternalProgramError
   FileFormatError
   InternalError
   require

.. autoexception:: OpenMMQMMMError

.. autoexception:: InputError

.. autoexception:: MissingDependencyError

.. autoexception:: ExternalProgramError

.. autoexception:: FileFormatError

.. autoexception:: InternalError

.. autofunction:: require

Logging
-------

.. currentmodule:: openmmqmmm

.. autosummary::

   configure_logging

.. autofunction:: configure_logging
