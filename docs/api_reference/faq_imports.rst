Frequently Used Import Targets
==============================

Includes a brief summary of important symbols and modules.

Service Users
-------------

If you write client code interacting with the OSS Vizier service, use
these import targets:

-  **from vizier.service import pyvizier as vz**: Exposes the same
   set of symbol names as ``vizier.pyvizier``. ``vizier.service.pyvizier.Foo``
   is a subclass or an alias of ``vizier.pyvizier.Foo``, and can be converted
   into protobufs.
-  **from vizier.service import ...**: Include binaries and internal
   utilities.

Algorithm Developers
--------------------

If you write algorithm code (Designers or Pythia policies) in OSS
Vizier, use these import targets:

-  **from vizier import pyvizier as vz**: Pure python building
   blocks of OSS Vizier. Cross-platform code, including Pythia policies,
   must use this ``pyvizier`` instance.

   -  ``Trial`` and ``ProblemStatement`` are important classes.

-  **from vizier.pyvizier import converters**: Convert between
   ``pyvizier`` objects and numpy arrays.

   -  ``TrialToNumpyDict``: Converts parameters (and metrics) into a
      dict of numpy arrays. Preferred conversion method if you intended
      to train an embedding of categorical/discrete parameters, or data
      includes missing parameters or metrics.
   -  ``TrialToArrayConverter``: Converts parameters (and metrics) into
      an array.

-  **from vizier.interfaces import serializable**: Abstractions for
   serializable objects.

   -  ``PartiallySerializable``, ``Serializable``

Algorithm Abstractions
~~~~~~~~~~~~~~~~~~~~~~

-  **from vizier import pythia**: Abstractions for Pythia policies.

   -  ``Policy``, ``PolicySupporter``: Key abstractions.
   -  ``LocalPolicyRunner``: Use it for running a ``Policy`` in RAM.

-  **from vizier import algorithms**: Abstractions for algorithms.

   -  ``Designer``: Stateful algorithm abstraction.
   -  ``DesignerPolicy``: Wraps ``Designer`` into a Pythia Policy.
   -  ``GradientFreeMaximizer``: For optimizing acquisition functions.
   -  ``(Partially)SerializableDesigner``: Designers who wish to
      optimize performance by saving states.

Tensorflow Modules
~~~~~~~~~~~~~~~~~~

-  **from vizier import tfp**: Tensorflow-Probability utilities.

   -  ``acquisitions``: Acquisition functions module.

      -  ``AcquisitionFunction``: Abstraction.
      -  ``UpperConfidenceBound``, ``ExpectedImprovement``, etc.

   -  ``bijectors``: Bijectors module.

      -  ``YeoJohnson``: Implements both Yeo-Johnson and Box-Cox
         transformations.
      -  ``optimal_power_transformation``: Returns the optimal power
         transformation.
      -  ``flip_sign``: returns a sign-flip bijector.

-  **from vizier import keras as vzk**:

   -  ``vzk.layers``: Layers usually wrapping ``tfp`` classes.

      -  ``variable_from_prior``: Utility layer for handling regularized
         variables.

   -  ``vzk.models``: Most of the useful models don’t easily fit into
      Keras's ``Model`` abstraction, but we may add some for display.
   -  ``vzk.optim``: Wrappers around optimizers in ``tfp`` or
      ``keras``.
