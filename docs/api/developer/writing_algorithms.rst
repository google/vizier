Writing Algorithms
##################

Pythia Policy is what runs in Vizier service.
---------------------------------------------

Vizier server keeps a mapping between algorithm names and ``Policy`` objects (`Example`_). All algorithm implementations
must be eventually wrapped into ``Policy``.

However, you should directly subclass a ``Policy`` **only if you are an advanced user
and wants to fully control the algorithm behavior** including all database
operations. For all other developers, we recommend using an alternative abstraction
listed in the remainder of this section.

A typical ``Policy`` implementation is injected with a ``PolicySupporter``, which is
used for fetching data.
For a minimal example, see `RandomPolicy`_.

To interact with a ``Policy`` locally, you can use
``LocalPolicyRunner`` which is a local in-ram server-client for ``Policy``.
For implementing a policy, one should use ``vizier.pyvizier`` instead of
``vizier.service.pyvizier`` library. The former is platform-independent, and the
latter is platform-dependent. The most notable difference is that
``vizier.pyvizier.ProblemStatement`` is a subset of ``vizier.service.pyvizier.ProblemStatement``
that does not carry any service-related attributes (such as study identifier
and algorithms)::

  from vizier.pythia import Policy, LocalPolicyRunner
  from vizier import pyvizier as vz

  problem = vz.ProblemStatement()
  problem.search_space.select_root().add_float_param('x', 0.0, 1.0)
  problem.metric_information.append(
      vz.MetricInformation(name='objective', goal=vz.MetricInformation.MAXIMIZE))

  runner = LocalPolicyRunner(problem)
  policy = MyPolicy(runner)

  # Run for 10 iterations, each of which evaluates 5 new trials.
  for _ in range(10):
    new_trials = runner.SuggestTrials(policy, 5)
    for trial in new_trials:
      trial.complete(vz.Measurement(
          {'objective': trial.parameters_dict['x'] ** 2}))


Designer API is the recommended starting point
----------------------------------------------
`Designer`_ API provides a simplified entry point for implementing suggestion algorithms. For a minimal example, see `EmukitDesigner`_ which wraps GP-EI algorithm implemented in ``emukit`` into ``Designer`` interface. ``Designer``s are trivially wrapped into ``Policy`` via `DesignerPolicy`_.

Also see our `designer testing routine`_ for an up-to-date example on how to interact with designers.

The ``Designer`` interface is designed to let you forget about the ultimate goal
of serving the algorithm in a distributed environment. Pretend you'll use it
locally by doing a suggest-update loop in RAM, during the lifetime of a study.


Serializing your designer
-----------------------------

You can consider making your ``Designer`` serializable so that you can save and
load its state. Vizier offers `two options`_:

* ``Serializable`` should be used if your algorithm can be easily serialized. You can save and restore the state in full.
* ``PartiallySerializable`` should be used if your algorithm has subcomponents that are not easily serializable. You can recover the designer's state as long as it was initialized with the same arguments.

For an example of a ``Serializable`` object, see `Population`_, which is the internal state used by NSGA2. `NSGA2 itself`_ is only
``PartiallySerializable`` so that people can easily plug in their own mutation
and selection operations without worrying about serializations.

Serialization also makes your ``Designer`` run faster if its state size scales sublinearly in the number of observed Trials. For example, typical evolution algorithms and metaheuristics qualify, while GP-based algorithms do not because they use a non-parametric model. All you have to do is wrap your ``(Partially)SerializableDesigner`` into ``(Partially)SerializableDesignerPolicy``, which takes care of the state management.

.. _`Example`: https://github.com/google/vizier/blob/main/vizier/service/vizier_server.py
.. _`RandomPolicy`: https://github.com/google/vizier/blob/main/vizier/_src/algorithms/policies/random_policy.py
.. _`Designer`: https://github.com/google/vizier/_src/algorithms/core/abstractions.py
.. _`EmukitDesigner`: https://github.com/google/vizier/_src/algorithms/designers/emukit.py
.. _`DesignerPolicy`: https://github.com/google/vizier/_src/algorithms/policies/designer_policy.py
.. _`designer testing routine`: https://github.com/google/vizier/_src/algorithms/testing/test_runners.py
.. _`two options`: https://github.com/google/vizier/interfaces/serializable.py
.. _`Population`: https://github.com/google/vizier/_src/algorithms/evolution/numpy_populations.py
.. _`NSGA2 itself`: https://github.com/google/vizier/_src/algorithms/evolution/templates.py