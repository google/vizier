<figure>
<img src="assets/vizier_logo.png" width=20% align="right" />
</figure>

# Open Source Vizier: Reliable and Flexible Blackbox Optimization.
Open Source (OSS) Vizier is a Python-based interface for blackbox optimization and research, based on Google's original internal [Vizier](https://dl.acm.org/doi/10.1145/3097983.3098043), one of the first hyperparameter tuning services designed to scale.

It consists of two main APIs:

* **User API:** Allows a user to setup a main Vizier Server, which can host blackbox optimization algorithms to serve multiple clients simultaneously in a fault-tolerant manner to tune their objective functions.
* **Developer API:** Defines abstractions and utilities for implementing new optimization algorithms for research and benchmarking.

[TOC]


## Installation
The simplest way is to run the provided `install.sh`. It installs the necessary dependencies, and builds the relevant protobuf libraries needed for the service.

## User API: Running Vizier
An example of the entire server + client loop running locally can be found in `vizier/service/vizier_client_test.py`.
We also present the core components of the example below:

### Running the Server
An example is provided at `vizier/run_vizier_server.py`. To start the Vizier service, the standard way via GRPC is to do the following on the host machine:

```python
import grpc
import portpicker

# Setup Vizier Service and its data.
servicer = vizier_server.VizierService()

# Setup local networking.
port = portpicker.pick_unused_port()
address = f'localhost:{port}'

# Setup server.
server = grpc.server(futures.ThreadPoolExecutor(max_workers=100))

vizier_service_pb2_grpc.add_VizierServiceServicer_to_server(servicer, server)
server.add_secure_port(address, grpc.local_server_credentials())
server.start()
```

### Running a client
The user may interact with the service via the client interface. The user first needs to setup the search space, metrics, and algorithm, in the `StudyConfig`:

```python
from vizier.service import pyvizier as vz

study_config = vz.StudyConfig() # Search space, metrics, and algorithm.
root = study_config.search_space.select_root() # "Root" params must exist in every trial.
root.add_float('learning_rate', min=1e-4, max=1e-2, scale=vz.ScaleType.LOG)
root.add_int('num_layers', min=1, max=5)
study_config.metrics.add('accuracy', goal=vz.ObjectiveMetricGoal.MAXIMIZE, min=0.0, max=1.0)
study_config.algorithm = vz.Algorithm.RANDOM_SEARCH
```

Using the `address` created above in the server section, we may now create the client (e.g. on a worker machine different from the server):

```python
from vizier.service import vizier_client

client = vizier_client.create_or_load_study(
    service_endpoint=address,
    owner_id='my_name',
    client_id='my_client_id',
    study_display_name='cifar10',
    study_config=study_config)
```

Note that the above can be called multiple times, one on each machine, to obtain `client_2`, `client_3`,..., all working on the same study, especially for tuning jobs which require multiple machines to parallelize the workload.

Each client may now send requests to the server and receive responses, for example:

```python
client.list_trials()  # List out trials for the corresponding study.
client.get_trial(trial_id='1')  # Get the first trial.
```

The default usage is to tune a user defined blackbox objective `_evaluate_trial()`, with an example shown below:

```python
while suggestions := client.get_suggestions(count=1)
  # Evaluate the suggestion(s) and report the results to Vizier.
  for trial in suggestions:
    metrics = _evaluate_trial(trial.parameters)
    client.complete_trial(metrics, trial_id=trial.id)
```

The Vizier service is designed to handle multiple concurrent clients all requesting suggestions and returning metrics.

## Developer API: Writing Algorithms
Writing blackbox optimization algorithms requires implementing the `Policy` interface as part of Vizier's Pythia service, with pseudocode shown below:

```python
class MyPolicy(Policy):
  def __init__(self, ...):
    self.policy_supporter = PolicySupporter(...)  # Used to obtain old trials.

  def suggest(self, request: SuggestRequest) -> List[SuggestDecision]:
    """Suggests trials to be evaluated."""
    suggestions = []
    for _ in range(request.count):
      old_trials = self.policy_supporter.GetTrials(...)
      trial = make_new_trial(old_trials, request.study_config)
      suggestions.append(base.SuggestDecision(trial))
    return suggestions

  def early_stop(self, request: EarlyStopRequest) -> List[EarlyStopDecision]:
    """Selects trials to stop from the request."""
    old_trials = self.policy_supporter.GetTrials(...)
    trials_to_stop = determine_trials_to_stop(old_trials, request.trial_ids)
    return [base.EarlyStopDecision(id) for id in trials_to_stop]
```

An example is given in `vizier/pythia/policies/random_policy.py`.


## Citing Vizier
If you found this code useful, please consider citing the [technical report (TBA)]() as well as the [original Vizier paper](https://dl.acm.org/doi/10.1145/3097983.3098043). Thanks!

```
@inproceedings{oss_vizier,
  author    = {Xingyou Song and
               Sagi Perel and
               Chansoo Lee and
               Greg Kochanski and
               Daniel Golovin},
  title     = {Open Source Vizier: Distributed Infrastructure and API for Reliable and Flexible Blackbox Optimization},
  year      = {2022},
}
@inproceedings{original_vizier,
  author    = {Daniel Golovin and
               Benjamin Solnik and
               Subhodeep Moitra and
               Greg Kochanski and
               John Karro and
               D. Sculley},
  title     = {Google Vizier: {A} Service for Black-Box Optimization},
  booktitle = {Proceedings of the 23rd {ACM} {SIGKDD} International Conference on
               Knowledge Discovery and Data Mining, Halifax, NS, Canada, August 13
               - 17, 2017},
  pages     = {1487--1495},
  publisher = {{ACM}},
  year      = {2017},
  url       = {https://doi.org/10.1145/3097983.3098043},
  doi       = {10.1145/3097983.3098043},
}
```


## Code structure

### Frequently used import targets

Includes a brief summary of important symbols and modules.

#### Service users
* `from vizier.service import pyvizier as vz`: Exposes the same set of symbol names as `vizier.pyvizier`. `vizier.service.pyvizier.Foo` is a subclass or an alias of `vizier.pyvizier.Foo`, and can be converted into protobufs.
<!-- TODO(b/226560768): Update this entry after the clean up -->
* `from vizier.service import ...`: Include binaries and internal utilities.

#### Developer essentials
* **`from vizier import pyvizier as vz`**: Pure python building blocks of Vizier. Cross-platform code including pythia policies must use this pyvizier instance.
  * `Trial` and `StudyConfig` are most important classes.
* **`from vizier.pyvizier import converters`**: Convert between pyvizier objects and numpy arrays.
  * `TrialToNumpyDict`: Converts parameters (and metrics) into a dict of numpy arrays. Preferred conversion method if you intended to train an embedding of categorical/discrete parameters, or data includes missing parameters or metrics.
  * `TrialToArrayConverter`: Converts parameters (and metrics) into an array.
* `from vizier.interfaces import serializable`
  * `PartiallySerializable`, `Serializable`

#### Algorithm abstractions
* **`from vizier import pythia`**
  * `Policy`, `PolicySupporter`: Key abstractions
  * `LocalPolicyRunner`: Use it for running a `Policy` in RAM.
* **`from vizier import algorithms`**
  * `Designer`:
  * `DesignerPolicy`: Wraps `Designer` into a pythia Policy.
  * `GradientFreeMaximizer`: For optimizing acquisition functions.
  * `(Partially)SerializableDesigner`: Designers who wish to optimize performance by saving states.

#### Tensorflow modules
* **`from vizier import tfp`**: Tensorflow-probability utilities.
  * `acquisitions`: Acquisition functions module.
     * `AcquisitionFunction`: abstraction
     * `UpperConfidenceBound`, `ExpectedImprovement`, etc.
  * `bijectors`: Bijectors module.
    * `YeoJohnson`: Implements both Yeo-Johnson and Box-Cox transformations.
    * `optimal_power_transformation`: Returns the optimal power transformation.
    * `flip_sign`: returns a sign-flip bijector.
* **`from vizier import keras as vzk`**
  * `vzk.layers`: Layers usually wrapping tfp classes
      * `variable_from_prior`: Utility layer for handling regularized variables.
  * `vzk.optim`: Wrappers around optimizers in tfp or keras
  * `vzk.models`: Most of the useful models don't easily fit into keras' Model abstraction, but we may add some for display.
