<figure>
<img src="assets/vizier_logo.png" width=20% align="right" />
</figure>

# Open Source Vizier: Reliable and Flexible Blackbox Optimization.
Open Source (OSS) Vizier is a Python-based interface for blackbox optimization and research, based on Google's original internal [Vizier](https://dl.acm.org/doi/10.1145/3097983.3098043), one of the first hyperparameter tuning services designed to work at scale.

It consists of two main APIs:

* **User API:** Allows a user to setup a main Vizier Server, which can host blackbox optimization algorithms to serve multiple clients simultaneously in a fault-tolerant manner to tune their objective functions.
* **Developer API:** Defines abstractions and utilities for implementing new optimization algorithms for research and benchmarking.

# Table of Contents
1. [Installation](#installation)
2. [User API: Running Vizier](#user_api)
    1. [Running the Server](#running_server)
    2. [Running the Client](#running_client)
3. [Developer API: Writing Algorithms](#developer_api)
4. [Code Structure](#code_structure)
    1. [Frequently Used Import Targets](#freq_import_targets)
5. [Citing Vizier](#citing_vizier)


## Installation <a name="installation"></a>
The simplest way is to run the provided `install.sh`. It installs the necessary dependencies, and builds the relevant protobuf libraries needed for the service. Check if all unit tests work by running `run_tests.sh`.

## User API: Running Vizier <a name="user_api"></a>
An example of the entire server + client loop running locally can be found in the unit test file `vizier/service/vizier_client_test.py`. A manual demo can be found in the `/demos/` folder. To run the manual demo, run the following command to start the server:

```
python run_vizier_server.py
```

which will print out an `address` of the form `localhost:[PORT]`.

Then run the following command using the `address` to start the client:

```
python run_vizier_client.py --address="localhost:[PORT]"
```

We explain how the core components work below:

### Running the Server <a name="running_server"></a>
To start the Vizier service, the standard way via GRPC is to do the following on the host machine:

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

### Running a client <a name="running_client"></a>
The user may then interact with the service via the client interface. The user first needs to setup the search space, metrics, and algorithm, in the `StudyConfig`:

```python
from vizier.service import pyvizier as vz

study_config = vz.StudyConfig() # Search space, metrics, and algorithm.
root = study_config.search_space.select_root() # "Root" params must exist in every trial.
root.add_float_param('learning_rate', min_value=1e-4, max_value=1e-2, scale_type=vz.ScaleType.LOG)
root.add_int_param('num_layers', min_value=1, max_value=5)
study_config.metric_information.append(vz.MetricInformation(name='accuracy', goal=vz.ObjectiveMetricGoal.MAXIMIZE, min_value=0.0, max_value=1.0))
study_config.algorithm = vz.Algorithm.RANDOM_SEARCH
```

Using the `address` created above in the server section, we may now create the client (e.g. on a worker machine different from the server):

```python
from vizier.service import vizier_client

client = vizier_client.create_or_load_study(
    service_endpoint=address,  # Same address as server.
    owner_id='my_name',
    client_id='my_client_id',
    study_display_name='cifar10',
    study_config=study_config)
```

Note that the above can be called multiple times, one on each machine, to obtain `client_2`, `client_3`,..., all working on the same study, especially for tuning jobs which require multiple machines to parallelize the workload.

Each client may now send requests to the server and receive responses, for example:

```python
client.list_trials()  # List out trials for the corresponding study.
client.get_trial(trial_id=1)  # Get the first trial.
```

The default usage is to tune a user defined blackbox objective `evaluate_trial()`, with an example shown below:

```python
suggestions = client.get_suggestions(suggestion_count=5)  # Batch of 5 suggestions.
# Evaluate the suggestion(s) and report the results to Vizier.
for trial in suggestions:
  measurement = evaluate_trial(trial)
  client.complete_trial(trial_id, measurement)
```

The Vizier service is designed to handle multiple concurrent clients all requesting suggestions and returning metrics.

## Developer API: Writing Algorithms <a name="developer_api"></a>
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

An example is given in `vizier/_src/algorithms/policies/random_policy.py`.

## Code structure <a name="code_structure"></a>

### Frequently used import targets <a name="freq_import_targets"></a>

Includes a brief summary of important symbols and modules.

#### Service users <a name="service_users"></a>
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


## Citing Vizier <a name="citing_vizier"></a>
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
