<figure>
<img src="assets/vizier_logo.png" width=15% align="right" />
</figure>

# Open Source Python implementation of the Vizier API and service.
OSS Vizier is an infrastructural API for blackbox optimization and hyperparameter tuning, based on Google's original internal [Vizier](https://dl.acm.org/doi/10.1145/3097983.3098043) service.

It allows a user to setup a main Vizier Server, which can host blackbox optimization algorithms to multiple clients simultaneously in a fault-tolerant manner.

# Table of Contents
* [Installation](#installation)
* [Running Vizier](#running-vizier)
  * [Running the Server](#running-the-server)
  * [Running a Client](#running-a-client)
* [Writing Pythia Policies](#writing-pythia-policies)
* [Citing Vizier](#citing-vizier)

# Installation
The simplest way is to run the provided `install.sh`. It installs the necessary dependencies, and builds the relevant protobuf libraries needed for the service.

# Running Vizier
An example of the entire server + client loop running locally can be found in `vizier/service/vizier_client_test.py`.
We also present the core components of the example below:

## Running the Server
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

vizier_service_pb2_grpc.add_VizierServiceServicer_to_server(self.servicer, self.server)
server.add_secure_port(address, grpc.local_server_credentials())
server.start()
```

## Running a client
The user may interact with the service via the client interface. The user first needs to setup the search space, metrics, and algorithm, in the `StudyConfig`:

```python
from vizier.pyvizier import oss

study_config = oss.StudyConfig() # Search space, metrics, and algorithm.
root = study_config.search_space.select_root() # "Root" params must exist in every trial.
root.add_float('learning_rate', min=1e-4, max=1e-2, scale=oss.ScaleType.LOG)
root.add_int('num_layers', min=1, max=5)
study_config.metrics.add('accuracy', goal=oss.ObjectiveMetricGoal.MAXIMIZE, min=0.0, max=1.0)
study_config.algorithm = oss.Algorithm.RANDOM_SEARCH
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

Note that the above can be called multiple times, one on each machine, to obtain `client_2`, `client_3`,...., all working on the same study, for tuning jobs which require multiple machines to compute the blackbox objective.

Each client may now send requests to the server and receive responses, for example:

```python
client.list_trials()  # List out trials for `my_study_id`.
client.get_trial(trial_id='1')  # Get the first trial.
```

# Writing Pythia Policies
Writing blackbox optimization algorithms requires implementing the `Policy` interface, with pseudocode shown below:

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

# Citing Vizier
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
