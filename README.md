# Open Source Python version of the Vizier API service.
OSS Vizier is an infrastructural API for blackbox optimization and hyperparameter tuning, based on Google's original internal [Vizier](https://dl.acm.org/doi/10.1145/3097983.3098043) service.

It allows a user to setup a main Vizier Server, which can host blackbox optimization algorithms to multiple clients simultaneously in a fault-tolerant manner.

# Table of Contents
* [Installation](#installation)
* [Running Vizier](#running-vizier)
  * [Running the Server](#running-the-server)
  * [Running a Client](#running-a-client)
* [Writing Pythia Policies](#writing-pythia-policies)
* [Tensorflow-Probability and Keras Modules](#tensorflow-probability-and-keras-modules)
* [Citing Vizier](#citing-vizier)

# Installation
The simplest way is to run the provided `install.sh`. It installs the necessary dependencies, and builds the relevant protobuf libraries needed for the service.

# Running Vizier
An example of the entire server + client loop running locally can be found in `vizier/service/vizier_client_test.py`.
We also present the core components of the example below:

## Running the Server
To start the Vizier service, the standard way via GRPC is to do the following on the host machine:

```
import grpc
import portpicker

# Setup Vizier Service and its data.
servicer = vizier_server.VizierService()

# Setup local networking.
port = portpicker.pick_unused_port()
address = f'localhost:{port}'

# Setup server.
server = grpc.server(futures.ThreadPoolExecutor(max_workers=100), ports=(port,))

vizier_service_pb2_grpc.add_VizierServiceServicer_to_server(self.servicer, self.server)
server.add_secure_port(address, grpc.local_server_credentials())
server.start()
```

## Running a client
Using the `address` created above, we may now create the client (e.g. on a different machine):

```
client = vizier_client.VizierClient(
    service_endpoint=address,
    owner_id='my_name',
    study_id='my_study_id')
```

The client may now send requests to the server and receive responses, for example:

```
client.list_trials()  # List out trials for `my_study_id`.
client.get_trial(trial_id='1')  # Get the first trial.
```

# Writing Pythia Policies
Writing blackbox optimization algorithms requires implementing the `Policy` interface, with pseudocode shown below:

```
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

An example is given in `random_policy.py`.

## Tensorflow-Probability and Keras Modules
For writing policies involving Bayesian Optimization, we have also provided Tensorflow-Probability (TFP) and Keras utility functions in `vizier/tfp/...` and `vizier/keras/...` respectively.

# Citing Vizier
If you found this code useful, please consider citing the [technical report (TBA)]() as well as the [original Vizier paper](https://dl.acm.org/doi/10.1145/3097983.3098043). Thanks!
