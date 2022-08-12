<figure>
<img src="docs/assets/vizier_logo.png" width=20% align="right" />
</figure>

# Open Source Vizier: Reliable and Flexible Blackbox Optimization.

![Continuous Integration](https://github.com/google/vizier/workflows/pytest/badge.svg)

[**Documentation**](https://oss-vizier.readthedocs.io/)
| [**Installation**](#installation)
| [**Code Structure**](#code_structure)
| [**Citing Vizier**](#citing_vizier)


## What is Open Source (OSS) Vizier?

[OSS Vizier](https://arxiv.org/abs/2207.13676) is a Python-based service for blackbox optimization and research, based on [Google Vizier](https://dl.acm.org/doi/10.1145/3097983.3098043), one of the first hyperparameter tuning services designed to work at scale. Please see OSS Vizier's [ReadTheDocs documentation](https://oss-vizier.readthedocs.io/) for detailed information.

OSS Vizier's interface consists of three main APIs:

* [**User API:**](https://oss-vizier.readthedocs.io/en/latest/guides/index.html#for-users) Allows a user to setup an OSS Vizier Server, which can host blackbox optimization algorithms to serve multiple clients simultaneously in a fault-tolerant manner to tune their objective functions.
* [**Developer API:**](https://oss-vizier.readthedocs.io/en/latest/guides/index.html#for-developers) Defines abstractions and utilities for implementing new optimization algorithms for research and to be hosted in the service.
* **Benchmarking API:** A wide collection of objective functions and methods to benchmark and compare algorithms.




## Installation <a name="installation"></a>
The simplest way is to run:

```
pip install google-vizier
```

which will download the code and launch the provided `install.sh`. This script installs the necessary dependencies, and builds the relevant protobuf libraries needed for the service. Check if all unit tests work by running `run_tests.sh`. OSS Vizier requires Python 3.9+.


## Code Structure <a name="code_structure"></a>

### Frequently used import targets <a name="freq_import_targets"></a>

Includes a brief summary of important symbols and modules.

#### Service users <a name="service_users"></a>

If you write client code interacting with the OSS Vizier service, use these
import targets:

* `from vizier.service import pyvizier as vz`: Exposes the same set of symbol names as `vizier.pyvizier`. `vizier.service.pyvizier.Foo` is a subclass or an alias of `vizier.pyvizier.Foo`, and can be converted into protobufs.
<!-- TODO(b/226560768): Update this entry after the clean up -->
* `from vizier.service import ...`: Include binaries and internal utilities.

#### Algorithm Developers

If you write algorithm code (Designers or Pythia policies) in OSS Vizier, use
these import targets:

* **`from vizier import pyvizier as vz`**: Pure python building blocks of OSS Vizier. Cross-platform code, including pythia policies, must use this pyvizier instance.
  * `Trial` and `ProblemStatement` are important classes.
* **`from vizier.pyvizier import converters`**: Convert between pyvizier objects and numpy arrays.
  * `TrialToNumpyDict`: Converts parameters (and metrics) into a dict of numpy arrays. Preferred conversion method if you intended to train an embedding of categorical/discrete parameters, or data includes missing parameters or metrics.
  * `TrialToArrayConverter`: Converts parameters (and metrics) into an array.
* `from vizier.interfaces import serializable`
  * `PartiallySerializable`, `Serializable`

##### Algorithm abstractions
* **`from vizier import pythia`**
  * `Policy`, `PolicySupporter`: Key abstractions
  * `LocalPolicyRunner`: Use it for running a `Policy` in RAM.
* **`from vizier import algorithms`**
  * `Designer`:
  * `DesignerPolicy`: Wraps `Designer` into a pythia Policy.
  * `GradientFreeMaximizer`: For optimizing acquisition functions.
  * `(Partially)SerializableDesigner`: Designers who wish to optimize performance by saving states.

##### Tensorflow modules
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
If you found this code useful, please consider citing the [OSS Vizier paper](https://arxiv.org/abs/2207.13676) as well as the [Google Vizier paper](https://dl.acm.org/doi/10.1145/3097983.3098043). Thanks!

```
@inproceedings{oss_vizier,
  author    = {Xingyou Song and
               Sagi Perel and
               Chansoo Lee and
               Greg Kochanski and
               Daniel Golovin},
  title     = {Open Source Vizier: Distributed Infrastructure and API for Reliable and Flexible Blackbox Optimization},
  booktitle = {Automated Machine Learning Conference, Systems Track (AutoML-Conf Systems)},
  year      = {2022},
}
@inproceedings{google_vizier,
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
