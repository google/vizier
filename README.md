<figure>
<img src="docs/assets/vizier_logo.png" width=20% align="right"/>
</figure>

# Open Source Vizier: Reliable and Flexible Blackbox Optimization.

[![PyPI version](https://badge.fury.io/py/google-vizier.svg)](https://badge.fury.io/py/google-vizier)
![Continuous Integration (Core)](https://github.com/google/vizier/workflows/pytest_core/badge.svg)
![Continuous Integration (Algorithms)](https://github.com/google/vizier/workflows/pytest_algorithms/badge.svg)
![Continuous Integration (Benchmarks)](https://github.com/google/vizier/workflows/pytest_benchmarks/badge.svg)
![Continuous Integration (Docs)](https://github.com/google/vizier/workflows/docs/badge.svg)

[**Documentation**](https://oss-vizier.readthedocs.io/)
| [**Installation**](#installation)
| [**Citing Vizier**](#citing_vizier)


## What is Open Source (OSS) Vizier?

[OSS Vizier](https://arxiv.org/abs/2207.13676) is a Python-based service for blackbox optimization and research, based on [Google Vizier](https://dl.acm.org/doi/10.1145/3097983.3098043), one of the first hyperparameter tuning services designed to work at scale.

<figure>
<p align="center" width=65%>
<img src="docs/assets/oss_vizier_service.gif"/>
  <br>
  <em><b>OSS Vizier's distributed client-server system. Animation by Tom Small.</b></em>
</p>
</figure>

OSS Vizier's interface consists of three main APIs:

* [**User API:**](https://oss-vizier.readthedocs.io/en/latest/guides/index.html#for-users) Allows a user to setup an OSS Vizier Server, which can host blackbox optimization algorithms to serve multiple clients simultaneously in a fault-tolerant manner to tune their objective functions.
* [**Developer API:**](https://oss-vizier.readthedocs.io/en/latest/guides/index.html#for-developers) Defines abstractions and utilities for implementing new optimization algorithms for research and to be hosted in the service.
* [**Benchmarking API:**](https://oss-vizier.readthedocs.io/en/latest/guides/index.html#for-benchmarking) A wide collection of objective functions and methods to benchmark and compare algorithms.

Please see OSS Vizier's [ReadTheDocs documentation](https://oss-vizier.readthedocs.io/) for detailed information.




## Installation <a name="installation"></a>
To install the **core API**, the simplest way is to run:

```
pip install google-vizier
```

which will install the necessary dependencies from `requirements.txt`.

For **full installation** (to support **all algorithms and benchmarks**), run:

```
pip install google-vizier[extra]
```

which will also install the dependencies:

* `requirements-jax.txt`: Jax libraries shared by both algorithms and benchmarks.
* `requirements-tf.txt`: Tensorflow libraries shared by both algorithms and benchmarks.
* `requirements-algorithms.txt`: Additional repositories (e.g. Emukit) for algorithms.
* `requirements-benchmarks.txt`: Additional repositories (e.g. NASBENCH-201) for benchmarks.

**For both cases, you must build relevant protobuf libraries by running `build_protos.sh`.**

Check if all unit tests work by running `run_tests.sh`. OSS Vizier requires Python 3.10+, while client-only packages require Python 3.7+.


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
