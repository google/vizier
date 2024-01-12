# Copyright 2024 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

"""Exceptions used in the PythiaServer and Pythia Policy code."""


class PythiaError(Exception):
  """Abstract superclass of Pythia-related errors."""


class TemporaryPythiaError(PythiaError):
  """Vizier should attempt this interaction with a different Pythia task.

  Policy code should raise this to terminate and retry its
  interaction with Vizier.  Maps to an ABORTED gRPC error code.

  Use case: Catch-all errors where there's some hope that a re-run will succeed
    (e.g. OOM, timeouts, snapshot/version problems, or failure of
    nondeterministic algorithms).  If a Policy raises this error, Vizier will
    re-try the computation a modest number of times.
  """


class InactivateStudyError(PythiaError):
  """Pythia cannot handle this Study as configured; inactivate it.

  This terminates the interaction and returns an ARGUMENT_INVALID gRPC error
  code.  It's intended for unrecoverable errors, or misconfigued Studies.
  There will be no re-tries.
  """


class PythiaFallbackError(PythiaError):
  """The algorithm could not handle a StudyConfig corner case: fall back.

  This is typically raised by a special-purpose algorithm.  When raised,
  Vizier should re-try once with a more generic algorithm.
  """


class LoadTooLargeError(PythiaError):
  """Raised when the Pythia server is overly busy.

  If a Policy raises this error, it will be retried a (nearly) infinite number
  of times by the Vizier server, on the assumption that there will eventually
  be a server that's not overly busy.
  """


class CancelComputeError(Exception):
  """Raised by the Pythia server to kill Policy computations.

  This is raised in the Pythia compute thread (it is raised by the
  PolicySupporter object) when Vizier cancels the interaction. Policy code
  may catch this temporarily to do clean-up actions, but should re-raise it.

  Vizier may cancel the interaction when the SuggestionWorker needs to shut down
  (i.e. because it's preempted), or when the Pythia computation runs too long.
  """


class PythiaProtocolError(Exception):
  """There was a problem with the interaction between Vizier and Pythia.

  This implies a bug in the protocol code.  Do not raise or catch this in
  Policy code; please report any occurrences to vizier-team@google.com.
  Maps to an INTERNAL gRPC error code.
  """


class VizierDatabaseError(Exception):
  """Vizier had an error on a database operation.

  This can mean an attempt to access unavailable data, access control, etc.
  """
