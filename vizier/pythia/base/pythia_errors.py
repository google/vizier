"""Exceptions used in the PythiaServer and Pythia Policy code."""


class PythiaError(Exception):
  """Abstract superclass of Pythia-related errors."""


class TemporaryPythiaError(PythiaError):
  """Vizier should attempt this interaction with a different Pythia task.

  Policy code should raise this to terminate and retry its
  interaction with Vizier.  Maps to an ABORTED gRPC error code.

  Use case: Catch-all errors where there's some hope that a re-run will succeed
    (e.g. OOM, timeouts, snapshot/version problems, or failure of
    nondeterministic algorithms).
  """


class CachedPolicyIsStaleError(PythiaError):
  """Raise this if the cached Policy instance is too stale.

  Raised by Policy code.  Because Policy instances are cached,
  they could potentially become stale.  If so, code should raise this,
  and the computation will be restarted with a freshly created Policy
  instance.
  """


class InactivateStudyError(PythiaError):
  """Pythia cannot handle this Study as configured; inactivate it.

  This terminates the interaction and returns an ARGUMENT_INVALID gRPC error
  code.  It's intended for unrecoverable errors, or misconfigued Studies.
  There will be no re-tries.
  """


class LoadTooLargeError(PythiaError):
  """Raised when the Pythia server is overly busy."""


class CancelledByVizierError(PythiaError):
  """Indicates that Vizier cancelled the interaction."""
  pass


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
