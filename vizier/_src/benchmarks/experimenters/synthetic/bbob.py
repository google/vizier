"""Experimenter function implementations for BBOB functions."""

import functools
import hashlib
import math
from typing import Any, Callable

import numpy as np
from vizier import pyvizier


def DefaultBBOBProblemStatement(
    dimension: int,
    *,
    min_value: float = -5.0,
    max_value: float = 5.0) -> pyvizier.ProblemStatement:
  """Returns default BBOB ProblemStatement for given dimension."""
  problem_statement = pyvizier.ProblemStatement()
  space = problem_statement.search_space
  for dim in range(dimension):
    space.select_root().add_float_param(
        name=f"x{dim}", min_value=min_value, max_value=max_value)
  problem_statement.metric_information.append(
      pyvizier.MetricInformation(
          name="bbob_eval", goal=pyvizier.ObjectiveMetricGoal.MINIMIZE))
  return problem_statement


## Utility Functions for BBOB.
def LambdaAlpha(alpha: float, dim: int) -> np.ndarray:
  """The BBOB LambdaAlpha matrix creation function.

  Args:
    alpha: Function parameter.
    dim: Dimension of matrix created.

  Returns:
    Diagonal matrix of dimension dim with values determined by alpha.
  """
  lambda_alpha = np.zeros([dim, dim])
  for i in range(dim):
    exp = (0.5 * (float(i) / (dim - 1))) if dim > 1 else 0.5
    lambda_alpha[i, i] = alpha**exp
  return lambda_alpha


def ArrayMap(vector: np.ndarray, fn: Callable[[float], float]) -> np.ndarray:
  """Create a new array by mapping fn() to each element of the original array.

  Args:
    vector: ndarray to be mapped.
    fn: scalar function for mapping.

  Returns:
    New ndarray be values mapped by fn.
  """
  results = np.zeros(vector.shape)
  for i, v in enumerate(vector.flat):
    results.flat[i] = fn(v)
  return results


def Tosz(element: float) -> float:
  """The BBOB T_osz function.

  Args:
    element: float input.

  Returns:
    Tosz(input).
  """
  x_carat = 0.0 if element == 0 else math.log(abs(element))
  c1 = 10.0 if element > 0 else 5.5
  c2 = 7.9 if element > 0 else 3.1
  return np.sign(element) * math.exp(
      x_carat + 0.049 * (math.sin(c1 * x_carat) + math.sin(c2 * x_carat)))


def Tasy(vector: np.ndarray, beta: float) -> np.ndarray:
  """The BBOB Tasy function.

  Args:
    vector: ndarray
    beta: Function parameter

  Returns:
    ndarray with values determined by beta.
  """
  dim = len(vector)
  result = np.zeros([dim, 1])
  for i, val in enumerate(vector.flat):
    if val > 0:
      t = i / (dim - 1.0) if dim > 1 else 1
      exp = 1 + beta * t * (val**0.5)
    else:
      exp = 1
    result[i] = val**exp
  return result


def SIndex(dim: int, to_sz) -> float:
  """Calculate the BBOB s_i.

  Assumes i is 0-index based.

  Args:
    dim: dimension
    to_sz: values

  Returns:
    float representing SIndex(i, d, to_sz).
  """
  s = np.zeros([dim,])
  for i in range(dim):
    if dim > 1:
      s[i] = 10**(0.5 * (i / (dim - 1.0)))
    else:
      s[i] = 10**0.5
    if i % 2 == 0 and to_sz[i] > 0:
      s[i] *= 10
  return s


def Fpen(vector: np.ndarray) -> float:
  """The BBOB Fpen function.

  Args:
    vector: ndarray.

  Returns:
    float representing Fpen(vector).
  """
  return sum([max(0.0, (abs(x) - 5.0))**2 for x in vector.flat])


def _Hash(*seeds: Any) -> int:
  """Returns a stable hash that fits in a positive int64."""
  message = hashlib.sha256()
  for s in seeds:
    message.update(bytes(s))
  # Limit the size of the returned value to fit into a np.int64.
  return int(message.hexdigest()[:15], 16)


def _ToFloat(a: int, b: np.ndarray) -> np.ndarray:
  """Convert a%b where b is an int into a float on [-0.5, 0.5]."""
  return (np.int64(a) % b) / np.float64(b) - 0.5


@functools.lru_cache(maxsize=128)
def _R(dim: int, seed: int, *moreseeds: bytes) -> np.ndarray:
  """Returns an orthonormal rotation matrix.

  Args:
    dim: size of the resulting matrix.
    seed: int seed
    *moreseeds: Additional parameters to include in the hash.  Arguments must be
      interpretable as a buffer of bytes.

  Returns:
    2-dimensional, dim x dim, ndarray, representing a rotation matrix.
  """
  if seed == 0:
    return np.identity(dim)
  h = 0 if seed == 0 else _Hash(*((seed, dim) + moreseeds))
  a = np.arange(dim * dim, dtype=np.int64)
  # We make a vector of (loosely speaking) random entries.
  b = (_ToFloat(h + 17, a + 127) + _ToFloat(h + 61031, a + 197) +
       _ToFloat(h + 503, a + 293))

  # The "Q" part of a Q-R decomposition is orthonormal, so $result is a pure
  # rotation matrix.  If elements of $b are independent normal, then the
  # distribution of rotations is uniform over the hypersphere.
  return np.linalg.qr(b.reshape(dim, dim))[0]


## BBOB Functions.
def Sphere(arr: np.ndarray, seed: int = 0) -> float:
  """Implementation for BBOB Sphere function."""
  del seed
  return float(np.sum(arr * arr))


def Rastrigin(arr: np.ndarray, seed: int = 0) -> float:
  """Implementation for BBOB Rastrigin function."""
  dim = len(arr)
  arr.shape = (dim, 1)
  z = np.matmul(_R(dim, seed, b"R"), arr)
  z = Tasy(ArrayMap(z, Tosz), 0.2)
  z = np.matmul(_R(dim, seed, b"Q"), z)
  z = np.matmul(LambdaAlpha(10.0, dim), z)
  z = np.matmul(_R(dim, seed, b"R"), z)
  return float(10 * (dim - np.sum(np.cos(2 * math.pi * z))) +
               np.sum(z * z, axis=0))


def BuecheRastrigin(arr: np.ndarray, seed: int = 0) -> float:
  """Implementation for BBOB BuecheRastrigin function."""
  del seed
  dim = len(arr)
  arr.shape = (dim, 1)
  t = ArrayMap(arr, Tosz)
  l = SIndex(dim, arr) * t.flat

  term1 = 10 * (dim - np.sum(np.cos(2 * math.pi * l), axis=0))
  term2 = np.sum(l * l, axis=0)
  term3 = 100 * Fpen(arr)
  return float(term1 + term2 + term3)


def LinearSlope(arr: np.ndarray, seed: int = 0) -> float:
  """Implementation for BBOB LinearSlope function."""
  dim = len(arr)
  arr.shape = (dim, 1)
  r = _R(dim, seed, b"R")
  z = np.matmul(r, arr)
  result = 0.0
  for i in range(dim):
    s = 10**(i / float(dim - 1) if dim > 1 else 1)
    z_opt = 5 * np.sum(np.abs(r[i, :]))
    result += float(s * (z_opt - z[i]))
  return result


def AttractiveSector(arr: np.ndarray, seed: int = 0) -> float:
  """Implementation for BBOB Attractive Sector function."""
  dim = len(arr)
  arr.shape = (dim, 1)
  x_opt = np.array([1 if i % 2 == 0 else -1 for i in range(dim)])
  x_opt.shape = (dim, 1)
  z_vec = np.matmul(_R(dim, seed, b"R"), arr - x_opt)
  z_vec = np.matmul(LambdaAlpha(10.0, dim), z_vec)
  z_vec = np.matmul(_R(dim, seed, b"Q"), z_vec)

  result = 0.0
  for i in range(dim):
    z = z_vec[i, 0]
    s = 100 if z * x_opt[i] > 0 else 1
    result += (s * z)**2

  return math.pow(Tosz(result), 0.9)


def StepEllipsoidal(arr: np.ndarray, seed: int = 0) -> float:
  """Implementation for BBOB StepEllipsoidal function."""
  dim = len(arr)
  arr.shape = (dim, 1)
  z_hat = np.matmul(_R(dim, seed, b"R"), arr)
  z_hat = np.matmul(LambdaAlpha(10.0, dim), z_hat)
  z_tilde = np.array([
      math.floor(0.5 + z) if (z > 0.5) else (math.floor(0.5 + 10 * z) / 10)
      for z in z_hat.flat
  ])
  z_tilde = np.matmul(_R(dim, seed, b"Q"), z_tilde)
  s = 0.0
  for i, val in enumerate(z_tilde):
    exponent = 2.0 * float(i) / (dim - 1.0) if dim > 1.0 else 2.0
    s += 10.0**exponent * val**2
  value = max(abs(z_hat[0, 0]) / 1000, s)
  return 0.1 * value + Fpen(arr)


def RosenbrockRotated(arr: np.ndarray, seed: int = 0) -> float:
  """Implementation for BBOB RosenbrockRotated function."""
  dim = len(arr)
  r_x = np.matmul(_R(dim, seed, b"R"), arr)
  z = max(1.0, (dim**0.5) / 8.0) * r_x + 0.5 * np.ones((dim,))
  return float(sum(
      [100.0 * (z[i]**2 - z[i + 1])**2 + (z[i] - 1)**2
       for i in range(dim - 1)]))


def Ellipsoidal(arr: np.ndarray, seed: int = 0) -> float:
  """Implementation for BBOB Ellipsoidal function."""
  del seed
  dim = len(arr)
  arr.shape = (dim, 1)
  z_vec = ArrayMap(arr, Tosz)
  s = 0.0
  for i in range(dim):
    exp = 6.0 * i / (dim - 1) if dim > 1 else 6.0
    s += float(10**exp * z_vec[i] * z_vec[i])
  return s


def Discus(arr: np.ndarray, seed: int = 0) -> float:
  """Implementation for BBOB Discus function."""
  dim = len(arr)
  arr.shape = (dim, 1)
  r_x = np.matmul(_R(dim, seed, b"R"), arr)
  z_vec = ArrayMap(r_x, Tosz)
  return float(10**6 * z_vec[0] * z_vec[0]) + sum(
      [z * z for z in z_vec[1:].flat])


def BentCigar(arr: np.ndarray, seed: int = 0) -> float:
  """Implementation for BBOB BentCigar function."""
  dim = len(arr)
  arr.shape = (dim, 1)
  z_vec = np.matmul(_R(dim, seed, b"R"), arr)
  z_vec = Tasy(z_vec, 0.5)
  z_vec = np.matmul(_R(dim, seed, b"R"), z_vec)
  return float(z_vec[0]**2) + 10**6 * np.sum(z_vec[1:]**2)


def SharpRidge(arr: np.ndarray, seed: int = 0) -> float:
  """Implementation for BBOB SharpRidge function."""
  dim = len(arr)
  arr.shape = (dim, 1)
  z_vec = np.matmul(_R(dim, seed, b"R"), arr)
  z_vec = np.matmul(LambdaAlpha(10, dim), z_vec)
  z_vec = np.matmul(_R(dim, seed, b"Q"), z_vec)
  return z_vec[0, 0]**2 + 100 * np.sum(z_vec[1:]**2)**0.5


def DifferentPowers(arr: np.ndarray, seed: int = 0) -> float:
  """Implementation for BBOB DifferentPowers function."""
  dim = len(arr)
  z = np.matmul(_R(dim, seed, b"R"), arr)
  s = 0.0
  for i in range(dim):
    exp = 2 + 4 * i / (dim - 1) if dim > 1 else 6
    s += abs(z[i])**exp
  return s**0.5


def Weierstrass(arr: np.ndarray, seed: int = 0) -> float:
  """Implementation for BBOB Weierstrass function."""
  k_order = 12
  dim = len(arr)
  arr.shape = (dim, 1)
  z = np.matmul(_R(dim, seed, b"R"), arr)
  z = ArrayMap(z, Tosz)
  z = np.matmul(_R(dim, seed, b"Q"), z)
  z = np.matmul(LambdaAlpha(1.0 / 100.0, dim), z)
  f0 = sum([0.5**k * math.cos(math.pi * 3**k) for k in range(k_order)])

  s = 0.0
  for i in range(dim):
    for k in range(k_order):
      s += 0.5**k * math.cos(2 * math.pi * (3**k) * (z[i] + 0.5))

  return float(10 * (s / dim - f0)**3) + 10 * Fpen(arr) / dim


def SchaffersF7(arr: np.ndarray, seed: int = 0) -> float:
  """Implementation for BBOB Weierstrass function."""
  dim = len(arr)
  arr.shape = (dim, 1)
  if dim == 1:
    return 0.0
  z = np.matmul(_R(dim, seed, b"R"), arr)
  z = Tasy(z, 0.5)
  z = np.matmul(_R(dim, seed, b"Q"), z)
  z = np.matmul(LambdaAlpha(10.0, dim), z)

  s_arr = np.zeros(dim - 1)
  for i in range(dim - 1):
    s_arr[i] = float((z[i]**2 + z[i + 1]**2)**0.5)
  s = 0.0
  for i in range(dim - 1):
    s += s_arr[i]**0.5 + (s_arr[i]**0.5) * math.sin(50 * s_arr[i]**0.2)**2

  return (s / (dim - 1.0))**2 + 10 * Fpen(arr)


def SchaffersF7IllConditioned(arr: np.ndarray, seed: int = 0) -> float:
  """Implementation for BBOB SchaffersF7 Ill Conditioned."""
  dim = len(arr)
  arr.shape = (dim, 1)
  if dim == 1:
    return 0.0
  z = np.matmul(_R(dim, seed, b"R"), arr)
  z = Tasy(z, 0.5)
  z = np.matmul(_R(dim, seed, b"Q"), z)
  z = np.matmul(LambdaAlpha(1000.0, dim), z)

  s_arr = np.zeros(dim - 1)
  for i in range(dim - 1):
    s_arr[i] = float((z[i]**2 + z[i + 1]**2)**0.5)
  s = 0.0
  for i in range(dim - 1):
    s += s_arr[i]**0.5 + (s_arr[i]**0.5) * math.sin(50 * s_arr[i]**0.2)**2

  return (s / (dim - 1.0))**2 + 10 * Fpen(arr)


def GriewankRosenbrock(arr: np.ndarray, seed: int = 0) -> float:
  """Implementation for BBOB GriewankRosenbrock function."""
  dim = len(arr)
  r_x = np.matmul(_R(dim, seed, b"R"), arr)
  # Slightly off BBOB documentation in order to center optima at origin.
  # Should be: max(1.0, (dim**0.5) / 8.0) * r_x + 0.5 * np.ones((dim,)).
  z_arr = max(1.0, (dim**0.5) / 8.0) * r_x + np.ones((dim,))
  s_arr = np.zeros(dim)
  for i in range(dim - 1):
    s_arr[i] = 100.0 * (z_arr[i]**2 - z_arr[i + 1])**2 + (z_arr[i] - 1)**2

  total = 0.0
  for i in range(dim - 1):
    total += (s_arr[i] / 4000.0 - math.cos(s_arr[i]))

  return (10.0 * total) / (dim - 1) + 10


def Schwefel(arr: np.ndarray, seed: int = 0) -> float:
  """Implementation for BBOB Schwefel function."""
  del seed
  dim = len(arr)
  bernoulli_arr = np.array([pow(-1, i + 1) for i in range(dim)])
  x_opt = 4.2096874633 / 2.0 * bernoulli_arr
  x_hat = 2.0 * (bernoulli_arr * arr)  # Element-wise multiplication

  z_hat = np.zeros([dim, 1])
  z_hat[0, 0] = x_hat[0]
  for i in range(1, dim):
    z_hat[i, 0] = x_hat[i] + 0.25 * (x_hat[i - 1] - 2 * abs(x_opt[i - 1]))

  x_opt.shape = (dim, 1)
  z_vec = 100 * (
      np.matmul(LambdaAlpha(10, dim), z_hat - 2 * abs(x_opt)) + 2 * abs(x_opt))

  total = sum([z * math.sin(abs(z)**0.5) for z in z_vec.flat])

  return -(total / (100.0 * dim)) + 4.189828872724339 + 100 * Fpen(z_vec / 100)


def Katsuura(arr: np.ndarray, seed: int = 0) -> float:
  """Implementation for BBOB Katsuura function."""
  dim = len(arr)
  arr.shape = (dim, 1)
  r_x = np.matmul(_R(dim, seed, b"R"), arr)
  z_vec = np.matmul(LambdaAlpha(100.0, dim), r_x)
  z_vec = np.matmul(_R(dim, seed, b"Q"), z_vec)

  prod = 1.0
  for i in range(dim):
    s = 0.0
    for j in range(1, 33):
      s += abs(2**j * z_vec[i, 0] - round(2**j * z_vec[i, 0])) / 2**j
    prod *= (1 + (i + 1) * s)**(10.0 / dim**1.2)

  return (10.0 / dim**2) * prod - 10.0 / dim**2 + Fpen(arr)


def Lunacek(arr: np.ndarray, seed: int = 0) -> float:
  """Implementation for BBOB Lunacek function."""
  dim = len(arr)
  arr.shape = (dim, 1)
  mu0 = 2.5
  s = 1.0 - 1.0 / (2.0 * (dim + 20.0)**0.5 - 8.2)
  mu1 = -((mu0**2 - 1) / s)**0.5

  x_opt = np.array([mu0 / 2] * dim)
  x_hat = np.array([2 * arr[i, 0] * np.sign(x_opt[i]) for i in range(dim)])
  x_vec = x_hat - mu0
  x_vec.shape = (dim, 1)
  x_vec = np.matmul(_R(dim, seed, b"R"), x_vec)
  z_vec = np.matmul(LambdaAlpha(100, dim), x_vec)
  z_vec = np.matmul(_R(dim, seed, b"Q"), z_vec)

  s1 = sum([(val - mu0)**2 for val in x_hat])
  s2 = sum([(val - mu1)**2 for val in x_hat])
  s3 = sum([math.cos(2 * math.pi * z) for z in z_vec.flat])
  return min(s1, dim + s * s2) + 10.0 * (dim - s3) + 10**4 * Fpen(arr)


def Gallagher101Me(arr: np.ndarray, seed: int = 0) -> float:
  """Implementation for BBOB Gallagher101 function."""
  dim = len(arr)
  arr.shape = (dim, 1)

  num_optima = 101
  optima_list = [np.zeros([dim, 1])]
  for i in range(num_optima - 1):
    vec = np.zeros([dim, 1])
    for j in range(dim):
      alpha = (i * dim + j + 1.0) / (dim * num_optima + 2.0)
      assert alpha > 0
      assert alpha < 1
      vec[j, 0] = -5 + 10 * alpha
    optima_list.append(vec)

  c_list = [LambdaAlpha(1000, dim)]
  for i in range(num_optima - 1):
    alpha = 1000.0**(2.0 * (i) / (num_optima - 2))
    c_mat = LambdaAlpha(alpha, dim) / (alpha**0.25)
    c_list.append(c_mat)

  rotation = _R(dim, seed, b"R")
  max_value = -1.0
  for i in range(num_optima):
    w = 10 if i == 0 else (1.1 + 8.0 * (i - 1.0) / (num_optima - 2.0))
    diff = np.matmul(rotation, arr - optima_list[i])
    e = np.matmul(diff.transpose(), np.matmul(c_list[i], diff))
    max_value = max(max_value, w * math.exp(-float(e) / (2.0 * dim)))

  return Tosz(10.0 - max_value)**2 + Fpen(arr)


def Gallagher21Me(arr: np.ndarray, seed: int = 0) -> float:
  """Implementation for BBOB Gallagher21 function."""
  dim = len(arr)
  arr.shape = (dim, 1)

  num_optima = 21
  optima_list = [np.zeros([dim, 1])]
  for i in range(num_optima - 1):
    vec = np.zeros([dim, 1])
    for j in range(dim):
      alpha = (i * dim + j + 1.0) / (dim * num_optima + 2.0)
      assert alpha > 0
      assert alpha < 1
      vec[j, 0] = -5 + 10 * alpha
    optima_list.append(vec)

  c_list = [LambdaAlpha(1000, dim)]
  for i in range(num_optima - 1):
    alpha = 1000.0**(2.0 * (i) / (num_optima - 2))
    c_mat = LambdaAlpha(alpha, dim) / (alpha**0.25)
    c_list.append(c_mat)

  rotation = _R(dim, seed, b"R")
  max_value = -1.0
  for i in range(num_optima):
    w = 10 if i == 0 else (1.1 + 8.0 * (i - 1.0) / (num_optima - 2.0))
    diff = np.matmul(rotation, arr - optima_list[i])
    e = np.matmul(diff.transpose(), np.matmul(c_list[i], diff))
    max_value = max(max_value, w * math.exp(-float(e) / (2.0 * dim)))

  return Tosz(10.0 - max_value)**2 + Fpen(arr)


## Additional BBOB-like functions to test exploration.


def NegativeSphere(arr: np.ndarray, seed: int = 0) -> float:
  """Implementation for BBOB Sphere function."""
  dim = len(arr)
  arr.shape = (dim, 1)
  z = np.matmul(_R(dim, seed, b"R"), arr)
  return float(100 + np.sum(z * z) - 2 * (z[0]**2))


def NegativeMinDifference(arr: np.ndarray, seed: int = 0) -> float:
  """Implementation for NegativeMinDifference function."""
  dim = len(arr)
  arr.shape = (dim, 1)
  z = np.matmul(_R(dim, seed, b"R"), arr)
  min_difference = 10000
  for i in range(len(z) - 1):
    min_difference = min(min_difference, z[i + 1] - z[i])
  return 10.0 - float(min_difference) + 1e-8 * float(sum(arr))


def FonsecaFleming(arr: np.ndarray, seed: int = 0) -> float:
  """Implementation for FonsecaFleming function."""
  del seed
  return 1.0 - float(np.exp(-np.sum(arr * arr)))
