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

"""Eagle strategy designer serialization."""

import json
from typing import Any
import attr
import numpy as np
from vizier import pyvizier as vz
from vizier._src.algorithms.designers.eagle_strategy import eagle_strategy_utils
from vizier.interfaces import serializable

FireflyPool = eagle_strategy_utils.FireflyPool
Firefly = eagle_strategy_utils.Firefly
EagleStrategyUtils = eagle_strategy_utils.EagleStrategyUtils
OBJECTIVE_NAME = eagle_strategy_utils.OBJECTIVE_NAME


class PartialFireflyPoolEncoder(json.JSONEncoder):
  """Eagle strategy pool partial encoder.

  The encoder encodes the '_pool' dictionary, 'capacity', '_last_id' and
  '_max_fly_id' into a string format.

  The encoder does not store the EagleStrategyUtils as its state only depends
  on the random generator which is handled separately.
  """

  def default(self, o: Any) -> Any:
    if isinstance(o, FireflyPool):
      return {
          'capacity': o._capacity,  # pylint: disable=protected-access
          '_last_id': o._last_id,  # pylint: disable=protected-access
          '_max_fly_id': o._max_fly_id,  # pylint: disable=protected-access
          '_pool': o._pool,  # pylint: disable=protected-access
      }
    elif isinstance(o, Firefly):
      return {
          'id_': o.id_,
          'perturbation': o.perturbation,
          'generation': o.generation,
          'trial': o.trial,
      }
    elif isinstance(o, vz.Trial):
      return {
          'parameters': o.parameters.as_dict(),
          'objective': o.final_measurement.metrics[
              eagle_strategy_utils.OBJECTIVE_NAME
          ].value,
          'infeasibility_reason': o.infeasibility_reason,
      }
    else:
      return json.JSONEncoder.default(self, o)


@attr.define
class FireflyPoolDecoder:
  """Eagle strategy pool decoder.

  Fully restores the state of the FireflyPool.

  Attributes:
    utils: EagleStrategyUtils initialized with the appropriate random generator.
  """

  _utils: EagleStrategyUtils

  def decode(self, obj: Any) -> FireflyPool:
    """Decodes a string object to partial FireflyPool."""
    obj_dict = json.loads(obj)
    restored_pool = {}

    # Check that all keys appear in restored dictionary.
    missing_keys = set(['_pool', 'capacity', '_last_id', '_max_fly_id']) - set(
        obj_dict.keys())
    if missing_keys:
      raise serializable.HarmlessDecodeError(
          "Couldn't load FireflyPool from metadata. The following keys are "
          'missing: %s' % str(missing_keys))

    # Restore FireFly objects in the pool.
    for id_, fly in obj_dict['_pool'].items():
      trial = vz.Trial(parameters=fly['trial']['parameters'])
      trial.complete(
          measurement=vz.Measurement(
              metrics={'objective': fly['trial'][OBJECTIVE_NAME]}
          ),
          infeasibility_reason=fly['trial']['infeasibility_reason'],
      )
      restored_pool[int(id_)] = Firefly(
          id_=fly['id_'],
          perturbation=fly['perturbation'],
          generation=fly['generation'],
          trial=trial,
      )

    restored_capacity = int(obj_dict['capacity'])
    restored_firefly_pool = FireflyPool(
        capacity=restored_capacity, utils=self._utils
    )
    # pylint: disable=protected-access
    restored_firefly_pool._pool = restored_pool
    restored_firefly_pool._last_id = int(obj_dict['_last_id'])
    restored_firefly_pool._max_fly_id = int(obj_dict['_max_fly_id'])
    return restored_firefly_pool


def partially_serialize_firefly_pool(firefly_pool: FireflyPool) -> str:
  """Serialize parts of the FireflyPool."""
  return json.dumps(firefly_pool, cls=PartialFireflyPoolEncoder)


def restore_firefly_pool(utils: EagleStrategyUtils, obj: str) -> FireflyPool:
  """Fully restore the FireflyPool."""
  return FireflyPoolDecoder(utils).decode(obj)


def serialize_rng(rng: np.random.Generator) -> str:
  """Serialize Numpy Random Genertor."""
  return json.dumps(rng.bit_generator.state)


def restore_rng(obj: str) -> np.random.Generator:
  """Restore Numpy Random Genertor."""
  rng = np.random.default_rng()
  rng.bit_generator.state = json.loads(obj)
  return rng
