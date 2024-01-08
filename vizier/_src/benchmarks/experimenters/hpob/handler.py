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

"""HPOB Handler copied (with slightly modified filesystem logic) from

https://github.com/releaunifreiburg/HPO-B/blob/main/hpob_handler.py.
"""
# pylint:skip-file

import json
import os

import functools
import numpy as np
import xgboost as xgb

Open = open
Exists = os.path.exists
IsDir = os.path.isdir


class HPOBHandler:

  def __init__(self,
               root_dir="hpob-data/",
               mode="v3-test",
               surrogates_dir="saved-surrogates/"):
    """Constructor for the HPOBHandler.

    Inputs:
        * root_dir: path to directory with the benchmark data.
        * mode: mode name indicating how to load the data. Options:
            - v1: Loads HPO-B-v1
            - v2: Loads HPO-B-v2
            - v3: Loads HPO-B-v3
            - v3-test: Loads only the meta-test split from HPO-B-v3
            - v3-train-augmented: Loads all splits from HPO-B-v3, but
            augmenting the meta-train data with the less frequent
            search-spaces.
        * surrogates_dir: path to directory with surrogates models.
    """

    print("Loading HPO-B handler")
    self.mode = mode
    self.surrogates_dir = surrogates_dir
    self.seeds = ["test0", "test1", "test2", "test3", "test4"]

    if self.mode == "v3-test":
      self.load_data(root_dir, only_test=True)
    elif self.mode == "v3-train-augmented":
      self.load_data(root_dir, only_test=False, augmented_train=True)
    elif self.mode in ["v1", "v2", "v3"]:
      self.load_data(root_dir, version=self.mode, only_test=False)
    else:
      raise ValueError("Provide a valid mode")

    self.surrogates_stats = HPOBHandler._cached_surrogates_stats(surrogates_dir)

  @classmethod
  @functools.lru_cache(maxsize=128)
  def _cached_load_data(cls,
                        rootdir="",
                        version="v3",
                        only_test=True,
                        augmented_train=False):

    meta_train_data = {}
    meta_validation_data = {}
    meta_test_data = {}

    print("Loading data...")
    meta_train_augmented_path = os.path.join(
        rootdir, "meta-train-dataset-augmented.json")
    meta_train_path = os.path.join(rootdir, "meta-train-dataset.json")
    meta_test_path = os.path.join(rootdir, "meta-test-dataset.json")
    meta_validation_path = os.path.join(rootdir, "meta-validation-dataset.json")
    bo_initializations_path = os.path.join(rootdir, "bo-initializations.json")

    with Open(meta_test_path, "rb") as f:
      meta_test_data = json.load(f)

    with Open(bo_initializations_path, "rb") as f:
      bo_initializations = json.load(f)

    if not only_test:
      if augmented_train or version == "v1":
        with Open(meta_train_augmented_path, "rb") as f:
          meta_train_data = json.load(f)
      else:
        with Open(meta_train_path, "rb") as f:
          meta_train_data = json.load(f)
      with Open(meta_validation_path, "rb") as f:
        meta_validation_data = json.load(f)

    if version != "v3":
      temp_data = {}
      for search_space in meta_train_data.keys():
        temp_data[search_space] = {}

        for dataset in meta_train_data[search_space].keys():
          temp_data[search_space][dataset] = meta_train_data[search_space][
              dataset]

        if search_space in meta_test_data.keys():
          for dataset in meta_test_data[search_space].keys():
            temp_data[search_space][dataset] = meta_test_data[search_space][
                dataset]

          for dataset in meta_validation_data[search_space].keys():
            temp_data[search_space][dataset] = meta_validation_data[
                search_space][dataset]
      meta_test_data = temp_data

    return meta_train_data, meta_validation_data, meta_test_data, bo_initializations

  @classmethod
  @functools.lru_cache(maxsize=128)
  def _cached_surrogates_stats(cls, surrogates_dir: str):
    surrogates_file = surrogates_dir + "summary-stats.json"
    if (Exists(surrogates_file) and not IsDir(surrogates_file)):
      with Open(surrogates_file, "rt") as f:
        surrogates_stats = json.load(f)
    else:
      surrogates_stats = None
    return surrogates_stats

  def load_data(self,
                rootdir="",
                version="v3",
                only_test=True,
                augmented_train=False):
    """
        Loads data with some specifications.
        Inputs:
            * root_dir: path to directory with the benchmark data.
            * version: name indicating what HPOB version to use. Options: v1,
            v2, v3).
            * Only test: Whether to load only testing data (valid only for
            version v3).  Options: True/False
            * augmented_train: Whether to load the augmented train data (valid
            only for version v3). Options: True/False
    """

    self.meta_train_data, self.meta_validation_data, self.meta_test_data, self.bo_initializations = HPOBHandler._cached_load_data(
        rootdir=rootdir,
        version=version,
        only_test=only_test,
        augmented_train=augmented_train)

  def normalize(self, y, y_min=None, y_max=None):

    if y_min is None:
      return (y - np.min(y)) / (np.max(y) - np.min(y))
    else:
      return (y - y_min) / (y_max - y_min)

  def evaluate(self,
               bo_method=None,
               search_space_id=None,
               dataset_id=None,
               seed=None,
               n_trials=10):
    """Evaluates a method on the benchmark with discretized search-spaces.

    Inputs:
        * bo_method: object to evaluate. It should have a function (class
        method) named 'observe_and_suggest'.
        * search_space_id: Identifier of the search spaces for the
        evaluation. Option: see original paper.
        * dataset_id: Identifier of the dataset for the evaluation. Options:
        see original paper.
        * seed: Identifier of the seed for the evaluation. Options: test0,
        test1, test2, test3, test4.
        * trails: Number of trials (iterations on the opoitmization).
    Ooutput:
        * a list with the maximumu performance (incumbent) for every trial.
    """

    assert bo_method != None, "Provide a valid method object for evaluation."
    assert hasattr(bo_method, "observe_and_suggest"), (
        "The provided  object does not have a method called "
        "´observe_and_suggest´")
    assert search_space_id != None, ("Provide a valid search space id. See "
                                     "documentatio for valid obptions.")
    assert dataset_id != None, ("Provide a valid dataset_id. See documentation "
                                "for valid options.")
    assert seed != None, ("Provide a valid initialization. Valid options are: "
                          "test0, test1, test2, test3, test4.")

    n_initial_evaluations = 5
    X = np.array(self.meta_test_data[search_space_id][dataset_id]["X"])
    y = np.array(self.meta_test_data[search_space_id][dataset_id]["y"])
    y = self.normalize(y)
    data_size = len(X)

    pending_evaluations = list(range(data_size))
    current_evaluations = []

    init_ids = self.bo_initializations[search_space_id][dataset_id][seed]

    for i in range(n_initial_evaluations):
      idx = init_ids[i]
      pending_evaluations.remove(idx)
      current_evaluations.append(idx)

    max_accuracy_history = [np.max(y[current_evaluations])]
    for i in range(n_trials):

      idx = bo_method.observe_and_suggest(X[current_evaluations],
                                          y[current_evaluations],
                                          X[pending_evaluations])
      idx = pending_evaluations[idx]
      pending_evaluations.remove(idx)
      current_evaluations.append(idx)
      max_accuracy_history.append(np.max(y[current_evaluations]))

    return max_accuracy_history

  def evaluate_continuous(self,
                          bo_method=None,
                          search_space_id=None,
                          dataset_id=None,
                          seed=None,
                          n_trials=10):
    """
        Evaluates a method on the benchmark with continuous search-spaces.
        Inputs:
            * bo_method: object to evaluate. It should have a function (class
            method) named 'observe_and_suggest'.
            * search_space_id: Identifier of the search spaces for the
            evaluation. Option: see original paper.
            * dataset_id: Identifier of the dataset for the evaluation. Options:
            see original paper.
            * seed: Identifier of the seed for the evaluation. Options: test0,
            test1, test2, test3, test4.
            * trails: Number of trials (iterations on the opoitmization).
        Ooutput:
            * a list with the maximumu performance (incumbent) for every trial.
        """

    assert bo_method != None, "Provide a valid method object for evaluation."
    assert hasattr(bo_method, "observe_and_suggest"), (
        "The provided  object does not have a method called "
        "´observe_and_suggest´")
    assert search_space_id != None, ("Provide a valid search space id. See "
                                     "documentatio for valid obptions.")
    assert dataset_id != None, ("Provide a valid dataset_id. See documentation "
                                "for valid options.")
    assert seed != None, ("Provide a valid initialization. Valid options are: "
                          "test0, test1, test2, test3, test4.")

    surrogate_name = "surrogate-" + search_space_id + "-" + dataset_id
    bst_surrogate = xgb.Booster()
    bst_surrogate.load_model(self.surrogates_dir + surrogate_name + ".json")

    n_initial_evaluations = 5
    X = np.array(self.meta_test_data[search_space_id][dataset_id]["X"])
    y = np.array(self.meta_test_data[search_space_id][dataset_id]["y"])
    y_min = self.surrogates_stats[surrogate_name]["y_min"]
    y_max = self.surrogates_stats[surrogate_name]["y_max"]
    dim = X.shape[1]
    current_evaluations = []
    init_ids = self.bo_initializations[search_space_id][dataset_id][seed]

    for i in range(n_initial_evaluations):
      idx = init_ids[i]
      current_evaluations.append(idx)

    x_observed = X[current_evaluations]
    y_observed = y[current_evaluations]

    max_accuracy_history = []

    for i in range(n_trials):
      y_tf_observed = self.normalize(y_observed, y_min, y_max)
      y_tf_observed = np.clip(y_tf_observed, 0, 1)
      best_f = np.max(y_tf_observed)
      max_accuracy_history.append(best_f)

      new_x = bo_method.observe_and_suggest(x_observed, y_tf_observed)
      x_q = xgb.DMatrix(new_x.reshape(-1, dim))
      new_y = bst_surrogate.predict(x_q)

      y_observed = np.append(y_observed, new_y).reshape(-1, 1)
      x_observed = np.append(x_observed, new_x).reshape(-1, x_observed.shape[1])

    y_tf_observed = self.normalize(y_observed, y_min, y_max)
    y_tf_observed = np.clip(y_tf_observed, 0, 1)
    max_accuracy_history.append(best_f)

    return max_accuracy_history

  def get_seeds(self):
    return self.seeds
