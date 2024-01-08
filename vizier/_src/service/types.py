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

"""Convenient types library for the /service/ folder."""
from typing import Union

from vizier._src.service import pythia_service_pb2_grpc
from vizier._src.service import vizier_service_pb2_grpc

# PythiaService and VizierService aliases allow us to use either a gRPC stub or
# the actual gRPC servicer implementation (for local setup) interchangeably.
PythiaService = Union[
    pythia_service_pb2_grpc.PythiaServiceStub,
    pythia_service_pb2_grpc.PythiaServiceServicer,
]

VizierService = Union[
    vizier_service_pb2_grpc.VizierServiceStub,
    vizier_service_pb2_grpc.VizierServiceServicer,
]
