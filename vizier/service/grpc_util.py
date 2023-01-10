# Copyright 2023 Google LLC.
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

"""Utilities for handling GRPC API."""

from typing import Optional
import grpc
from vizier.service import custom_errors


# TODO: Use this for all service errors.
def exception_into_context(
    e: Exception, context: Optional[grpc.ServicerContext] = None
) -> None:
  """Converts custom exception into correct context error code."""
  if not context:
    return

  if isinstance(e, custom_errors.ImmutableStudyError) or isinstance(
      e, custom_errors.ImmutableTrialError
  ):
    context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
  elif isinstance(e, custom_errors.NotFoundError):
    context.set_code(grpc.StatusCode.NOT_FOUND)
  elif isinstance(e, custom_errors.AlreadyExistsError):
    context.set_code(grpc.StatusCode.ALREADY_EXISTS)
  else:
    context.set_code(grpc.StatusCode.UNKNOWN)

  context.set_details(str(e))
