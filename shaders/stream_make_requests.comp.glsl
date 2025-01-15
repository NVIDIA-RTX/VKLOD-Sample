/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require

#include "shaders_scene.h"
#include "shaders_stream.h"

layout(local_size_x = STREAM_WORKGROUP_SIZE) in;

layout(push_constant) uniform BuildRequestsConstants_
{
  BuildRequestsConstants pc;
};

void main()
{
  const uint32_t groupId = gl_GlobalInvocationID.x;
  if(groupId >= pc.groupCount)
    return;

  NeededFlagsArray groupNeededFlags = NeededFlagsArray(pc.groupNeededFlagsAddress);

  bool isRoot    = (groupNeededFlags.array[groupId] & uint8_t(STREAMING_GROUP_IS_ROOT)) != 0;
  bool wasNeeded = (groupNeededFlags.array[groupId] & uint8_t(STREAMING_GROUP_WAS_NEEDED)) != 0;
  bool isNeeded  = (groupNeededFlags.array[groupId] & uint8_t(STREAMING_GROUP_IS_NEEDED)) != 0;

  // Force the last group to always be loaded. This contains the lowest detail
  // cluster so something can always be rendered.
  if(isRoot)
    isNeeded = true;

  // Shift isNeeded to wasNeeded, zero the next isNeeded and keep isRoot
  groupNeededFlags.array[groupId] =
      uint8_t(isNeeded ? STREAMING_GROUP_WAS_NEEDED : 0) | uint8_t(isRoot ? STREAMING_GROUP_IS_ROOT : 0);

  // Emit rising and falling edges as streaming requests for the group
  if(wasNeeded != isNeeded)
  {
    MutStreamRequestCountsArray requestCounts = MutStreamRequestCountsArray(pc.streamRequestCountsAddress);
    uint32_t                    request       = atomicAdd(requestCounts.array[0].requestsCount, 1u);
    if(request < requestCounts.array[0].requestsSize)
    {
      MutGroupRequestArray requests              = MutGroupRequestArray(pc.requestsAddress);
      requests.array[request].globalGroupAndLoad = (groupId & 0x7fffffff) | ((isNeeded ? 1u : 0u) << 31);
    }
    else
    {
      // There wasn't enough room in the request list. Restore the wasNeeded flag
      // so that the load request will still be emitted next frame if it's still
      // needed
      groupNeededFlags.array[groupId] =
          uint8_t(wasNeeded ? STREAMING_GROUP_WAS_NEEDED : 0) | uint8_t(isRoot ? STREAMING_GROUP_IS_ROOT : 0);
    }
  }
}
