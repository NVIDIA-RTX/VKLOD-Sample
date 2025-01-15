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
#include "traverse_device_host.h"

layout(local_size_x = TRAVERSAL_WORKGROUP_SIZE) in;

layout(push_constant, scalar) uniform WriteInstancesConstant_
{
  WriteInstancesConstant pc;
};

uint div_up(uint n, uint d)
{
  return (n + d - 1) / d;
}

void main()
{
  const uint32_t instanceId = gl_GlobalInvocationID.x;
  if(instanceId >= pc.instancesSize)
    return;

  // Write the instance's BLAS address into its TLAS info struct.
  // This is executed after per-mesh traversal.
  InstanceArray     instances                                    = InstanceArray(pc.instances);
  Instance          instance                                     = instances.array[instanceId];
  Uint64Array       meshBlasAddresses                            = Uint64Array(pc.meshBlasAddresses);
  uint64_t          meshBlasAddress                              = meshBlasAddresses.array[instance.meshIndex];
  InstanceInfoArray instanceInfos                                = InstanceInfoArray(pc.tlasInfos);
  instanceInfos.array[instanceId].accelerationStructureReference = meshBlasAddress;
}
