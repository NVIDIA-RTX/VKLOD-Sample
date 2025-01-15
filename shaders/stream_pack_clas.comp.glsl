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

layout(push_constant) uniform PackClasConstants_
{
  PackClasConstants pc;
};

void main()
{
  const uint32_t loadClusterIndex = gl_GlobalInvocationID.x;
  if(loadClusterIndex >= pc.clusterCount)
    return;

  Uint32Array    loadClusterLoadGroups = Uint32Array(pc.loadClusterLoadGroupsAddress);
  uint32_t       loadGroupIndex        = loadClusterLoadGroups.array[loadClusterIndex];
  Uint32Array    clasSizes             = Uint32Array(pc.clasSizesAddress);
  uint32_t       clasSize              = clasSizes.array[loadClusterIndex];
  MutUint32Array groupClasAllocNext    = MutUint32Array(pc.groupClasAllocNextAddress);

  // This shader is run in two passes. The first computes sizes with this
  // atomicAdd. Allocations are made for the CLAS data using those sizes. Then
  // this shader is rerun with pc.groupClasBaseAddressesAddress set to compute
  // final addresses.
  uint32_t addressOffset = atomicAdd(groupClasAllocNext.array[loadGroupIndex], clasSize);
  if(pc.groupClasBaseAddressesAddress != 0)
  {
    Uint64Array    groupClasBaseAddresses = Uint64Array(pc.groupClasBaseAddressesAddress);
    MutUint64Array packedClasAddresses    = MutUint64Array(pc.packedClasAddressesAddress);
    uint64_t       clasAddress            = groupClasBaseAddresses.array[loadGroupIndex] + addressOffset;

    // Write to the temporary destination addresses for moving the cluster
    // acceleration structures. This exists becase the API requires a contiguous
    // array.
    packedClasAddresses.array[loadClusterIndex] = clasAddress;

    // Also write to the per group CLAS addresses array that traversal reads for
    // writing BLASes.
    Uint32Array    loadGroupClusterOffsets = Uint32Array(pc.loadGroupClusterOffsetsAddress);
    uint32_t       groupClusterOffset      = loadGroupClusterOffsets.array[loadGroupIndex];
    uint32_t       clusterIndex            = loadClusterIndex - groupClusterOffset;
    LoadGroupArray loadGroups              = LoadGroupArray(pc.loadGroupsAddress);
    LoadGroup      loadGroup               = loadGroups.array[loadGroupIndex];
    MutUint64Array groupClasAddresses      = MutUint64Array(loadGroup.groupData.clasAddressesAddress);
    groupClasAddresses.array[clusterIndex] = clasAddress;
  }
}
