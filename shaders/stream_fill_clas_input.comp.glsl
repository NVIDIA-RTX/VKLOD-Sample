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

layout(push_constant) uniform FillClasInputConstants_
{
  FillClasInputConstants pc;
};

void main()
{
  const uint32_t loadClusterIndex = gl_GlobalInvocationID.x;
  if(loadClusterIndex >= pc.clusterCount)
    return;

  Uint32Array          loadClusterLoadGroups   = Uint32Array(pc.loadClusterLoadGroupsAddress);
  uint32_t             loadGroupIndex          = loadClusterLoadGroups.array[loadClusterIndex];
  Uint32Array          loadGroupClusterOffsets = Uint32Array(pc.loadGroupClusterOffsetsAddress);
  uint32_t             groupClusterOffset      = loadGroupClusterOffsets.array[loadGroupIndex];
  uint32_t             clusterIndex            = loadClusterIndex - groupClusterOffset;
  LoadGroupArray       loadGroups              = LoadGroupArray(pc.loadGroupsAddress);
  LoadGroup            loadGroup               = loadGroups.array[loadGroupIndex];
  ClusterGeometryArray clusterGeometries = ClusterGeometryArray(loadGroup.groupData.clusterGeometryAddressesAddress);
  ClusterGeometry      clusterGeometry   = clusterGeometries.array[clusterIndex];
  MutCLASInputArray    clasInput         = MutCLASInputArray(pc.clasInfoAddress);

  // Fill VkClusterAccelerationStructureBuildTriangleClusterInfoNV for each
  // cluster in the streamed batch
  clasInput.array[loadClusterIndex].clusterID =
      (loadGroup.groupIndex << CLUSTER_ID_GROUP_SHIFT) | (clusterIndex & CLUSTER_ID_CLUSTER_MASK);
  clasInput.array[loadClusterIndex].clusterFlags = 0;

  uint32_t bits0 = 0;
  bits0          = bitfieldInsert(bits0, clusterGeometry.triangleCount, 0, 9);  // .triangleCount
  bits0          = bitfieldInsert(bits0, clusterGeometry.vertexCount, 9, 9);    // .vertexCount
  bits0          = bitfieldInsert(bits0, pc.positionTruncateBits, 18, 6);       // .positionTruncateBitCount
  bits0          = bitfieldInsert(bits0, VK_CLUSTER_ACCELERATION_STRUCTURE_INDEX_FORMAT_8BIT_NV, 24, 4);  // .indexType
  bits0          = bitfieldInsert(bits0, 0, 28, 4);  // .opacityMicromapIndexType
  clasInput.array[loadClusterIndex].bits0 = bits0;

  uint32_t bits1 = 0;
  bits1          = bitfieldInsert(bits1, loadGroup.meshIndex, 0, 24);                              // .baseGeometryIndex
  bits1          = bitfieldInsert(bits1, 0, 24, 5);                                                // .reserved
  bits1 = bitfieldInsert(bits1, VK_CLUSTER_ACCELERATION_STRUCTURE_GEOMETRY_OPAQUE_BIT_NV, 29, 3);  // .baseGeometryFlags
  clasInput.array[loadClusterIndex].bits1 = bits1;

  clasInput.array[loadClusterIndex].indexBufferStride = uint16_t(1) /* sizeof(mesh.clusteredMesh.meshTriIndices.data()->x) */;
  clasInput.array[loadClusterIndex].vertexBufferStride = uint16_t(4 * 3) /* sizeof(*mesh.clusteredMesh.meshPositions.data()) */;
  clasInput.array[loadClusterIndex].geometryIndexAndFlagsBufferStride = uint16_t(0);
  clasInput.array[loadClusterIndex].opacityMicromapIndexBufferStride  = uint16_t(0);

  clasInput.array[loadClusterIndex].indexBuffer                 = clusterGeometry.triangleVerticesAddress;
  clasInput.array[loadClusterIndex].vertexBuffer                = clusterGeometry.vertexPositionsAddress;
  clasInput.array[loadClusterIndex].geometryIndexAndFlagsBuffer = 0;
  clasInput.array[loadClusterIndex].opacityMicromapArray        = 0;
  clasInput.array[loadClusterIndex].opacityMicromapIndexBuffer  = 0;
}
