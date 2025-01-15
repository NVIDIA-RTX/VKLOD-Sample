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
#extension GL_EXT_debug_printf : require

#include "traverse_device_host.h"

#ifndef TRAVERSE_PER_INSTANCE
#error "Must define TRAVERSE_PER_INSTANCE"
#endif

#if IS_RASTERIZATION
layout(local_size_x = 1) in;
#else
layout(local_size_x = TRAVERSAL_WORKGROUP_SIZE) in;
#endif

layout(set = 0, binding = BTraversalConstants, scalar) uniform TraverseConstants_
{
  TraversalConstants pc;
};

void main()
{
#if IS_RASTERIZATION

  if(gl_GlobalInvocationID.x != 0)
    return;

  DrawMeshTasksIndirectRef drawMeshTasksIndirect = DrawMeshTasksIndirectRef(pc.drawMeshTasksIndirectAddress);
  DrawStatsRef             drawStats             = DrawStatsRef(pc.drawStatsAddress);

  uint requestedClusterCount = drawMeshTasksIndirect.d.taskCount;

  // we might overshoot the counter during traversal, but we don't record more than buffer is sized for
  drawMeshTasksIndirect.d.taskCount = min(requestedClusterCount, pc.drawClustersSize);

  // for statistics, keep what was requested
  drawStats.d.requestedClusterCount = requestedClusterCount;

#else
#if TRAVERSE_PER_INSTANCE
  const uint32_t instanceId = gl_GlobalInvocationID.x;
  if(instanceId >= pc.itemsSize)
    return;
  const uint32_t itemId    = instanceId;
  InstanceArray  instances = InstanceArray(pc.instancesAddress);
  Instance       instance  = instances.array[instanceId];
  MeshArray      meshes    = MeshArray(pc.meshesAddress);
  Mesh           mesh      = meshes.array[instance.meshIndex];
#else
  const uint32_t meshId = gl_GlobalInvocationID.x;
  if(meshId >= pc.itemsSize)
    return;
  const uint32_t itemId = meshId;
  MeshArray      meshes = MeshArray(pc.meshesAddress);
  Mesh           mesh   = meshes.array[meshId];
#endif

  JobStatusArray         jobStatus = JobStatusArray(pc.jobStatusAddress);
  ClusterBLASInfoNVArray blasInput = ClusterBLASInfoNVArray(pc.blasInputAddress);

  // Replace with lowest detail clusters if the traversal canary died or no
  // clusters were produced. Neither should typically happen.
  if(jobStatus.array[0].remaining != 0 || blasInput.array[itemId].clusterReferencesCount == 0)
  {
    // Get the mesh's last group's cluster acceleration structures
    ClusterGroupArray groups                                         = ClusterGroupArray(mesh.groupsAddress);
    ClusterGroup      lastGroup                                      = groups.array[mesh.groupCount - 1];
    Uint64Array       lastGroupClusterAccelerationStructureAddresses = Uint64Array(lastGroup.clasAddressesAddress);

    // Write the last group's cluster as the only BLAS input. This should be the
    // lowest detail LOD.
    MutUint64Array blasInputClusters               = MutUint64Array(blasInput.array[itemId].clusterReferences);
    blasInput.array[itemId].clusterReferencesCount = 1;
    blasInputClusters.array[0]                     = lastGroupClusterAccelerationStructureAddresses.array[0];
  }
  else if(blasInput.array[itemId].clusterReferencesCount > mesh.residentClusterCount)
  {
    // Overflow protection - shouldn't happen as traverse_init.comp.glsl allocates this
    blasInput.array[itemId].clusterReferencesCount = mesh.residentClusterCount;

    // DEBUGGING: Make it more visible by disappearing geometry
    blasInput.array[itemId].clusterReferencesCount = 1;
  }

#if 0
  JobStatusArray jobStatus = JobStatusArray(pc.jobStatusAddress);
  debugPrintfEXT("item %i: nodeQueue %i clusterQueue %i\n", itemId, jobStatus.array[0].nodeQueue.write,
                 jobStatus.array[0].clusterQueue.write);
#endif
#endif
}
