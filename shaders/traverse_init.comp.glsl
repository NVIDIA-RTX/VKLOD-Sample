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

#ifndef TRAVERSE_PER_INSTANCE
#error "Must define TRAVERSE_PER_INSTANCE"
#endif

#include "shaders_scene.h"
#include "traverse_device_host.h"

layout(local_size_x = TRAVERSAL_WORKGROUP_SIZE) in;

layout(set = 0, binding = BTraversalConstants, scalar) uniform TraverseConstants_
{
  TraversalConstants pc;
};

uint div_up(uint n, uint d)
{
  return (n + d - 1) / d;
}

void main()
{
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

  // Get the root node for each mesh
  uint32_t  rootIndex   = 0;
  NodeArray nodes       = NodeArray(mesh.nodesAddress);
  Node      root        = nodes.array[rootIndex];
  NodeRange rootChilren = decodeChildren(root.childrenOrClusters);

  // Create jobs for all children of the root node
  JobStatusArray jobStatus     = JobStatusArray(pc.jobStatusAddress);
  NodeQueueArray nodeQueue     = NodeQueueArray(pc.nodeQueueAddress);
  uint32_t       numChildren   = rootChilren.childCountMinusOne + 1;
  uint32_t       newJobBatches = div_up(numChildren, NODE_BATCH_SIZE);
  atomicAdd(jobStatus.array[0].remaining, int(newJobBatches));
  int writeIndex = atomicAdd(jobStatus.array[0].nodeQueue.write, int(newJobBatches));
  for(uint32_t batchIndex = 0; batchIndex < newJobBatches; ++batchIndex)
    nodeQueue.array[writeIndex + batchIndex] = encodeNodeJob(itemId, rootIndex, batchIndex);

#if IS_RASTERIZATION
#else
  // zero the output cluster count for each instance
  ClusterBLASInfoNVArray blasInput = ClusterBLASInfoNVArray(pc.blasInputAddress);

  uint64_t instanceClustersOffset = atomicAdd(jobStatus.array[0].blasInputClustersAlloc, int(mesh.residentClusterCount));
  blasInput.array[itemId].clusterReferencesCount  = 0;
  blasInput.array[itemId].clusterReferencesStride = 8 /* sizeof(VkDeviceAddress) */;
  blasInput.array[itemId].clusterReferences = pc.blasInputClustersAddress + instanceClustersOffset * 8 /* sizeof(VkDeviceAddress) */;

  // Copy the data needed for traversal for the nearest few instances after
  // sorting
#if !TRAVERSE_PER_INSTANCE
  SortingMeshInstancesArray sortingMeshInstances = SortingMeshInstancesArray(pc.sortingMeshInstances);
  MeshInstancesArray        meshInstances        = MeshInstancesArray(pc.meshInstances);
  for(int i = 0; i < TRAVERSAL_NEAREST_INSTANCE_COUNT; ++i)
  {
    meshInstances.array[meshId].enabled[i] = uint8_t(0);
    uint32_t instanceIndex = uint32_t(sortingMeshInstances.array[meshId].nearest[i] & 0xffffffff);  // index is stored in the lower 4 bytes
    if(instanceIndex != 0xffffffff)
    {
      InstanceArray instances                      = InstanceArray(pc.instancesAddress);
      Instance      instance                       = instances.array[instanceIndex];
      mat4          instanceToEye                  = pc.traversalParams.viewTransform * instance.transform;
      meshInstances.array[meshId].instanceToEye[i] = mat4x3(instanceToEye);
      meshInstances.array[meshId].uniformScale[i]  = instance.uniformScale;
      meshInstances.array[meshId].enabled[i]       = uint8_t(1);
    }
  }
#endif
#endif
}
