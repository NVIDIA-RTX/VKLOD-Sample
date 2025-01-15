/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#extension GL_EXT_control_flow_attributes : enable
#extension GL_EXT_shader_explicit_arithmetic_types : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_shader_atomic_int64 : enable
#extension GL_EXT_shader_subgroup_extended_types_int64 : enable
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_KHR_shader_subgroup_shuffle : require

#ifndef TRAVERSE_PER_INSTANCE
#error "Must define TRAVERSE_PER_INSTANCE"
#endif

#ifndef IS_RASTERIZATION
#error "Must define IS_RASTERIZATION"
#endif

#if IS_RASTERIZATION && !TRAVERSE_PER_INSTANCE
#error "Rasterization must define TRAVERSE_PER_INSTANCE 1"
#endif

#include "traverse_device_host.h"

layout(local_size_x = TRAVERSAL_WORKGROUP_SIZE) in;

layout(set = 0, binding = BTraversalConstants, scalar) uniform TraverseConstants_
{
  TraversalConstants pc;
};

#if IS_RASTERIZATION
layout(set = 0, binding = BTraversalHiZTex) uniform sampler2D texHizFar;

const float c_epsilon    = 1.2e-07f;
const float c_depthNudge = 2.0 / float(1 << 24);

// is greater than 1 pixel
bool intersectSize(vec4 clipMin, vec4 clipMax)
{
  vec2 rect          = clipMax.xy - clipMin.xy;
  vec2 clipThreshold = vec2(2.0) / pc.traversalParams.hizViewport.xy;
  return any(greaterThan(rect, clipThreshold));
  //return true;
}

vec4 getClip(vec4 hPos, out bool valid)
{
  valid = !(-c_epsilon < hPos.w && hPos.w < c_epsilon);
  return vec4(hPos.xyz / abs(hPos.w), hPos.w);
}

uint getCullBits(vec4 hPos)
{
  uint cullBits = 0;
  cullBits |= hPos.x < -hPos.w ? 1 : 0;
  cullBits |= hPos.x > hPos.w ? 2 : 0;
  cullBits |= hPos.y < -hPos.w ? 4 : 0;
  cullBits |= hPos.y > hPos.w ? 8 : 0;
  cullBits |= hPos.z < 0 ? 16 : 0;
  cullBits |= hPos.z > hPos.w ? 32 : 0;
  cullBits |= hPos.w <= 0 ? 64 : 0;
  return cullBits;
}

vec4 getBoxCorner(vec3 bboxMin, vec3 bboxMax, int n)
{
  bvec3 useMax = bvec3((n & 1) != 0, (n & 2) != 0, (n & 4) != 0);
  return vec4(mix(bboxMin, bboxMax, useMax), 1);
}

bool intersectFrustum(vec3 bboxMin, vec3 bboxMax, mat4 worldTM, out vec4 oClipmin, out vec4 oClipmax, out bool oClipvalid)
{
  mat4 worldViewProjTM = pc.traversalParams.hizViewProj * worldTM;
  bool valid;
  // clipspace bbox
  vec4 hPos      = worldViewProjTM * getBoxCorner(bboxMin, bboxMax, 0);
  vec4 clip      = getClip(hPos, valid);
  uint bits      = getCullBits(hPos);
  vec4 clipMin   = clip;
  vec4 clipMax   = clip;
  bool clipValid = valid;

  [[unroll]] for(int n = 1; n < 8; n++)
  {
    hPos = worldViewProjTM * getBoxCorner(bboxMin, bboxMax, n);
    clip = getClip(hPos, valid);
    bits &= getCullBits(hPos);

    clipMin = min(clipMin, clip);
    clipMax = max(clipMax, clip);

    clipValid = clipValid && valid;
  }

  oClipvalid = clipValid;
  oClipmin   = vec4(clamp(clipMin.xy, vec2(-1), vec2(1)), clipMin.zw);
  oClipmax   = vec4(clamp(clipMax.xy, vec2(-1), vec2(1)), clipMax.zw);

  //return true;
  return bits == 0;
}

bool intersectHiz(vec4 clipMin, vec4 clipMax)
{
  clipMin.xy = clipMin.xy * 0.5 + 0.5;
  clipMax.xy = clipMax.xy * 0.5 + 0.5;

  clipMin.xy *= pc.traversalParams.hizSizeFactors.xy;
  clipMax.xy *= pc.traversalParams.hizSizeFactors.xy;

  clipMin.xy = min(clipMin.xy, pc.traversalParams.hizSizeFactors.zw);
  clipMax.xy = min(clipMax.xy, pc.traversalParams.hizSizeFactors.zw);

  vec2  size     = (clipMax.xy - clipMin.xy);
  float maxsize  = max(size.x, size.y) * pc.traversalParams.hizSizeMax;
  float miplevel = ceil(log2(maxsize));

  float depth  = textureLod(texHizFar, ((clipMin.xy + clipMax.xy) * 0.5), miplevel).r;
  bool  result = clipMin.z <= depth + c_depthNudge;

  return result;
}

bool wasVisible(mat4 instanceTransform, vec4 boundingSphere)
{
  vec3 bboxMin = boundingSphere.xyz - boundingSphere.w;
  vec3 bboxMax = boundingSphere.xyz + boundingSphere.w;

  vec4 clipMin;
  vec4 clipMax;
  bool clipValid;

  bool useOcclusion = pc.traversalParams.useOcclusion != 0;

  bool inFrustum = intersectFrustum(bboxMin, bboxMax, instanceTransform, clipMin, clipMax, clipValid);
  bool isVisible =
      inFrustum && (!useOcclusion || !clipValid || (intersectSize(clipMin, clipMax) && intersectHiz(clipMin, clipMax)));

  return isVisible;
}
#endif

uint div_up(uint n, uint d)
{
  return (n + d - 1) / d;
}

float conservativeErrorOverDistance(mat4x3 instanceToEye, float uniformScale, vec4 boundingSphere, float objectSpaceQuadricError)
{
  float sphereDistance = length(instanceToEye * vec4(boundingSphere.xyz, 1.0f));
  float error          = objectSpaceQuadricError * uniformScale;
  float errorDistance  = max(error, sphereDistance - boundingSphere.w * uniformScale);
  return error / errorDistance;
}

bool traverseChild(mat4x3 instanceToEye, float uniformScale, vec4 boundingSphere, float quadricError)
{
  return conservativeErrorOverDistance(instanceToEye, uniformScale, boundingSphere, quadricError)
         >= pc.traversalParams.errorOverDistanceThreshold;
}

bool renderCluster(mat4x3 instanceToEye, float uniformScale, vec4 boundingSphere, float quadricError)
{
  return conservativeErrorOverDistance(instanceToEye, uniformScale, boundingSphere, quadricError)
         < pc.traversalParams.errorOverDistanceThreshold;
}

#if IS_RASTERIZATION == 0 && TRAVERSE_PER_INSTANCE == 0

// Keep traversing while at least one instance needs more detail
bool traverseChildAny(MeshInstances meshInstances, vec4 boundingSphere, float quadricError)
{
  for(uint32_t i = 0; i < TRAVERSAL_NEAREST_INSTANCE_COUNT && meshInstances.enabled[i] != 0; ++i)
  {
    float instanceStreamPriority;
    if(traverseChild(meshInstances.instanceToEye[i], meshInstances.uniformScale[i], boundingSphere, quadricError))
      return true;
  }
  return false;
}

// All instances must be above the detail threshold to render
// TODO: turn anyEnabled into an assumption - shouldn't be running traversal when no instances exist
bool renderClusterAll(MeshInstances meshInstances, vec4 boundingSphere, float quadricError)
{
  bool anyEnabled = false;
  bool result     = true;
  for(uint32_t i = 0; i < TRAVERSAL_NEAREST_INSTANCE_COUNT && meshInstances.enabled[i] != 0; ++i)
  {
    anyEnabled = true;
    result = result && renderCluster(meshInstances.instanceToEye[i], meshInstances.uniformScale[i], boundingSphere, quadricError);
  }
  return anyEnabled && result;
}

#endif

shared bool running;

// Persistent threads to traverse a LOD hierarchy and write a list of cluster
// acceleration structure handles to combine into a BLAS. Clusters are chosen
// when they exist between two LOD thresholds:
// 1. traverseChild() "Clusters in children nodes are still big enough to be
//    rendered". This ends traversal (necessary for performance) and stops
//    clusters that are too small being rendered or even tested.
// 2. renderCluster() "Cluster is small enough to be rendered". This stops
//    clusters that are too big from being rendered. Currently all clusters up
//    to the selected LOD are checked since it is unknown whether the ideal
//    detail clusters are resident from streaming.
//
// The hierarchy is traversed in breadth first order (TODO: depth first may
// reduce the memory requirements).
// - Each thread gets a nodeJobIndex and processes that node in the hierarchy if
//   the traverseChild() check passes, possibly appending more jobs to the
//   queue. When a cluster node is encountered, they are all added to a second
//   cluster queue for processing.
// - Each thread gets a clusterJobIndex and decides if that cluster should be
//   rendered with renderCluster(). If it should, it gets appended to the output
//   handle array for the BLAS build.
// - Persistent thread termination is handled when all jobs are finished. This
//   is done with a separate 'remaining' counter because the queue 'read' and
//   'write' indices only allocate readers and writers and cannot indicate
//   whether the operations up to that index are complete.
//
// TODO: this mechanism is not a ring buffer and needs memory allocated for all
// possible jobs. This may need to be improved.
void main()
{
  int canary = 1000;

  int nodeJobIndex    = -1;
  int clusterJobIndex = -1;

  MeshArray              meshes       = MeshArray(pc.meshesAddress);
  NodeQueueArray         nodeQueue    = NodeQueueArray(pc.nodeQueueAddress);
  ClusterQueueArray      clusterQueue = ClusterQueueArray(pc.clusterQueueAddress);
  JobStatusArray         jobStatus    = JobStatusArray(pc.jobStatusAddress);
  ClusterBLASInfoNVArray blasInput    = ClusterBLASInfoNVArray(pc.blasInputAddress);

#if TRAVERSE_PER_INSTANCE
  InstanceArray instances = InstanceArray(pc.instancesAddress);
#endif

  for(;;)
  {

    // Get a unique node job. The first of every 4 threads read the node job
    // and the remaining threads handle each of the 4 children. Note that
    // this can get ahead of 'write' and even overflow the queue.
    // TODO: use neater cooperative_group?
    uint64_t nodeJobEncoded  = 0;
    uint32_t nodeBatchThread = gl_SubgroupInvocationID.x % NODE_BATCH_SIZE;
    if(nodeBatchThread == 0)
    {
      if(nodeJobIndex == -1)
      {
        nodeJobIndex = atomicAdd(jobStatus.array[0].nodeQueue.read, 1);
      }
      nodeJobEncoded =
          nodeJobIndex < pc.nodeQueueSize ? atomicExchange(nodeQueue.array[nodeJobIndex], uint64_t(0u)) : uint64_t(0u);
    }
    nodeJobEncoded = subgroupShuffle(nodeJobEncoded, gl_SubgroupInvocationID.x & ~(NODE_BATCH_SIZE - 1));

    // Do the node job if there is one. This must be done in two parts so
    // that __syncthreads() can be between atomically adding to remaining
    // and writing the jobs. Otherwise it seems CUDA can reorder the atomic
    // ops and 'remaining' can dip to or below zero for other threads.
    NodeJob   nodeJob;
    Instance  instance;
    Mesh      mesh;
    NodeArray nodes;
    NodeRange parentChilren;
    uint32_t  nodeIndex;
    Node      node;
    bool      nodeValid            = false;
    bool      isClusterNode        = false;
    bool      visitNode            = false;
    int       clusterJobWriteIndex = 99999999;
    int       nodeJobWriteIndex    = 99999999;
    if(nodeJobEncoded != 0)
    {
      nodeJob = decodeNodeJob(nodeJobEncoded);
#if TRAVERSE_PER_INSTANCE
      instance = instances.array[uint32_t(nodeJob.objectIndex)];
      mesh     = meshes.array[instance.meshIndex];
#else
      mesh = meshes.array[uint32_t(nodeJob.objectIndex)];
#endif
      nodes = NodeArray(mesh.nodesAddress);

      // TODO: each job must read the parent and child. This is to fit the jobs
      // into a single uint32_t. It's probably faster to use 64bit jobs so the
      // batch range can be encoded in full.
      Node      parent         = nodes.array[uint32_t(nodeJob.parentIndex)];
      NodeRange parentChilren  = decodeChildren(parent.childrenOrClusters);
      uint32_t  nodeBatchIndex = uint32_t(nodeJob.batchIndex) * NODE_BATCH_SIZE + nodeBatchThread;
      nodeValid                = nodeBatchIndex < parentChilren.childCountMinusOne + 1;
      if(nodeValid)
      {
        nodeIndex               = parentChilren.childOffset + nodeBatchIndex;
        node                    = nodes.array[nodeIndex];
        isClusterNode           = decodeChildren(node.childrenOrClusters).clusterNode != 0;
        vec4 nodeBoundingSphere = vec4(node.bsx, node.bsy, node.bsz, node.bsw);
#if IS_RASTERIZATION
        mat4 instanceToEye = pc.traversalParams.viewTransform * instance.transform;
        visitNode          = wasVisible(instance.transform, nodeBoundingSphere)
                    && traverseChild(mat4x3(instanceToEye), instance.uniformScale, nodeBoundingSphere,
                                     node.maxClusterQuadricError);
#elif TRAVERSE_PER_INSTANCE
        mat4 instanceToEye = pc.traversalParams.viewTransform * instance.transform;
        visitNode = traverseChild(mat4x3(instanceToEye), instance.uniformScale, nodeBoundingSphere,
                                  node.maxClusterQuadricError);
#else
        MeshInstancesArray meshInstances = MeshInstancesArray(pc.meshInstances);
        visitNode = traverseChildAny(meshInstances.array[uint32_t(nodeJob.objectIndex)], nodeBoundingSphere, node.maxClusterQuadricError);
#endif

        uint32_t jobsToCreate = 0;
        if(visitNode && isClusterNode)
        {
          ClusterRange      clusters = decodeClusters(node.childrenOrClusters);
          ClusterGroupArray groups   = ClusterGroupArray(mesh.groupsAddress);
          ClusterGroup      group    = groups.array[clusters.clusterGroup];

          // Always mark the group as needed
          NeededFlagsArray groupNeededFlags = NeededFlagsArray(mesh.groupNeededFlagsAddress);
          groupNeededFlags.array[clusters.clusterGroup] |= uint8_t(STREAMING_GROUP_IS_NEEDED);  // the group is/is still needed for streaming

          // Only render the group if it has been streamed in. Group pointers
          // are null if not resident.
          // TODO: Move this info to Node::clusters as it would already be hot
          // in cache now. As it is this is likely an expensive check.
          if(group.clusterGeneratingGroupsAddress == 0)
          {
            visitNode = false;  // don't create jobs for unloaded clusters
          }
          else
          {
            uint32_t numClusters   = clusters.clusterCountMinusOne + 1;
            uint32_t newJobBatches = div_up(numClusters, CLUSTER_BATCH_SIZE);
            clusterJobWriteIndex   = atomicAdd(jobStatus.array[0].clusterQueue.write, int(newJobBatches));

            // Take care to only promise as many jobs as will be written,
            // accounting for queue overflow
            jobsToCreate += min(clusterJobWriteIndex + newJobBatches, max(clusterJobWriteIndex, pc.clusterQueueSize)) - clusterJobWriteIndex;
          }
        }
        if(visitNode && !isClusterNode)
        {
          NodeRange nodeChildren  = decodeChildren(node.childrenOrClusters);
          uint32_t  numChildren   = nodeChildren.childCountMinusOne + 1;
          uint32_t  newJobBatches = div_up(numChildren, NODE_BATCH_SIZE);
          nodeJobWriteIndex       = atomicAdd(jobStatus.array[0].nodeQueue.write, int(newJobBatches));

          // Take care to only promise as many jobs as will be written,
          // accounting for queue overflow
          jobsToCreate += min(nodeJobWriteIndex + newJobBatches, max(nodeJobWriteIndex, pc.nodeQueueSize)) - nodeJobWriteIndex;
        }

        atomicAdd(jobStatus.array[0].remaining, int(jobsToCreate));
      }
    }

    // Make sure all threads in the block finish at the same time via shared
    // 'running' variable.
    if(gl_LocalInvocationID.x == 0)
    {
      // atomicAdd() above may have returned this value
      running = canary-- > 0 && jobStatus.array[0].remaining > 0;
    }

    // Sync with the intention of flushing remaining.fetch_add() before
    // writing jobs to either queue. Combined with if(!running), since that
    // also needs a sync after writing to 'running' to avoid hanging if only
    // some threads in the block exit.
    barrier();
    if(!running)
      break;

    if(nodeJobEncoded != 0 && nodeValid)
    {
      if(visitNode && isClusterNode)
      {
        // Schedule jobs for every cluster batch
        ClusterRange clusters      = decodeClusters(node.childrenOrClusters);
        uint32_t     numClusters   = clusters.clusterCountMinusOne + 1;
        uint32_t     newJobBatches = div_up(numClusters, CLUSTER_BATCH_SIZE);
        for(uint32_t i = 0; i < newJobBatches; ++i)
        {
          uint64_t clusterJobEncoded = encodeClusterJob(nodeJob.objectIndex, nodeIndex, i);

          // TODO: atomicStore!?
          // TODO: remove overflow protection
          if(clusterJobWriteIndex + i < pc.clusterQueueSize)
            atomicExchange(clusterQueue.array[clusterJobWriteIndex + i], clusterJobEncoded);
        }
      }
      if(visitNode && !isClusterNode)
      {
        // Create a new job for all children
        NodeRange nodeChildren  = decodeChildren(node.childrenOrClusters);
        uint32_t  numChildren   = nodeChildren.childCountMinusOne + 1;
        uint32_t  newJobBatches = div_up(numChildren, NODE_BATCH_SIZE);
        for(uint32_t i = 0; i < newJobBatches; ++i)
        {
          uint64_t nodeJobEncoded = encodeNodeJob(nodeJob.objectIndex, nodeIndex, i);

          // TODO: atomicStore!?
          // TODO: remove overflow protection
          if(nodeJobWriteIndex + i < pc.nodeQueueSize)
            atomicExchange(nodeQueue.array[nodeJobWriteIndex + i], nodeJobEncoded);
        }
      }

      // Node job done. Thread 0 subtracts 1 for the whole batch
      if(nodeBatchThread == 0)
      {
        atomicAdd(jobStatus.array[0].remaining, -1);
        nodeJobIndex = -1;
      }
    }

    // Get a unique cluster job. The first of every 8 threads read the
    // cluster job and the remaining threads process the batch (depending on
    // the batch size). Note that this can get ahead of 'write' and even
    // overflow the queue.
    uint64_t clusterJobEncoded  = 0;
    uint32_t clusterBatchThread = gl_SubgroupInvocationID.x % CLUSTER_BATCH_SIZE;
    if(clusterBatchThread == 0)
    {
      if(clusterJobIndex == -1)
      {
        clusterJobIndex = atomicAdd(jobStatus.array[0].clusterQueue.read, 1);
      }

      // Wait for the job to be written and swap it back out with a zero.
      clusterJobEncoded = clusterJobIndex < pc.clusterQueueSize ?
                              atomicExchange(clusterQueue.array[clusterJobIndex], uint64_t(0u)) :
                              uint64_t(0u);
    }
    clusterJobEncoded = subgroupShuffle(clusterJobEncoded, gl_SubgroupInvocationID.x & ~(CLUSTER_BATCH_SIZE - 1));

    // Do the cluster job if there is one.
    if(clusterJobEncoded != 0)
    {
      ClusterJob clusterJob = decodeClusterJob(clusterJobEncoded);
#if TRAVERSE_PER_INSTANCE
      Instance instance = instances.array[uint32_t(clusterJob.objectIndex)];
      Mesh     mesh     = meshes.array[instance.meshIndex];
#else
      Mesh mesh = meshes.array[uint32_t(clusterJob.objectIndex)];
#endif
      NodeArray    nodes             = NodeArray(mesh.nodesAddress);
      Node         parent            = nodes.array[uint32_t(clusterJob.parentIndex)];
      ClusterRange clusterRange      = decodeClusters(parent.childrenOrClusters);
      uint32_t     clusterBatchIndex = uint32_t(clusterJob.batchIndex) * CLUSTER_BATCH_SIZE + clusterBatchThread;
      if(clusterBatchIndex < clusterRange.clusterCountMinusOne + 1)
      {
        ClusterGroupArray groups                 = ClusterGroupArray(mesh.groupsAddress);
        ClusterGroup      group                  = groups.array[clusterRange.clusterGroup];
        uint32_t          clusterIndex           = clusterBatchIndex;  // local to the group
        UintArray         clusterGenratingGroups = UintArray(group.clusterGeneratingGroupsAddress);
        uint32_t          generatingGroupIndex   = clusterGenratingGroups.array[clusterIndex];
        bool              appendClas             = false;

        // If the generating group is not yet resident we must draw the cluster.
        // I.e. this is the fallback even if it's not the ideal cluster.
        // TODO: Move this info to Node::clusters as it would already be hot
        // in cache now. As it is this is likely an expensive check.
        if(generatingGroupIndex == 0xffffffffu || groups.array[generatingGroupIndex].clusterGeneratingGroupsAddress == 0)
        {
          appendClas = true;
        }
        else
        {
          FloatArray groupQuadricErrors            = FloatArray(mesh.groupQuadricErrorsAddress);
          Vec4Array  groupBoundingSpheres          = Vec4Array(mesh.groupBoundingSpheresAddress);
          float      generatingGroupQuadricError   = groupQuadricErrors.array[generatingGroupIndex];
          vec4       generatingGroupBoundingSphere = groupBoundingSpheres.array[generatingGroupIndex];

#if IS_RASTERIZATION
          mat4 instanceToEye = pc.traversalParams.viewTransform * instance.transform;
          appendClas         = wasVisible(instance.transform, generatingGroupBoundingSphere)
                       && renderCluster(mat4x3(instanceToEye), instance.uniformScale,
                                        generatingGroupBoundingSphere, generatingGroupQuadricError);
#elif TRAVERSE_PER_INSTANCE
          mat4 instanceToEye = pc.traversalParams.viewTransform * instance.transform;
          appendClas         = renderCluster(mat4x3(instanceToEye), instance.uniformScale,
                                             generatingGroupBoundingSphere, generatingGroupQuadricError);
#else
          MeshInstancesArray meshInstances = MeshInstancesArray(pc.meshInstances);
          appendClas                       = renderClusterAll(meshInstances.array[uint32_t(clusterJob.objectIndex)],
                                                              generatingGroupBoundingSphere, generatingGroupQuadricError);
#endif
        }

        // Append the cluster's cluster acceleration structure to the bottom
        // level acceleration structure build input list.
        if(appendClas)
        {
#if IS_RASTERIZATION
          DrawMeshTasksIndirectRef drawMeshTasksIndirect = DrawMeshTasksIndirectRef(pc.drawMeshTasksIndirectAddress);
          DrawClusterArray         drawClusters          = DrawClusterArray(pc.drawClustersAddress);

          uint clusterWriteIndex = atomicAdd(drawMeshTasksIndirect.d.taskCount, 1);
          if(clusterWriteIndex < pc.drawClustersSize)
          {
            DrawCluster dc;
            dc.instanceIndex = uint32_t(clusterJob.objectIndex);
            dc.meshIndex     = instance.meshIndex;
            dc.cluster       = group.clusterGeometryAddressesAddress + sizeof(ClusterGeometryRef) * clusterIndex;
            drawClusters.array[clusterWriteIndex] = dc;
          }
#else
          Uint64Array            clusterAccelerationStructures = Uint64Array(group.clasAddressesAddress);
          ClusterBLASInfoNVArray blasInput                     = ClusterBLASInfoNVArray(pc.blasInputAddress);
          MutUint64Array blasInputClusters = MutUint64Array(blasInput.array[uint32_t(clusterJob.objectIndex)].clusterReferences);
          uint blasWriteIndex = atomicAdd(blasInput.array[uint32_t(clusterJob.objectIndex)].clusterReferencesCount, 1);
          blasInputClusters.array[blasWriteIndex] = clusterAccelerationStructures.array[clusterIndex];
#endif
        }
      }

      // Cluster job done. Thread 0 subtracts 1 for the whole batch
      if(clusterBatchThread == 0)
      {
        atomicAdd(jobStatus.array[0].remaining, -1);
        clusterJobIndex = -1;
      }
    }
  }
}
