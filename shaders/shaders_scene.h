/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef SHADERS_SCENE_H
#define SHADERS_SCENE_H

#include "shaders_buffer_ref.h"
#include "shaders_glm.h"
#include "shaders_lod_structs.h"

#ifdef __cplusplus
namespace shaders {
#endif  // __cplusplus

// Due to streaming indirection, the cluster ID encodes the group and a cluster
// index relative to the group. This saves e.g. binary searching to find the
// group from a mesh-relative ID.
#define CLUSTER_ID_GROUP_SHIFT 8u
#define CLUSTER_ID_CLUSTER_MASK 0xffu

#define STREAMING_GROUP_IS_NEEDED 0x1u
#define STREAMING_GROUP_WAS_NEEDED 0x2u
#define STREAMING_GROUP_IS_ROOT 0x4u

#define VISUALIZE_NONE 0
#define VISUALIZE_TRIANGLE_COLORS 1
#define VISUALIZE_CLUSTER_COLORS 2
#define VISUALIZE_GENERATING_GROUP_COLORS 3
#define VISUALIZE_MESH_COLORS 4
#define VISUALIZE_CLUSTER_LOD 5
#define VISUALIZE_TRIANGLE_AREA 6
#define VISUALIZE_TARGET_PIXEL_ERROR 7

// Definition for the UI
#define VISUALIZE_ENUM_NAMES \
  {"None", "Triangle Colors", "Cluster Colors", "Generating Group Colors", "Mesh Colors", "Cluster LOD", "Triangle Area", "Target Pixel Error"}

struct Instance
{
  mat4     transform;
  uint32_t meshIndex;
  float    uniformScale;
};

struct Material
{
  vec4   albedo;
  int8_t albedoTexture;
  int8_t padding1;
  int8_t padding2;
  int8_t padding3;
  float  roughness;
  float  metallic;
};

struct ClusterGeometry
{
  uint32_t triangleCount;
  uint32_t vertexCount;
  DEVICE_ADDRESS(u8vec3) triangleVerticesAddress;
  DEVICE_ADDRESS(vec3) vertexPositionsAddress;
  DEVICE_ADDRESS(vec3) vertexNormalsAddress;
  DEVICE_ADDRESS(vec2) vertexTexcoordsAddress;
};

// Cluster geometry indirection for streaming at cluster group granularity
struct ClusterGroup
{
  DEVICE_ADDRESS(ClusterGeometry) clusterGeometryAddressesAddress;
  DEVICE_ADDRESS(uint32_t) clusterGeneratingGroupsAddress;
  DEVICE_ADDRESS(uint64_t) clasAddressesAddress;
  uint32_t clusterCount;
  uint32_t padding_;
};

// More traversal related, but useful to visualize in pathtrace.rchit
struct Mesh
{
  DEVICE_ADDRESS(Node) nodesAddress;
  DEVICE_ADDRESS(ClusterGroup) groupsAddress;
  DEVICE_ADDRESS(float) groupQuadricErrorsAddress;
  DEVICE_ADDRESS(vec4) groupBoundingSpheresAddress;  // { float x, y, z; float radius; }
  DEVICE_ADDRESS(uint8_t) groupNeededFlagsAddress;
  DEVICE_ADDRESS(uint8_t) groupLodLevelsAddress;
  Material material;  // could be an index instead
  uint32_t groupCount;
  uint32_t residentClusterCount;
};

struct DrawCluster
{
  uint32_t instanceIndex;
  uint32_t meshIndex;

  DEVICE_ADDRESS(ClusterGeometry) cluster;
};

struct DrawMeshTasksIndirect
{
  uint32_t taskCount;
  uint32_t firstTask;
};

struct DrawStats
{
  uint32_t requestedClusterCount;
  uint32_t triangleCount;

  //uint32_t debug[128];
};

#ifndef __cplusplus
DECL_BUFFER_REF(InstanceArray, Instance);
DECL_BUFFER_REF(ClusterGeometryArray, ClusterGeometry);
DECL_BUFFER_REF(ClusterGroupArray, ClusterGroup);
DECL_MUTABLE_BUFFER_REF(MutClusterGroupArray, ClusterGroup);
DECL_BUFFER_REF(MeshArray, Mesh);
DECL_MUTABLE_BUFFER_REF(MutMeshArray, Mesh);
DECL_BUFFER_REF_BASE(NeededFlagsArray, , uint8_t, 1)
DECL_BUFFER_REF_SINGLE_BASE(ClusterGeometryRef, readonly, ClusterGeometry, 8)
DECL_BUFFER_REF_SINGLE_BASE(DrawStatsRef, , DrawStats, 4)
DECL_BUFFER_REF_SINGLE_BASE(DrawMeshTasksIndirectRef, , DrawMeshTasksIndirect, 8)
#endif

#ifdef __cplusplus
}  // namespace shaders
#endif

#endif  // SHADERS_SCENE_H
