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

#ifndef SHADERS_STREAM_H
#define SHADERS_STREAM_H

#include "shaders_lod_structs.h"
#include "shaders_scene.h"

#define STREAM_WORKGROUP_SIZE 256

#ifdef __cplusplus
namespace shaders {
#endif  // __cplusplus

#ifndef __cplusplus
#define VK_CLUSTER_ACCELERATION_STRUCTURE_GEOMETRY_OPAQUE_BIT_NV 0x00000004
#define VK_CLUSTER_ACCELERATION_STRUCTURE_INDEX_FORMAT_8BIT_NV 1
#endif

struct GroupRequest
{
#ifdef __cplusplus
  struct
  {
    uint32_t globalGroup : 31;  // linearized group index among all scene meshes
    uint32_t load : 1;
  } decoded;
#else
  uint32_t globalGroupAndLoad;
#endif  // __cplusplus
};

struct StreamRequestCounts
{
  uint32_t requestsCount;
  uint32_t requestsSize;
};

struct BuildRequestsConstants
{
  DEVICE_ADDRESS(uint8_t) groupNeededFlagsAddress;
  DEVICE_ADDRESS(StreamRequestCounts) streamRequestCountsAddress;  // indirection to write requestsCount
  DEVICE_ADDRESS(GroupRequest) requestsAddress;
  uint32_t groupCount;
};

struct LoadGroup
{
  ClusterGroup groupData;
  uint32_t     meshIndex;
  uint32_t     groupIndex;
};

struct UnloadGroup
{
  uint32_t meshIndex;
  uint32_t groupIndex;
};

struct StreamGroupModsList
{
  DEVICE_ADDRESS(LoadGroup) loadGroupsAddress;
  DEVICE_ADDRESS(UnloadGroup) unloadGroupsAddress;
  uint32_t loadGroupCount;
  uint32_t unloadGroupCount;
};

struct StreamGroupsConstants
{
  DEVICE_ADDRESS(Mesh) meshesAddress;
  StreamGroupModsList mods;
  uint32_t            load;  // shader runs twice for loads and unloads
};

struct FillClasInputConstants
{
  DEVICE_ADDRESS(LoadGroup) loadGroupsAddress;
  DEVICE_ADDRESS(ClusterCLASInfoNV) clasInfoAddress;
  DEVICE_ADDRESS(uint32_t) loadClusterLoadGroupsAddress;
  DEVICE_ADDRESS(uint32_t) loadGroupClusterOffsetsAddress;
  uint32_t clusterCount;
  uint32_t positionTruncateBits;
};

struct PackClasConstants
{
  DEVICE_ADDRESS(LoadGroup) loadGroupsAddress;
  DEVICE_ADDRESS(uint32_t) loadClusterLoadGroupsAddress;
  DEVICE_ADDRESS(uint32_t) loadGroupClusterOffsetsAddress;
  DEVICE_ADDRESS(uint32_t) clasSizesAddress;
  DEVICE_ADDRESS(uint32_t) groupClasAllocNextAddress;
  DEVICE_ADDRESS(uint64_t) packedClasAddressesAddress;
  DEVICE_ADDRESS(uint64_t) groupClasBaseAddressesAddress;
  uint32_t clusterCount;
};

#ifndef __cplusplus
DECL_MUTABLE_BUFFER_REF(MutGroupRequestArray, GroupRequest);
DECL_MUTABLE_BUFFER_REF(MutStreamRequestCountsArray, StreamRequestCounts);
DECL_MUTABLE_BUFFER_REF(MutCLASInputArray, ClusterCLASInfoNV);
DECL_BUFFER_REF(LoadGroupArray, LoadGroup);
DECL_BUFFER_REF(UnloadGroupArray, UnloadGroup);
#endif

#ifdef __cplusplus
}  // namespace shaders
#endif

#endif  // SHADERS_STREAM_H