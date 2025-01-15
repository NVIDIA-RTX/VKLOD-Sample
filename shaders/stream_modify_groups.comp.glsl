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

layout(push_constant) uniform StreamGroupsConstants_
{
  StreamGroupsConstants pc;
};

void main()
{
  const uint32_t itemIndex = gl_GlobalInvocationID.x;
  MutMeshArray   meshes    = MutMeshArray(pc.meshesAddress);
  if(pc.load != 0)
  {
    if(itemIndex >= pc.mods.loadGroupCount)
      return;

    // Copy pointers to the loaded cluster group to the mesh.groups array
    LoadGroupArray       loadGroups    = LoadGroupArray(pc.mods.loadGroupsAddress);
    LoadGroup            loadGroup     = loadGroups.array[itemIndex];
    Mesh                 mesh          = meshes.array[loadGroup.meshIndex];
    MutClusterGroupArray groups        = MutClusterGroupArray(mesh.groupsAddress);
    groups.array[loadGroup.groupIndex] = loadGroup.groupData;
    atomicAdd(meshes.array[loadGroup.meshIndex].residentClusterCount, loadGroup.groupData.clusterCount);
  }
  else
  {
    if(itemIndex >= pc.mods.unloadGroupCount)
      return;

    // Reset pointers to the unloaded cluster group in the mesh.groups array to null
    UnloadGroupArray     unloadGroups = UnloadGroupArray(pc.mods.unloadGroupsAddress);
    UnloadGroup          unloadGroup  = unloadGroups.array[itemIndex];
    Mesh                 mesh         = meshes.array[unloadGroup.meshIndex];
    MutClusterGroupArray groups       = MutClusterGroupArray(mesh.groupsAddress);
    atomicAdd(meshes.array[unloadGroup.meshIndex].residentClusterCount, -int(groups.array[unloadGroup.groupIndex].clusterCount));
    groups.array[unloadGroup.groupIndex].clusterGeometryAddressesAddress = 0;
    groups.array[unloadGroup.groupIndex].clusterGeneratingGroupsAddress  = 0;
    groups.array[unloadGroup.groupIndex].clasAddressesAddress            = 0;
    groups.array[unloadGroup.groupIndex].clusterCount                    = 0;
  }
}
