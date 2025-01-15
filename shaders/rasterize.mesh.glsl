/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_buffer_reference2 : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_shader_atomic_int64 : enable

#extension GL_NV_mesh_shader : enable

////////////////////////////////////////////

#ifndef CLUSTER_VERTEX_COUNT
#define CLUSTER_VERTEX_COUNT 64
#endif

#ifndef CLUSTER_TRIANGLE_COUNT
#define CLUSTER_TRIANGLE_COUNT 64
#endif

////////////////////////////////////////////

#include "rasterize_device_host.h"

////////////////////////////////////////////

layout(set = 0, binding = BRasterFrameInfo) uniform FrameParams_ { FrameParams frameInfo; };
layout(set = 0, binding = BRasterConstants) uniform RasterParams_ { RasterizeConstants rc; };

////////////////////////////////////////////

// keep outputs as small as possible 
layout(location = 0) out Interpolants1
{
  flat uint drawIndex;
}
OUT[];

layout(location = 1) out Interpolants2
{
  flat uint vertexIndex;
}
OUTBary[];

////////////////////////////////////////////

#define MESHSHADER_WORKGROUP_SIZE 32

layout(local_size_x = MESHSHADER_WORKGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;
layout(max_vertices = CLUSTER_VERTEX_COUNT, max_primitives = CLUSTER_TRIANGLE_COUNT) out;
layout(triangles) out;

const uint MESHLET_VERTEX_ITERATIONS   = ((CLUSTER_VERTEX_COUNT + MESHSHADER_WORKGROUP_SIZE - 1) / MESHSHADER_WORKGROUP_SIZE);
const uint MESHLET_TRIANGLE_ITERATIONS = ((CLUSTER_TRIANGLE_COUNT + MESHSHADER_WORKGROUP_SIZE - 1) / MESHSHADER_WORKGROUP_SIZE);

////////////////////////////////////////////

void main()
{  
  DrawStatsRef drawStats = DrawStatsRef(rc.drawStatsAddress);

  uint drawIndex = uint(gl_WorkGroupID.x);
  
  DrawCluster dinfo        = DrawClusterArray(rc.drawClustersAddress).array[drawIndex];
  ClusterGeometry cluster  = ClusterGeometryRef(dinfo.cluster).d; 
  Instance instance        = InstanceArray(rc.instancesAddress).array[dinfo.instanceIndex];
  
  uint vertMax = cluster.vertexCount - 1;
  uint triMax  = cluster.triangleCount - 1;

  if (gl_LocalInvocationID.x == 0) {
    gl_PrimitiveCountNV = cluster.triangleCount;
    
    atomicAdd(drawStats.d.triangleCount, uint(cluster.triangleCount));
  }

  if (cluster.triangleCount == 0 || cluster.vertexCount == 0) return;
  
  Vec3Array   oPositions      = Vec3Array(cluster.vertexPositionsAddress);
  U8Vec3Array localTriangles  = U8Vec3Array(cluster.triangleVerticesAddress);

  mat4 worldMatrix   = instance.transform;

  [[unroll]] for(uint i = 0; i < uint(MESHLET_VERTEX_ITERATIONS); i++)
  {
    uint vert        = gl_LocalInvocationID.x + i * MESHSHADER_WORKGROUP_SIZE;
    uint vertLoad    = min(vert, vertMax);
    uint vertexIndex = vertLoad;

    vec3 oPos = oPositions.array[vertexIndex];
    vec4 wPos = worldMatrix * vec4(oPos,1);

    if(vert <= vertMax)
    {
      gl_MeshVerticesNV[vert].gl_Position = frameInfo.viewProj * wPos;
      OUT[vert].drawIndex                 = drawIndex;
      OUTBary[vert].vertexIndex           = vertexIndex;
    }
  }

  [[unroll]] for(uint i = 0; i < uint(MESHLET_TRIANGLE_ITERATIONS); i++)
  {
    uint tri     = gl_LocalInvocationID.x + i * MESHSHADER_WORKGROUP_SIZE;
    uint triLoad = min(tri, triMax);

    u8vec3 indices = localTriangles.array[triLoad];

    if(tri <= triMax)
    {
      gl_PrimitiveIndicesNV[tri * 3 + 0] = indices.x;
      gl_PrimitiveIndicesNV[tri * 3 + 1] = indices.y;
      gl_PrimitiveIndicesNV[tri * 3 + 2] = indices.z;
      gl_MeshPrimitivesNV[tri].gl_PrimitiveID = int(tri);
    }
  }
}