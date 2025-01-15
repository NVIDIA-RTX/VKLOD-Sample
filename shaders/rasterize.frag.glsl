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
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_buffer_reference2 : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_shader_atomic_int64 : enable
#extension GL_EXT_fragment_shader_barycentric : enable


////////////////////////////////////////////

#include "rasterize_device_host.h"

////////////////////////////////////////////

layout(set = 0, binding = BRasterFrameInfo) uniform FrameParams_ { FrameParams frameInfo; };
layout(set = 0, binding = BRasterConstants) uniform RasterParams_ { RasterizeConstants rc; };

///////////////////////////////////////////////////

layout(location = 0) in Interpolants1
{
  flat uint drawIndex;
}
IN;

layout(location = 1) pervertexEXT in Interpolants2
{
  uint vertexIndex;
}
INBary[];

///////////////////////////////////////////////////

layout(location = 0, index = 0) out vec4 out_Color;
layout(early_fragment_tests) in;

///////////////////////////////////////////////////

vec3 hsv2rgb(vec3 c)
{
  vec4 K = {1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0};
  vec3 p = abs(fract(vec3(c.x) + vec3(K)) * 6.0f - vec3(K.w));
  return c.z * mix(vec3(K.x), clamp(p - vec3(K.x), vec3(0.0), vec3(1.0)), c.y);
};


// random hsv([0,1], [.5,1], [.25,1])
vec3 clusterColor(uint h)
{
  h ^= h >> 13;
  h *= 0x5bd1e995;
  h ^= h >> 15;
  return hsv2rgb(vec3(float(h & 0xffff) / 65535.f, float((h >> 16) & 0xff) / 510.f + 0.5, float((h >> 24) & 0xff) / 340.f + 0.25));
};

///////////////////////////////////////////////////

vec2 mixBary(vec2 a, vec2 b, vec2 c, vec3 bary)
{
  return a * bary.x + b * bary.y + c * bary.z;
}

vec3 mixBary(vec3 a, vec3 b, vec3 c, vec3 bary)
{
  return a * bary.x + b * bary.y + c * bary.z;
}

///////////////////////////////////////////////////

void main()
{
  DrawCluster dinfo       = DrawClusterArray(rc.drawClustersAddress).array[IN.drawIndex];
  Instance instance       = InstanceArray(rc.instancesAddress).array[dinfo.instanceIndex];
  Mesh     mesh           = MeshArray(rc.meshesAddress).array[dinfo.meshIndex];

  ClusterGeometry      clusterGeometry         = ClusterGeometryRef(dinfo.cluster).d;
  Vec3Array            clusterVertexPositions  = Vec3Array(clusterGeometry.vertexPositionsAddress);
  Vec3Array            clusterVertexNormals    = Vec3Array(clusterGeometry.vertexNormalsAddress);
  Vec2Array            clusterVertexTexcoords  = Vec2Array(clusterGeometry.vertexTexcoordsAddress);

  uvec3 tri             = uvec3(INBary[0].vertexIndex, INBary[1].vertexIndex, INBary[2].vertexIndex);
  vec3   v0             = clusterVertexPositions.array[tri.x];
  vec3   v1             = clusterVertexPositions.array[tri.y];
  vec3   v2             = clusterVertexPositions.array[tri.z];
  vec3   interpPosition = mixBary(clusterVertexPositions.array[tri.x], clusterVertexPositions.array[tri.y],
    clusterVertexPositions.array[tri.z], gl_BaryCoordEXT);
  vec3   interpNormal   = mixBary(clusterVertexNormals.array[tri.x], clusterVertexNormals.array[tri.y],
    clusterVertexNormals.array[tri.z], gl_BaryCoordEXT);
  vec2   interpTexCoord = clusterGeometry.vertexTexcoordsAddress == 0 ?
    vec2(0.0) :
    mixBary(clusterVertexTexcoords.array[tri.x], clusterVertexTexcoords.array[tri.y],
      clusterVertexTexcoords.array[tri.z], gl_BaryCoordEXT);

  mat3 normalMatrix = mat3(instance.transform);  // assumes uniform scale! no inverse(transpose())
  vec3 wPos         = vec3(instance.transform * vec4(interpPosition, 1.0));
  vec3 wGeomNormU   = normalMatrix * cross(v1 - v0, v2 - v0);
  vec3 wGeomNorm    = normalize(wGeomNormU);
  vec3 wNorm        = normalize(normalMatrix * interpNormal);

  vec3 wDir   = wPos - frameInfo.camPos;
  vec3 wEye   = -wDir;

  float lightDot = dot(wNorm, normalize(wEye));
  
  //out_Color = vec4(vec3(lightDot) * 0.8 + 0.2, 1);

  out_Color = vec4(clusterColor(uint(gl_PrimitiveID)), 1);
}
