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

#ifndef LOD_STRUCTS_H
#define LOD_STRUCTS_H
#ifdef __cplusplus
namespace shaders {
#endif  // __cplusplus

// VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV
struct ClusterBLASInfoNV
{
  uint32_t clusterReferencesCount;
  uint32_t clusterReferencesStride;
  uint64_t clusterReferences;
};

// VkClusterAccelerationStructureBuildTriangleClusterInfoNV
struct ClusterCLASInfoNV
{
  uint32_t clusterID;
  uint32_t clusterFlags;

  uint32_t bits0;
  //{
  //  uint32_t triangleCount : 9;
  //  uint32_t vertexCount : 9;
  //  uint32_t positionTruncateBitCount : 6;
  //  uint32_t indexType : 4;
  //  uint32_t opacityMicromapIndexType : 4;
  //}

  uint32_t bits1;
  //{
  //  uint32_t baseGeometryIndex : 24;
  //  uint32_t reserved : 5;
  //  uint32_t baseGeometryFlags : 3;
  //}

  uint16_t indexBufferStride;
  uint16_t vertexBufferStride;
  uint16_t geometryIndexAndFlagsBufferStride;
  uint16_t opacityMicromapIndexBufferStride;
  uint64_t indexBuffer;
  uint64_t vertexBuffer;
  uint64_t geometryIndexAndFlagsBuffer;
  uint64_t opacityMicromapArray;
  uint64_t opacityMicromapIndexBuffer;
};

// VkAccelerationStructureInstanceKHR
struct InstanceInfo
{
  float    transform[3][4];
  uint32_t instanceCustomIndexAndMask;
  uint32_t instanceShaderBindingTableRecordOffsetAndFlags;
  uint64_t accelerationStructureReference;
};

struct NodeRange
{
  uint32_t clusterNode;         // bits: 1
  uint32_t childOffset;         // bits: 26
  uint32_t childCountMinusOne;  // bits: 5
};

struct ClusterRange
{
  uint32_t clusterNode;           // bits: 1
  uint32_t clusterGroup;          // bits: 23
  uint32_t clusterCountMinusOne;  // bits: 8
};

struct Node
{
  //union
  //{
  //  NodeRange    children;
  //  ClusterRange clusters;
  //};
  uint32_t childrenOrClusters;

  //vec4   boundingSphere;
  float bsx;
  float bsy;
  float bsz;
  float bsw;

  float maxClusterQuadricError;
};

#ifndef __cplusplus
void pushBits(inout uint32_t result, uint32_t bitValues, int bitCount)
{
  result = bitfieldInsert(result << bitCount, bitValues, 0, bitCount);
}

uint32_t popBits(inout uint32_t value, int bitCount)
{
  uint32_t result = bitfieldExtract(value, 0, bitCount);
  value >>= bitCount;
  return result;
}

void pushBits(inout uint64_t result, uint64_t bitValues, int bitCount)
{
  uint64_t mask = (uint64_t(1u) << bitCount) - 1;
  result <<= bitCount;
  result |= bitValues & mask;
}

uint64_t popBits(inout uint64_t value, int bitCount)
{
  uint64_t mask   = (uint64_t(1) << bitCount) - 1;
  uint64_t result = value & mask;
  value >>= bitCount;
  return result;
}

NodeRange decodeChildren(uint32_t encoded)
{
  NodeRange result;
  result.clusterNode        = popBits(encoded, 1);
  result.childOffset        = popBits(encoded, 26);
  result.childCountMinusOne = popBits(encoded, 5);
  return result;
}

ClusterRange decodeClusters(uint32_t encoded)
{
  ClusterRange result;
  result.clusterNode          = popBits(encoded, 1);
  result.clusterGroup         = popBits(encoded, 23);
  result.clusterCountMinusOne = popBits(encoded, 8);
  return result;
}
#endif

#ifdef __cplusplus
}  // namespace shaders
#endif
#endif  // LOD_STRUCTS_H
