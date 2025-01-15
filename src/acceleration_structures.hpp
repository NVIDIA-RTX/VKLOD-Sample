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

#pragma once

#include <algorithm>
#include <condition_variable>
#include <debug_range_summary.hpp>
#include <iostream>
#include <lod_streaming_jobs.hpp>
#include <ostream>
#include <sample_glsl_compiler.hpp>
#include <sample_producer_consumer.hpp>
#include <sample_raytracing_objects.hpp>
#include <sample_vulkan_objects.hpp>
#include <scene.hpp>
#include <unordered_set>
#include <vulkan/vulkan_core.h>

inline std::ostream& operator<<(std::ostream& os, const VkClusterAccelerationStructureBuildTriangleClusterInfoNV& x)
{
  PrefixedLines indent(os.rdbuf(), "  ");
  std::ostream  ios(&indent);
  ios << "VkClusterAccelerationStructureBuildTriangleClusterInfoNV{\n";
  ios << "clusterID " << x.clusterID << "\n";
  ios << "clusterFlags " << x.clusterFlags << "\n";
  ios << "triangleCount " << x.triangleCount << "\n";
  ios << "vertexCount " << x.vertexCount << "\n";
  ios << "positionTruncateBitCount " << x.positionTruncateBitCount << "\n";
  ios << "indexType " << x.indexType << "\n";
  ios << "opacityMicromapIndexType " << x.opacityMicromapIndexType << "\n";
  ios << "baseGeometryIndexAndGeometryFlags.geometryIndex " << x.baseGeometryIndexAndGeometryFlags.geometryIndex << "\n";
  ios << "baseGeometryIndexAndGeometryFlags.geometryFlags " << x.baseGeometryIndexAndGeometryFlags.geometryFlags << "\n";
  ios << "baseGeometryIndexAndGeometryFlags.reserved " << x.baseGeometryIndexAndGeometryFlags.reserved << "\n";
  ios << "indexBufferStride " << x.indexBufferStride << "\n";
  ios << "vertexBufferStride " << x.vertexBufferStride << "\n";
  ios << "geometryIndexAndFlagsBufferStride " << x.geometryIndexAndFlagsBufferStride << "\n";
  ios << "opacityMicromapIndexBufferStride " << x.opacityMicromapIndexBufferStride << "\n";
  ios << "indexBuffer " << x.indexBuffer << "\n";
  ios << "vertexBuffer " << x.vertexBuffer << "\n";
  ios << "geometryIndexAndFlagsBuffer " << x.geometryIndexAndFlagsBuffer << "\n";
  ios << "opacityMicromapArray " << x.opacityMicromapArray << "\n";
  ios << "opacityMicromapIndexBuffer " << x.opacityMicromapIndexBuffer << "\n";
  ios << "Indices ";
  if(x.indexType == VK_CLUSTER_ACCELERATION_STRUCTURE_INDEX_FORMAT_8BIT_NV && x.indexBufferStride == sizeof(glm::u8vec3::x))
    rangeSummaryVk<glm::u8vec3>(ios, x.indexBuffer, x.triangleCount) << "\n";
  else if(x.indexType == VK_CLUSTER_ACCELERATION_STRUCTURE_INDEX_FORMAT_16BIT_NV && x.indexBufferStride == sizeof(glm::u16vec3::x))
    rangeSummaryVk<glm::u16vec3>(ios, x.indexBuffer, x.triangleCount) << "\n";
  else if(x.indexType == VK_CLUSTER_ACCELERATION_STRUCTURE_INDEX_FORMAT_32BIT_NV && x.indexBufferStride == sizeof(glm::uvec3::x))
    rangeSummaryVk<glm::uvec3>(ios, x.indexBuffer, x.triangleCount) << "\n";
  else
    ios << "<range summary not supported>\n";
  ios << "Vertices ";
  if(x.vertexBufferStride == sizeof(glm::vec3))
    rangeSummaryVk<glm::vec3>(ios, x.vertexBuffer, x.vertexCount) << "\n";
  else
    ios << "<range summary not supported>\n";
  os << "}";
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV& x)
{
  PrefixedLines indent(os.rdbuf(), "  ");
  std::ostream  ios(&indent);
  ios << "VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV{\n";
  ios << "clusterReferencesCount " << x.clusterReferencesCount << "\n";
  ios << "clusterReferencesStride " << x.clusterReferencesStride << "\n";
  ios << "clusterReferences[Address] " << x.clusterReferences << "\n";
  {
    std::span hostArray(reinterpret_cast<const VkDeviceAddress*>(
                            BufferDownloader::download(x.clusterReferences, x.clusterReferencesCount * sizeof(VkDeviceAddress))),
                        x.clusterReferencesCount);
    rangeSummary(ios << "clusterReferences ", hostArray, 20) << "\n";
#if !defined(NDEBUG)
    assert(std::ranges::find(hostArray, VkDeviceAddress(0)) == hostArray.end());  // no nullptrs
    assert(std::unordered_set<VkDeviceAddress>(hostArray.begin(), hostArray.end()).size() == hostArray.size());  // all unique
    for(auto& addr : hostArray)
      assert((addr & 127) == 0);  // properly aligned
#endif
  }
  os << "}";
  return os;
}

// Container for multiple bottom level acceleration structures, including a
// linearized array of input cluster adddresses and
class BlasArray
{
public:
  BlasArray(ResourceAllocator* allocator, uint32_t blasCount, uint32_t maxClustersPerMesh, uint32_t maxTotalClusters);

  struct OldBuffers
  {
    vkobj::Buffer<std::byte>       m_blas;
    vkobj::Buffer<std::byte>       m_blasScratchBuffer;
    vkobj::Buffer<VkDeviceAddress> m_clasAddresses;
  };

  [[nodiscard]] OldBuffers resize(ResourceAllocator* allocator, uint32_t maxTotalClusters);

  const vkobj::Buffer<VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV>& input() const
  {
    return m_blasInfos;
  }

  const vkobj::Buffer<VkDeviceAddress>& inputPointers() const { return m_clasAddresses; }

  // Per-mesh traversal creates a BLAS per mesh that is then shared by instances
  void cmdBuild(VkCommandBuffer cmd, vkobj::Buffer<VkDeviceAddress>& outputBlasAddresses);

  // Per-instance traversal creates a BLAS per instance that can be written directly to the TLAS input by the BLAS build
  void cmdBuild(VkCommandBuffer cmd, vkobj::Buffer<VkAccelerationStructureInstanceKHR>& outputTlasInfos);

  // Common build, called by the above overloads
  void cmdBuild(VkCommandBuffer cmd, VkStridedDeviceAddressRegionKHR addresses);

  VkDeviceSize deviceMemory() const { return m_deviceMemory; }

private:
  // Worst case storage for per-instance BLAS cluster input
  vkobj::Buffer<VkDeviceAddress> m_clasAddresses;

  // Per-BLAS input descriptions
  vkobj::Buffer<VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV> m_blasInfos;

  // Black box acceleration structure and scratch storage for the build
  vkobj::Buffer<std::byte> m_blas;
  vkobj::Buffer<std::byte> m_blasScratchBuffer;

  // alt: VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR
  VkBuildAccelerationStructureFlagsKHR m_buildFlags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;

  VkDeviceSize m_deviceMemory = 0;

  uint32_t m_maxTotalClusters   = 0;
  uint32_t m_maxClustersPerMesh = 0;
};

// Top level acceleration structure. This contains instantiated BLASes.
// Depending on traversal, there may be a BLAS per instance or a BLAS per mesh
// that can be instantiated multiple times.
class Tlas
{
public:
  Tlas(ResourceAllocator* allocator, const vkobj::Buffer<VkAccelerationStructureInstanceKHR>& tlasInfo, VkCommandBuffer initCmd);

  void cmdUpdate(const vkobj::Buffer<VkAccelerationStructureInstanceKHR>& tlasInfo, VkCommandBuffer cmd, bool rebuild);

  // Returns the top level acceleration structure to be used by raytracing
  // shaders
  VkAccelerationStructureKHR output() const { return *m_tlas; }

  VkDeviceSize deviceMemory() const { return m_deviceMemory; }

private:
  std::unique_ptr<rt::BuiltAccelerationStructure> m_tlas;
  vkobj::Buffer<std::byte>                        m_tlasScratchBuffer;

  // alt: VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
  VkBuildAccelerationStructureFlagsKHR m_buildFlags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;

  VkDeviceSize m_deviceMemory = 0;
};

// Holds temporary fixed sized buffers for the streaming thread to build cluster
// acceleration structures in batches, which are then copied and compacted into
// per-LOD-"group" allocations.
class ClasStaging
{
public:
  ClasStaging() = delete;
  ClasStaging(ResourceAllocator*                   allocator,
              const Scene&                         scene,
              uint32_t                             maxGroupsPerBuild,
              uint32_t                             maxClustersPerBuild,
              uint32_t                             positionTruncateBits,
              VkBuildAccelerationStructureFlagsKHR buildFlags);

  // Building and compacting is split into separate calls so that we can execute
  // the build as one batch on the main render thread and vulkan queue. The
  // compaction step could be combined and happen immediately after if it
  // weren't for having a host side allocator. Moving to a GPU based memory pool
  // allocator would be best.
  void buildClas(ResourceAllocator*               allocator,
                 streaming::FillClasInputProgram& fillClasInputProgram,
                 streaming::PackClasProgram&      packClasProgram,
                 shaders::StreamGroupModsList     uploadedMods,
                 std::span<const uint32_t>        loadClusterLoadGroupsHost,
                 std::span<const uint32_t>        loadGroupClusterOffsetsHost,
                 uint32_t                         totalClusters,
                 VkCommandBuffer                  cmd);
  void compactClas(ResourceAllocator*           allocator,
                   PoolAllocator&               memoryPool,
                   vkobj::SemaphoreValue        readySemaphoreState,  // signalled when buildClas(..., cmd) has finished
                   streaming::PackClasProgram&  packClasProgram,
                   shaders::StreamGroupModsList uploadedMods,
                   uint32_t                     totalClusters,
                   VkCommandBuffer              cmd,
                   std::vector<PoolMemory>&     newClases);

  size_t maxClustersPerBuild() const { return m_maxClustersPerBuild; }

  VkPhysicalDeviceClusterAccelerationStructurePropertiesNV clasProperties() const { return m_clasProperties; }

  size_t memoryUsage() const
  {
    // TODO: vkobj::ByteBuffer with size()?
    return m_clasInfo.size_bytes() + m_buildClasSizesInfo.accelerationStructureSize + m_clasAddresses.size_bytes()
           + m_clasPackedAddresses.size_bytes() + m_clasSizes.size_bytes()
           + std::max(m_buildClasSizesInfo.buildScratchSize, m_moveClasSizesInfo.updateScratchSize)
           + m_groupClasAllocNext.size_bytes() + m_groupClasBaseAddresses.size_bytes() + m_loadClusterLoadGroups.size_bytes()
           + m_loadGroupClusterOffsets.size_bytes() + m_groupTotalClasSizes.size_bytes();
  }

private:
  uint32_t m_maxClustersPerBuild;

  VkBuildAccelerationStructureFlagsKHR                     m_buildFlags;
  VkClusterAccelerationStructureTriangleClusterInputNV     m_triangleClusterInput;  // conservative maximums
  VkAccelerationStructureBuildSizesInfoKHR                 m_buildClasSizesInfo;
  VkAccelerationStructureBuildSizesInfoKHR                 m_moveClasSizesInfo;
  VkPhysicalDeviceClusterAccelerationStructurePropertiesNV m_clasProperties;
  vkobj::Buffer<VkClusterAccelerationStructureBuildTriangleClusterInfoNV> m_clasInfo;
  vkobj::ByteBuffer                                                       m_clasData;
  vkobj::Buffer<VkDeviceAddress>                                          m_clasAddresses;
  vkobj::Buffer<VkDeviceAddress>                                          m_clasPackedAddresses;
  vkobj::Buffer<uint32_t>                                                 m_clasSizes;
  vkobj::ByteBuffer                                                       m_clasScratch;
  vkobj::Buffer<uint32_t>                                                 m_groupClasAllocNext;
  vkobj::Buffer<VkDeviceAddress>                                          m_groupClasBaseAddresses;
  vkobj::Buffer<uint32_t>                                                 m_loadClusterLoadGroups;
  vkobj::Buffer<uint32_t>                                                 m_loadGroupClusterOffsets;
  vkobj::Buffer<uint32_t>        m_groupTotalClasSizes;         // dedicated staging buffer
  vkobj::BufferMapping<uint32_t> m_groupTotalClasSizesMapping;  // conservatively sized
  std::span<uint32_t>            m_groupTotalClasSizesHost;     // subspan of m_groupTotalClasSizesMapping
};
