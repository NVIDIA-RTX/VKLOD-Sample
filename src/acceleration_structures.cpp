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

#include <acceleration_structures.hpp>
#include <sample_allocation.hpp>
#include <sample_vulkan_objects.hpp>
#include <stdexcept>
#include <vulkan/vulkan_core.h>

BlasArray::BlasArray(ResourceAllocator* allocator, uint32_t blasCount, uint32_t maxClustersPerMesh, uint32_t maxTotalClusters)
    : m_maxClustersPerMesh(maxClustersPerMesh)
{
  // Allocate BLAS input buffer, a per-BLAS list of CLAS addresses
  m_blasInfos = vkobj::Buffer<VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV>(
      allocator, blasCount,
      VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
          | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  m_deviceMemory += m_blasInfos.size_bytes();

  // Call common init and resize code. There will be no garbage on the first
  // call.
  std::ignore = resize(allocator, maxTotalClusters);
}

BlasArray::OldBuffers BlasArray::resize(ResourceAllocator* allocator, uint32_t maxTotalClusters)
{
  LOGI("Reallocating BLAS: %u\n", maxTotalClusters);
  assert(maxTotalClusters > 0);
  m_maxTotalClusters = maxTotalClusters;

  // First, query the driver for the worst case memory usage given the new
  // cluster count
  // DANGER: keep in sync with SceneBlas::cmdBuild()
  VkClusterAccelerationStructureClustersBottomLevelInputNV custerBlasInput{
      .sType                = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_CLUSTERS_BOTTOM_LEVEL_INPUT_NV,
      .pNext                = nullptr,
      .maxTotalClusterCount = m_maxTotalClusters,
      .maxClusterCountPerAccelerationStructure = std::min(m_maxTotalClusters, m_maxClustersPerMesh),  // Far bigger than we should ever generate
  };
  VkClusterAccelerationStructureInputInfoNV inputs = {
      .sType                         = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV,
      .pNext                         = nullptr,
      .maxAccelerationStructureCount = uint32_t(m_blasInfos.size()),
      .flags                         = m_buildFlags,
      .opType                        = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_CLUSTERS_BOTTOM_LEVEL_NV,
      .opMode                        = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV,
      .opInput =
          VkClusterAccelerationStructureOpInputNV{
              .pClustersBottomLevel = &custerBlasInput,
          },
  };
  VkAccelerationStructureBuildSizesInfoKHR sizesInfo = {
      .sType                     = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR,
      .pNext                     = nullptr,
      .accelerationStructureSize = 0,
      .updateScratchSize         = 0,
      .buildScratchSize          = 0,
  };
  vkGetClusterAccelerationStructureBuildSizesNV(allocator->getDevice(), &inputs, &sizesInfo);

  // Allocate a buffer for the input CLAS addresses, linearized for all BLASes
  OldBuffers result;
  m_deviceMemory -= m_clasAddresses.size_bytes();
  result.m_clasAddresses = std::move(m_clasAddresses);
  m_clasAddresses        = vkobj::Buffer<VkDeviceAddress>(allocator, m_maxTotalClusters,
                                                          VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                                              | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                                                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  m_deviceMemory += m_clasAddresses.size_bytes();

  // Allocate space for all output BLASes, i.e. the acceleration structure data
  m_deviceMemory -= m_blas.size_bytes();
  result.m_blas = std::move(m_blas);
  m_blas        = vkobj::Buffer<std::byte>(allocator, sizesInfo.accelerationStructureSize,
                                           VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                               | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
                                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  m_deviceMemory += m_blas.size_bytes();
  assert(static_cast<VkDeviceAddress>(m_blas.address()) % 128 == 0);  // TODO: verify alignment from VkPhysicalDeviceAccelerationStructurePropertiesKHR?

  // Scratch space to build BLASes with a single indirect call
  m_deviceMemory -= m_blasScratchBuffer.size_bytes();
  result.m_blasScratchBuffer = std::move(m_blasScratchBuffer);
  m_blasScratchBuffer        = vkobj::Buffer<std::byte>(allocator, sizesInfo.buildScratchSize,
                                                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  m_deviceMemory += m_blasScratchBuffer.size_bytes();
  return result;
}

void BlasArray::cmdBuild(VkCommandBuffer cmd, vkobj::Buffer<VkDeviceAddress>& outputBlasAddresses)
{
  cmdBuild(cmd, VkStridedDeviceAddressRegionKHR{
                    .deviceAddress = static_cast<VkDeviceAddress>(outputBlasAddresses.address()),
                    .stride        = sizeof(VkDeviceAddress),
                    .size          = VkDeviceSize(outputBlasAddresses.size()),
                });
}

void BlasArray::cmdBuild(VkCommandBuffer cmd, vkobj::Buffer<VkAccelerationStructureInstanceKHR>& outputTlasInfos)
{
  cmdBuild(cmd, VkStridedDeviceAddressRegionKHR{
                    .deviceAddress = static_cast<VkDeviceAddress>(outputTlasInfos.address())
                                     + offsetof(VkAccelerationStructureInstanceKHR, accelerationStructureReference),
                    .stride = sizeof(VkAccelerationStructureInstanceKHR),
                    .size   = VkDeviceSize(outputTlasInfos.size()),
                });
}

void BlasArray::cmdBuild(VkCommandBuffer cmd, VkStridedDeviceAddressRegionKHR addresses)
{
  // DANGER: keep in sync with SceneBlas::resize()
  assert(m_clasAddresses.size() == size_t(m_maxTotalClusters));
  VkClusterAccelerationStructureClustersBottomLevelInputNV custerBlasInput{
      .sType                = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_CLUSTERS_BOTTOM_LEVEL_INPUT_NV,
      .pNext                = nullptr,
      .maxTotalClusterCount = m_maxTotalClusters,
      .maxClusterCountPerAccelerationStructure = std::min(m_maxTotalClusters, m_maxClustersPerMesh),  // Far bigger than we should ever generate
  };
  VkClusterAccelerationStructureInputInfoNV input = {
      .sType                         = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV,
      .pNext                         = nullptr,
      .maxAccelerationStructureCount = uint32_t(m_blasInfos.size()),
      .flags                         = m_buildFlags,
      .opType                        = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_CLUSTERS_BOTTOM_LEVEL_NV,
      .opMode                        = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV,
      .opInput =
          VkClusterAccelerationStructureOpInputNV{
              .pClustersBottomLevel = &custerBlasInput,
          },
  };
  assert(m_blasInfos.size() == addresses.size);
  VkClusterAccelerationStructureCommandsInfoNV blasCommandsInfo = {
      .sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV,
      .pNext = nullptr,
      .input = input,
      .dstImplicitData = static_cast<VkDeviceAddress>(m_blas.address()),  // implicit meaning the driver will populate dstAddressesArray for us
      .scratchData       = static_cast<VkDeviceAddress>(m_blasScratchBuffer.address()),
      .dstAddressesArray = addresses,
      .dstSizesArray =
          VkStridedDeviceAddressRegionKHR{
              .deviceAddress = 0,
              .stride        = 0,
              .size          = 0,
          },
      .srcInfosArray =
          VkStridedDeviceAddressRegionKHR{
              .deviceAddress = static_cast<VkDeviceAddress>(m_blasInfos.address()),
              .stride        = sizeof(VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV),
              .size          = VkDeviceSize(m_blasInfos.size()),
          },
      .srcInfosCount          = 0 /* optional device/dynamic size, but we want everything */,
      .addressResolutionFlags = 0,
  };
  vkCmdBuildClusterAccelerationStructureIndirectNV(cmd, &blasCommandsInfo);

  memoryBarrier(cmd, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR, VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
                VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR);
}

Tlas::Tlas(ResourceAllocator* allocator, const vkobj::Buffer<VkAccelerationStructureInstanceKHR>& tlasInfo, VkCommandBuffer initCmd)
{
  VkAccelerationStructureGeometryKHR instancesGeometry = {
      .sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
      .pNext        = nullptr,
      .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
      .geometry =
          VkAccelerationStructureGeometryDataKHR{
              .instances =
                  {
                      .sType           = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
                      .pNext           = nullptr,
                      .arrayOfPointers = VK_FALSE,
                      .data =
                          VkDeviceOrHostAddressConstKHR{
                              .deviceAddress = static_cast<VkDeviceAddress>(tlasInfo.address()),
                          },
                  },
          },
      .flags = 0,
  };
  VkAccelerationStructureBuildRangeInfoKHR rangeInfo{
      .primitiveCount  = uint32_t(tlasInfo.size()),
      .primitiveOffset = 0,
      .firstVertex     = 0,
      .transformOffset = 0,
  };
  VkBuildAccelerationStructureFlagsKHR buildFlags = m_buildFlags | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR
                                                    | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_DATA_ACCESS_KHR;
  rt::AccelerationStructureSizes sizes(allocator, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, buildFlags,
                                       std::span(&instancesGeometry, 1), std::span(&rangeInfo, 1));

  m_tlasScratchBuffer = vkobj::Buffer<std::byte>(allocator, std::max(sizes->buildScratchSize, sizes->updateScratchSize),
                                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  m_deviceMemory += m_tlasScratchBuffer.size_bytes();

  // We want to record a command buffer that updates the acceleration structure,
  // so we do a regular build first, i.e.
  // VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR now and
  // VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR in cmdUpdate()
  VkAccelerationStructureCreateFlagsKHR createFlags = 0;
  m_tlas                                            = std::make_unique<rt::BuiltAccelerationStructure>(
      rt::AccelerationStructure(allocator, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, *sizes, createFlags), buildFlags,
      std::span(&instancesGeometry, 1), std::span(&rangeInfo, 1), VkDeviceAddress(m_tlasScratchBuffer.address()), initCmd);
  m_deviceMemory += sizes->accelerationStructureSize;

  // Barrier for immediate update afterwards
  memoryBarrier(initCmd, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR);
}

void Tlas::cmdUpdate(const vkobj::Buffer<VkAccelerationStructureInstanceKHR>& tlasInfo, VkCommandBuffer cmd, bool rebuild)
{
  VkAccelerationStructureGeometryKHR instancesGeometry = {
      .sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
      .pNext        = nullptr,
      .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
      .geometry =
          VkAccelerationStructureGeometryDataKHR{
              .instances =
                  {
                      .sType           = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
                      .pNext           = nullptr,
                      .arrayOfPointers = VK_FALSE,
                      .data =
                          VkDeviceOrHostAddressConstKHR{
                              .deviceAddress = static_cast<VkDeviceAddress>(tlasInfo.address()),
                          },
                  },
          },
      .flags = 0,
  };
  VkAccelerationStructureBuildRangeInfoKHR rangeInfo{
      .primitiveCount  = uint32_t(tlasInfo.size()),
      .primitiveOffset = 0,
      .firstVertex     = 0,
      .transformOffset = 0,
  };
  VkBuildAccelerationStructureFlagsKHR buildFlags = m_buildFlags | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR
                                                    | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_DATA_ACCESS_KHR;

  // Record a build or update into cmd
  m_tlas->build(buildFlags, std::span(&instancesGeometry, 1), std::span(&rangeInfo, 1), !rebuild /* update */,
                static_cast<VkDeviceAddress>(m_tlasScratchBuffer.address()), cmd);
  memoryBarrier(cmd, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
                VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR);
}

ClasStaging::ClasStaging(ResourceAllocator*                   allocator,
                         const Scene&                         scene,
                         uint32_t                             maxGroupsPerBuild,
                         uint32_t                             maxClustersPerBuild,
                         uint32_t                             positionTruncateBits,
                         VkBuildAccelerationStructureFlagsKHR buildFlags)
    : m_maxClustersPerBuild(maxClustersPerBuild)
    , m_buildFlags(buildFlags)
    , m_groupTotalClasSizes(allocator,
                            maxClustersPerBuild,
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT)
    , m_groupTotalClasSizesMapping(m_groupTotalClasSizes)
{
  // Query alignment requirements
  m_clasProperties = VkPhysicalDeviceClusterAccelerationStructurePropertiesNV{
      .sType                           = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CLUSTER_ACCELERATION_STRUCTURE_PROPERTIES_NV,
      .pNext                           = nullptr,
      .maxVerticesPerCluster           = 0xffffffffu,
      .maxTrianglesPerCluster          = 0xffffffffu,
      .clusterScratchByteAlignment     = 0xffffffffu,
      .clusterByteAlignment            = 0xffffffffu,
      .clusterTemplateByteAlignment    = 0xffffffffu,
      .clusterBottomLevelByteAlignment = 0xffffffffu,
      .clusterTemplateBoundsByteAlignment = 0xffffffffu,
      .maxClusterGeometryIndex            = 0xffffffffu,
  };
  VkPhysicalDeviceProperties2 props2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &m_clasProperties, .properties = {}};
  vkGetPhysicalDeviceProperties2(allocator->getPhysicalDevice(), &props2);

  // Compute conservative CLAS sizes
  assert(scene.counts.maxClusterTriangleCount <= m_clasProperties.maxTrianglesPerCluster);
  assert(scene.counts.maxClusterVertexCount <= m_clasProperties.maxVerticesPerCluster);
  m_triangleClusterInput = VkClusterAccelerationStructureTriangleClusterInputNV{
      .sType                         = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_TRIANGLE_CLUSTER_INPUT_NV,
      .pNext                         = nullptr,
      .vertexFormat                  = VK_FORMAT_R32G32B32_SFLOAT,
      .maxGeometryIndexValue         = 0,
      .maxClusterUniqueGeometryCount = 0,
      .maxClusterTriangleCount       = scene.counts.maxClusterTriangleCount,
      .maxClusterVertexCount         = scene.counts.maxClusterVertexCount,
      .maxTotalTriangleCount         = maxClustersPerBuild * scene.counts.maxClusterTriangleCount,
      .maxTotalVertexCount           = maxClustersPerBuild * scene.counts.maxClusterVertexCount,
      .minPositionTruncateBitCount   = positionTruncateBits,
  };
  VkClusterAccelerationStructureInputInfoNV buildClasInputInfo = {
      .sType                         = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV,
      .pNext                         = nullptr,
      .maxAccelerationStructureCount = maxClustersPerBuild,
      .flags                         = m_buildFlags,
      .opType                        = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_NV,
      .opMode                        = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV,
      .opInput = VkClusterAccelerationStructureOpInputNV{.pTriangleClusters = &m_triangleClusterInput},
  };
  m_buildClasSizesInfo = VkAccelerationStructureBuildSizesInfoKHR{
      .sType                     = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR,
      .pNext                     = nullptr,
      .accelerationStructureSize = 0xffffffffffffffffull /* output */,
      .updateScratchSize         = 0xffffffffffffffffull /* output */,
      .buildScratchSize          = 0xffffffffffffffffull /* output */,
  };
  vkGetClusterAccelerationStructureBuildSizesNV(allocator->getDevice(), &buildClasInputInfo, &m_buildClasSizesInfo);

  // Compute scratch memory for a move and compaction operation to per-group
  // CLAS buffers. CLASes are built to a conservatively sized output. To
  // conserve memory, the acceleration structures are moved and compacted
  // after the build.
  VkClusterAccelerationStructureMoveObjectsInputNV moveInfo = {
      .sType         = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_MOVE_OBJECTS_INPUT_NV,
      .pNext         = nullptr,
      .type          = VK_CLUSTER_ACCELERATION_STRUCTURE_TYPE_TRIANGLE_CLUSTER_NV,
      .noMoveOverlap = true,  // not in-place, copying to separate buffers
      .maxMovedBytes = uint32_t(m_buildClasSizesInfo.accelerationStructureSize),
  };
  VkClusterAccelerationStructureInputInfoNV moveObjectsInputInfo = {
      .sType                         = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV,
      .pNext                         = nullptr,
      .maxAccelerationStructureCount = maxClustersPerBuild,
      .flags                         = m_buildFlags,
      .opType                        = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_MOVE_OBJECTS_NV,
      .opMode                        = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV,
      .opInput                       = VkClusterAccelerationStructureOpInputNV{.pMoveObjects = &moveInfo},
  };
  m_moveClasSizesInfo = VkAccelerationStructureBuildSizesInfoKHR{
      .sType                     = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR,
      .pNext                     = nullptr,
      .accelerationStructureSize = 0xffffffffffffffffull /* output */,
      .updateScratchSize         = 0xffffffffffffffffull /* output */,
      .buildScratchSize          = 0xffffffffffffffffull /* output */,
  };
  vkGetClusterAccelerationStructureBuildSizesNV(allocator->getDevice(), &moveObjectsInputInfo, &m_moveClasSizesInfo);

  // Temporary buffers to build just one batch of acceleration structures
  VkDeviceSize scratchSize = std::max(m_buildClasSizesInfo.buildScratchSize, m_moveClasSizesInfo.updateScratchSize);  // TODO: buildScratchSize or buildScratchSize to move?
  m_clasData      = vkobj::ByteBuffer(allocator, m_buildClasSizesInfo.accelerationStructureSize,
                                      VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                          | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
                                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  m_clasAddresses = vkobj::Buffer<VkDeviceAddress>(allocator, maxClustersPerBuild,
                                                   VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  m_clasPackedAddresses =
      vkobj::Buffer<VkDeviceAddress>(allocator, maxClustersPerBuild,
                                     VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                         | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  m_groupClasAllocNext = vkobj::Buffer<uint32_t>(allocator, maxGroupsPerBuild,
                                                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                                     | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                                                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  m_groupClasBaseAddresses =
      vkobj::Buffer<VkDeviceAddress>(allocator, maxGroupsPerBuild,
                                     VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                         | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  m_clasInfo = vkobj::Buffer<VkClusterAccelerationStructureBuildTriangleClusterInfoNV>(
      allocator, maxClustersPerBuild,
      VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
          | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  m_clasSizes = vkobj::Buffer<uint32_t>(allocator, maxClustersPerBuild,
                                        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                            | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  m_clasScratch = vkobj::ByteBuffer(allocator, scratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  m_loadClusterLoadGroups   = vkobj::Buffer<uint32_t>(allocator, maxClustersPerBuild,
                                                      VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  m_loadGroupClusterOffsets = vkobj::Buffer<uint32_t>(allocator, maxGroupsPerBuild,
                                                      VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
}

void ClasStaging::buildClas(ResourceAllocator*               allocator,
                            streaming::FillClasInputProgram& fillClasInputProgram,
                            streaming::PackClasProgram&      packClasProgram,
                            shaders::StreamGroupModsList     uploadedMods,
                            std::span<const uint32_t>        loadClusterLoadGroupsHost,
                            std::span<const uint32_t>        loadGroupClusterOffsetsHost,
                            uint32_t                         totalClusters,
                            VkCommandBuffer                  cmd)
{
  assert(loadClusterLoadGroupsHost.size() <= maxClustersPerBuild());
  assert(m_groupTotalClasSizesHost.size() == 0);
  uint32_t newGroupCount = uploadedMods.loadGroupCount;

  // Barrier from uploading data in ClusterGroupGeometryVk() and
  // GroupModsList::write() to its direct use in the acceleration data
  // structure build. Added for completeness; probably reduncant with
  // subsequent barriers.
  memoryBarrier(cmd, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR);

  // Temporary data for per-cluster threads to look up their group and cluster
  // index within the group
  std::ranges::copy(loadClusterLoadGroupsHost,
                    allocator->getStaging()->cmdToBufferT<uint32_t>(cmd, m_loadClusterLoadGroups, 0,
                                                                    loadClusterLoadGroupsHost.size() * sizeof(uint32_t)));
  std::ranges::copy(loadGroupClusterOffsetsHost,
                    allocator->getStaging()->cmdToBufferT<uint32_t>(cmd, m_loadGroupClusterOffsets, 0,
                                                                    loadGroupClusterOffsetsHost.size() * sizeof(uint32_t)));

  memoryBarrier(cmd, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

  // Have the GPU populate
  // VkClusterAccelerationStructureBuildTriangleClusterInfoNV CLAS
  // input structures since addresses are already resident
  shaders::FillClasInputConstants fillClasInputConstants{
      .loadGroupsAddress              = uploadedMods.loadGroupsAddress,
      .clasInfoAddress                = deviceReinterpretCast<shaders::ClusterCLASInfoNV>(m_clasInfo.address()),
      .loadClusterLoadGroupsAddress   = m_loadClusterLoadGroups.address(),
      .loadGroupClusterOffsetsAddress = m_loadGroupClusterOffsets.address(),
      .clusterCount                   = totalClusters,
      .positionTruncateBits           = m_triangleClusterInput.minPositionTruncateBitCount,
  };
  vkCmdPushConstants(cmd, fillClasInputProgram.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                     sizeof(fillClasInputConstants), &fillClasInputConstants);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, fillClasInputProgram.pipeline);
  vkCmdDispatch(cmd, div_ceil(totalClusters, uint32_t(STREAM_WORKGROUP_SIZE)), 1, 1);
  memoryBarrier(cmd, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR);

  // Build all cluster acceleration structures in one indirect call
  VkClusterAccelerationStructureInputInfoNV buildClasInputInfo = {
      .sType                         = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV,
      .pNext                         = nullptr,
      .maxAccelerationStructureCount = totalClusters,  // NOTE: set the exact CLAS count (sizes were based on the max.)
      .flags                         = m_buildFlags,
      .opType                        = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_NV,
      .opMode                        = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV,
      .opInput = VkClusterAccelerationStructureOpInputNV{.pTriangleClusters = &m_triangleClusterInput},
  };
  VkClusterAccelerationStructureCommandsInfoNV buildClasCmdInfo = {
      .sType           = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV,
      .pNext           = nullptr,
      .input           = buildClasInputInfo,
      .dstImplicitData = m_clasData.address(),  // implicit meaning the driver will populate dstAddressesArray for us
      .scratchData     = m_clasScratch.address(),
      .dstAddressesArray =
          VkStridedDeviceAddressRegionKHR{
              .deviceAddress = static_cast<VkDeviceAddress>(m_clasAddresses.address()),
              .stride        = sizeof(VkDeviceAddress),
              .size          = totalClusters,
          },
      .dstSizesArray =
          VkStridedDeviceAddressRegionKHR{
              .deviceAddress = static_cast<VkDeviceAddress>(m_clasSizes.address()),
              .stride        = sizeof(uint32_t),
              .size          = totalClusters,
          },
      .srcInfosArray =
          VkStridedDeviceAddressRegionKHR{
              .deviceAddress = static_cast<VkDeviceAddress>(m_clasInfo.address()),
              .stride        = sizeof(VkClusterAccelerationStructureBuildTriangleClusterInfoNV),
              .size          = totalClusters,
          },
      .srcInfosCount          = 0 /* optional device/dynamic size, but we want everything */,
      .addressResolutionFlags = 0,
  };
  vkCmdBuildClusterAccelerationStructureIndirectNV(cmd, &buildClasCmdInfo);
  memoryBarrier(cmd, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR, VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
                VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR);

  // Compute compacted per-group destination address offsets. Clear
  // the buffer first so all offsets start at zero.
  vkCmdFillBuffer(cmd, m_groupClasAllocNext, 0, sizeof(uint32_t) * newGroupCount, 0);
  memoryBarrier(cmd, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  shaders::PackClasConstants packClasConstants{
      .loadGroupsAddress              = uploadedMods.loadGroupsAddress,
      .loadClusterLoadGroupsAddress   = m_loadClusterLoadGroups.address(),
      .loadGroupClusterOffsetsAddress = m_loadGroupClusterOffsets.address(),
      .clasSizesAddress               = m_clasSizes.address(),
      .groupClasAllocNextAddress      = m_groupClasAllocNext.address(),  // output
      .packedClasAddressesAddress     = vkobj::DeviceAddress<uint64_t>(0llu),
      .groupClasBaseAddressesAddress  = vkobj::DeviceAddress<uint64_t>(0llu),
      .clusterCount                   = totalClusters,
  };
  vkCmdPushConstants(cmd, packClasProgram.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(packClasConstants), &packClasConstants);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, packClasProgram.pipeline);
  vkCmdDispatch(cmd, div_ceil(totalClusters, uint32_t(STREAM_WORKGROUP_SIZE)), 1, 1);
  memoryBarrier(cmd, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT);

  // Download the packed sizes. This won't happen immediately. The pointer
  // becomes valid after the command buffer has been executed. Note that we
  // use a dedicated staging buffer, not the allocator's staging buffers,
  // because the buffer and its mapped pointer needs to remain valid until the
  // call to compactClas().
  {
    VkBufferCopy bufferCopy{.srcOffset = 0, .dstOffset = 0, .size = sizeof(uint32_t) * newGroupCount};
    vkCmdCopyBuffer(cmd, m_groupClasAllocNext, m_groupTotalClasSizes, 1, &bufferCopy);
    m_groupTotalClasSizesHost = m_groupTotalClasSizesMapping.span().subspan(0, newGroupCount);
  }
}

void ClasStaging::compactClas(ResourceAllocator*           allocator,
                              PoolAllocator&               memoryPool,
                              vkobj::SemaphoreValue        readySemaphoreState,
                              streaming::PackClasProgram&  packClasProgram,
                              shaders::StreamGroupModsList uploadedMods,
                              uint32_t                     totalClusters,
                              VkCommandBuffer              cmd,
                              std::vector<PoolMemory>&     newClases)
{
  // Make sure buildClas() has completed before reading
  // clasSizes.groupTotalClasSizesHost
  if(!readySemaphoreState.wait(allocator->getDevice(), 0 /* don't wait */))
    throw std::runtime_error("CLAS sizes not ready in compactClas()");

  size_t newGroupCount = m_groupTotalClasSizesHost.size();
  assert(newGroupCount != 0);

  // Allocate per-group CLAS data and upload the allocated address
  assert(newClases.empty());
  std::span<uint64_t> groupClasBaseAddressesHost(
      allocator->getStaging()->cmdToBufferT<uint64_t>(cmd, m_groupClasBaseAddresses, 0, sizeof(uint64_t) * newGroupCount), newGroupCount);
  memoryBarrier(cmd, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  for(size_t loadIndex = 0; loadIndex < newGroupCount; ++loadIndex)
  {
    newClases.emplace_back(PoolMemory(memoryPool, m_groupTotalClasSizesHost[loadIndex], m_clasProperties.clusterByteAlignment));
    groupClasBaseAddressesHost[loadIndex] = newClases.back();
  }

  // Add the allocated base addresses to the offsets, writing both:
  // - packedClasAddresses for vulkan to move the CLASes
  // - shaders::ClusterGroup::clasAddressesAddress array for
  //   rendering
  // The groupClasAddressOffsets is cleared again to re-compute
  // offsets from the base address. An alternative would be to write
  // the offsets earlier and append the base address at the cost of
  // an additional fetch.
  vkCmdFillBuffer(cmd, m_groupClasAllocNext, 0, sizeof(uint32_t) * newGroupCount, 0);
  memoryBarrier(cmd, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  shaders::PackClasConstants packClasConstants{
      .loadGroupsAddress              = uploadedMods.loadGroupsAddress,
      .loadClusterLoadGroupsAddress   = m_loadClusterLoadGroups.address(),
      .loadGroupClusterOffsetsAddress = m_loadGroupClusterOffsets.address(),
      .clasSizesAddress               = m_clasSizes.address(),
      .groupClasAllocNextAddress      = m_groupClasAllocNext.address(),
      .packedClasAddressesAddress     = m_clasPackedAddresses.address(),     // one of two outputs
      .groupClasBaseAddressesAddress  = m_groupClasBaseAddresses.address(),  // new input
      .clusterCount                   = totalClusters,
  };
  vkCmdPushConstants(cmd, packClasProgram.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(packClasConstants), &packClasConstants);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, packClasProgram.pipeline);
  vkCmdDispatch(cmd, div_ceil(totalClusters, uint32_t(STREAM_WORKGROUP_SIZE)), 1, 1);
  memoryBarrier(cmd, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR);

  // Copy and tightly pack CLASes from the temporary array
  // to per-group allocations
  static_assert(sizeof(VkClusterAccelerationStructureMoveObjectsInfoNV) == sizeof(VkDeviceSize));
  VkClusterAccelerationStructureMoveObjectsInputNV moveInfo = {
      .sType         = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_MOVE_OBJECTS_INPUT_NV,
      .pNext         = nullptr,
      .type          = VK_CLUSTER_ACCELERATION_STRUCTURE_TYPE_TRIANGLE_CLUSTER_NV,
      .noMoveOverlap = true,  // not in-place, copying to separate buffers
      .maxMovedBytes = uint32_t(m_buildClasSizesInfo.accelerationStructureSize),
  };
  VkClusterAccelerationStructureInputInfoNV moveObjectsInputInfo = {
      .sType                         = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV,
      .pNext                         = nullptr,
      .maxAccelerationStructureCount = totalClusters,  // NOTE: set the exact CLAS count (sizes were based on the max.)
      .flags                         = m_buildFlags,
      .opType                        = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_MOVE_OBJECTS_NV,
      .opMode                        = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV,
      .opInput                       = VkClusterAccelerationStructureOpInputNV{.pMoveObjects = &moveInfo},
  };
  VkClusterAccelerationStructureCommandsInfoNV moveClasCmdInfo = {
      .sType           = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV,
      .pNext           = nullptr,
      .input           = moveObjectsInputInfo,
      .dstImplicitData = 0,  // dstAddressesArray is already populated
      .scratchData     = m_clasScratch.address(),
      .dstAddressesArray =
          VkStridedDeviceAddressRegionKHR{
              .deviceAddress = static_cast<VkDeviceAddress>(m_clasPackedAddresses.address()),
              .stride        = sizeof(VkDeviceAddress),
              .size          = totalClusters,
          },
      .dstSizesArray =
          VkStridedDeviceAddressRegionKHR{
              .deviceAddress = 0,
              .stride        = sizeof(uint32_t),
              .size          = 0,
          },
      .srcInfosArray =
          VkStridedDeviceAddressRegionKHR{
              .deviceAddress = static_cast<VkDeviceAddress>(m_clasAddresses.address()),
              .stride        = sizeof(VkClusterAccelerationStructureMoveObjectsInfoNV),
              .size          = totalClusters,
          },
      .srcInfosCount          = 0 /* optional indirect size, but we already have totalClusters host side */,
      .addressResolutionFlags = 0,
  };
  vkCmdBuildClusterAccelerationStructureIndirectNV(cmd, &moveClasCmdInfo);
  memoryBarrier(cmd, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR, VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
                VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR);

  m_groupTotalClasSizesHost = {};  // Marker for out of order buildClas() + compactClas()
}
