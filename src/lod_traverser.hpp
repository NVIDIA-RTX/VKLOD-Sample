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

#pragma once

#include <acceleration_structures.hpp>
#include <debug_range_summary.hpp>
#include <nvvk/profiler_vk.hpp>
#include <optional>
#include <sample_glsl_compiler.hpp>
#include <sample_vulkan_objects.hpp>
#include <scene.hpp>
#include <shaders/traverse_device_host.h>

// Returns parameters that will give the lowest detail mesh
inline shaders::TraversalParams initialTraversalParams()
{
  return shaders::TraversalParams{
      .viewTransform              = glm::translate(glm::mat4(1.0f), {0, 0, 1e10f}),
      .distanceToUNorm32          = 1.0f,
      .errorOverDistanceThreshold = 1.0f,
      .useOcclusion               = 0,
      .hizViewProj                = glm::translate(glm::mat4(1.0f), {0, 0, 1e10f}),
      .hizSizeFactors             = glm::vec4(1.0f),
      .hizViewport                = glm::vec2(1.0f),
      .hizSizeMax                 = 1.0f,
  };
}

// Traversal related garbage. Explicit types for demonstration. Could equally
// be a std::variant, std::any or std::function/std::move_only_function.
struct OldTraverseBuffers
{
  vkobj::Buffer<shaders::EncodedClusterJob> clusterQueue;
};
struct OldTraverseAndBVHBuffers
{
  OldTraverseBuffers    clusterQueue;
  BlasArray::OldBuffers blasBuffers;
};

// Encapsulate LOD hierarchy traversal, which computes which clusters to render
// for a given camera view for all instances. Holds intermediate job queue data
// and traversal compute shaders.
// The *instance* traverser produces a BLAS for each instance to match its LOD.
// This has a substantial performance and memory cost and doesn't save much for
// raytracing since the BVH for the highest detail instance must be built either
// way. See the *mesh* traverser below.
class LodInstanceTraverser
{
public:
  LodInstanceTraverser(ResourceAllocator*  allocator,
                       SampleGlslCompiler& glslCompiler,
                       VkCommandPool       initPool,
                       VkQueue             initQueue,
                       uint32_t            initQueueFamilyIndex,
                       const Scene&        scene,
                       const SceneVK&      sceneVk);
  [[nodiscard]] OldTraverseAndBVHBuffers traverseAndBuildBVH(ResourceAllocator*              allocator,
                                                             const shaders::TraversalParams& traversalParams,
                                                             const SceneVK&                  sceneVk,
                                                             nvvk::ProfilerVK&               profiler,
                                                             VkCommandBuffer                 cmd);
  VkDeviceSize                           traversalMemory() const
  {
    return m_nodeQueue.size_bytes() + m_clusterQueue.size_bytes() + m_jobStatus.size_bytes();
  }
  VkAccelerationStructureKHR tlas() const { return m_tlas->output(); }
  VkDeviceSize               blasDeviceMemory() const { return m_blas.deviceMemory(); }
  VkDeviceSize tlasDeviceMemory() const { return m_tlas.value().deviceMemory() + m_tlasInfos.size_bytes(); }

private:
  [[nodiscard]] OldTraverseBuffers traverse(ResourceAllocator*              allocator,
                                            const shaders::TraversalParams& traversalParams,
                                            const SceneVK&                  sceneVk,
                                            const vkobj::Buffer<VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV>& blasInput,
                                            const vkobj::Buffer<VkDeviceAddress>& blasInputClusters,
                                            VkCommandBuffer                       cmd);

  vkobj::Buffer<shaders::TraversalConstants>        m_traversalConstants;
  vkobj::SingleDescriptorSet                        m_uboDescriptorSet;
  vkobj::SimpleComputePipeline<>                    m_traverseInitPipeline;
  vkobj::SimpleComputePipeline<>                    m_traversePipeline;
  vkobj::SimpleComputePipeline<>                    m_traverseVerifyPipeline;
  vkobj::Buffer<shaders::EncodedNodeJob>            m_nodeQueue;
  vkobj::Buffer<shaders::EncodedClusterJob>         m_clusterQueue;
  vkobj::Buffer<shaders::JobStatus>                 m_jobStatus;
  BlasArray                                         m_blas;
  vkobj::Buffer<VkAccelerationStructureInstanceKHR> m_tlasInfos;
  std::optional<Tlas>                               m_tlas;  // optional for delayed init
};

// Encapsulate LOD hierarchy traversal, which computes which clusters to render
// for a given camera view for all instances. Holds intermediate job queue data
// and traversal compute shaders.
// The *mesh* traverser produces a BLAS for each mesh to match the highest LOD
// of the closest few instances. The same BLAS is used for all instances, which
// saves time and memory for raytracing at the cost of over-detailed instances
// in the distance. The trace performance of this is insignificant.
class LodMeshTraverser
{
public:
  LodMeshTraverser(ResourceAllocator*  allocator,
                   SampleGlslCompiler& glslCompiler,
                   VkCommandPool       initPool,
                   VkQueue             initQueue,
                   uint32_t            initQueueFamilyIndex,
                   const Scene&        scene,
                   const SceneVK&      sceneVk);

  [[nodiscard]] OldTraverseAndBVHBuffers traverseAndBuildBVH(ResourceAllocator*              allocator,
                                                             const shaders::TraversalParams& traversalParams,
                                                             const SceneVK&                  sceneVk,
                                                             nvvk::ProfilerVK&               profiler,
                                                             VkCommandBuffer                 cmd);
  VkDeviceSize                           traversalMemory() const
  {
    return m_nodeQueue.size_bytes() + m_clusterQueue.size_bytes() + m_jobStatus.size_bytes();
  }
  VkAccelerationStructureKHR tlas() const { return m_tlas->output(); }
  VkDeviceSize               blasDeviceMemory() const { return m_blas.deviceMemory(); }
  VkDeviceSize tlasDeviceMemory() const { return m_tlas.value().deviceMemory() + m_tlasInfos.size_bytes(); }

private:
  [[nodiscard]] OldTraverseBuffers traverse(ResourceAllocator*              allocator,
                                            const shaders::TraversalParams& traversalParams,
                                            const SceneVK&                  sceneVk,
                                            const vkobj::Buffer<VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV>& blasInput,
                                            const vkobj::Buffer<VkDeviceAddress>& blasInputClusters,
                                            VkCommandBuffer                       cmd);

  vkobj::Buffer<shaders::TraversalConstants>                    m_traversalConstants;
  vkobj::SingleDescriptorSet                                    m_uboDescriptorSet;
  vkobj::SimpleComputePipeline<shaders::SortInstancesConstant>  m_traverseSortInstances;
  vkobj::SimpleComputePipeline<>                                m_traverseInit;
  vkobj::SimpleComputePipeline<>                                m_traverse;
  vkobj::SimpleComputePipeline<>                                m_traverseVerify;
  vkobj::SimpleComputePipeline<shaders::WriteInstancesConstant> m_instanceWriter;
  vkobj::Buffer<shaders::EncodedNodeJob>                        m_nodeQueue;
  vkobj::Buffer<shaders::EncodedClusterJob>                     m_clusterQueue;
  vkobj::Buffer<shaders::JobStatus>                             m_jobStatus;
  vkobj::Buffer<shaders::MeshInstances>                         m_meshInstances;
  vkobj::Buffer<shaders::SortingMeshInstances>                  m_sortingMeshInstances;
  vkobj::Buffer<VkDeviceAddress>                                m_blasAddresses;
  BlasArray                                                     m_blas;
  vkobj::Buffer<VkAccelerationStructureInstanceKHR>             m_tlasInfos;
  std::optional<Tlas>                                           m_tlas;  // optional for delayed init
};

namespace shaders {

inline std::ostream& operator<<(std::ostream& os, const shaders::MeshInstances& x)
{
  using numerical_chars::operator<<;
  using ::operator<<;
  os << "MeshInstances{\n";
  for(int i = 0; i < TRAVERSAL_NEAREST_INSTANCE_COUNT; ++i)
  {
    if(i != 0)
      os << ", ";
    os << "  [" << i << "]={\n";
    os << "    instanceToEye " << x.instanceToEye[i] << "\n";
    os << "    uniformScale " << x.uniformScale[i] << "\n";
    os << "    enabled " << x.enabled[i] << "\n";
    os << "  }";
  }
  os << "}";
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const shaders::SortingMeshInstances& x)
{
  using numerical_chars::operator<<;
  os << "SortingMeshInstances{\n";
  for(int i = 0; i < TRAVERSAL_NEAREST_INSTANCE_COUNT; ++i)
  {
    os << (i == 0 ? "  " : ", ");
    os << "[" << i << "]={\n";
    os << "    nearest w0 " << (x.nearest[i] >> 32) << "\n";
    os << "    nearest w1 " << (x.nearest[i] & 0xffffffff) << "\n";
    os << "  }";
  }
  os << "}";
  return os;
}

}  // namespace shaders
