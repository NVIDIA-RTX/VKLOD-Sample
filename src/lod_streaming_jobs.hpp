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

#include <any>
#include <deque>
#include <memory_resource>
#include <nvvk/compute_vk.hpp>
#include <sample_glsl_compiler.hpp>
#include <sample_raytracing_objects.hpp>
#include <sample_vulkan_objects.hpp>
#include <scene.hpp>
#include <shaders/shaders_stream.h>
#include <unordered_set>
#include <vulkan/vulkan_core.h>

namespace streaming {

// Container to track ClusterGroupGeometryVk and Allocation since they are
// each created at different times. This is the granulartiy of streaming,
// split into two separate allocations.
struct ClusterGroupVk
{
  ClusterGroupGeometryVk geometry;
  PoolMemory             clasAddresses;
  ClusterGroupVk(ClusterGroupGeometryVk&& geometry_, PoolMemory&& clasAddresses_)
      : geometry(std::move(geometry_))
      , clasAddresses(std::move(clasAddresses_))
  {
  }
  ClusterGroupVk(ClusterGroupGeometryVk&& geometry_)
      : geometry(std::move(geometry_))
  {
  }
};

// Separate types so that a function can take a specific program as a parameter
class MakeRequestsProgram : public vkobj::SimpleComputePipeline<shaders::BuildRequestsConstants>
{
public:
  MakeRequestsProgram(VkDevice device, SampleGlslCompiler& glslCompiler)
      : vkobj::SimpleComputePipeline<shaders::BuildRequestsConstants>(device, glslCompiler, "stream_make_requests.comp.glsl")
  {
  }
};

class ModifyGroupsProgram : public vkobj::SimpleComputePipeline<shaders::StreamGroupsConstants>
{
public:
  ModifyGroupsProgram(VkDevice device, SampleGlslCompiler& glslCompiler)
      : vkobj::SimpleComputePipeline<shaders::StreamGroupsConstants>(device, glslCompiler, "stream_modify_groups.comp.glsl")
  {
  }
};

class FillClasInputProgram : public vkobj::SimpleComputePipeline<shaders::FillClasInputConstants>
{
public:
  FillClasInputProgram(VkDevice device, SampleGlslCompiler& glslCompiler)
      : vkobj::SimpleComputePipeline<shaders::FillClasInputConstants>(device, glslCompiler, "stream_fill_clas_input.comp.glsl")
  {
  }
};

class PackClasProgram : public vkobj::SimpleComputePipeline<shaders::PackClasConstants>
{
public:
  PackClasProgram(VkDevice device, SampleGlslCompiler& glslCompiler)
      : vkobj::SimpleComputePipeline<shaders::PackClasConstants>(device, glslCompiler, "stream_pack_clas.comp.glsl")
  {
  }
};

// A reusable GPU buffer of cluster groups indices to either load or unload.
// These are passed in a queue from the render thread to the streaming thread.
// The streaming thread will block until the data is ready.
class RequestList
{
public:
  RequestList(ResourceAllocator* allocator, uint32_t maxRequests, VkCommandPool initPool, VkQueue initQueue);
  void gather(MakeRequestsProgram&    program,
              vkobj::Buffer<uint8_t>& groupNeededFlags,
              vkobj::SemaphoreValue   promisedSubmitSemaphoreState,
              VkCommandBuffer         cmd);
  void download(ResourceAllocator* allocator, VkCommandPool pool, VkQueue queue, std::vector<shaders::GroupRequest>& result);
  size_t memoryUsage() const { return m_requests.size_bytes() + m_requestCounts.size_bytes(); }

private:
  uint32_t                                    m_maxRequests;
  vkobj::SemaphoreValue                       m_readySemaphore;  // render queue to streaming thread sync
  vkobj::Buffer<shaders::GroupRequest>        m_requests;
  vkobj::Buffer<shaders::StreamRequestCounts> m_requestCounts;  // single element
  vkobj::ReadyCommandBuffer                   m_pendingGather;  // tmp storage for submitted command buffer
};

// Moves the given object into a std::shared_ptr. This is necessary to house
// move-only objects inside an std::any. A nicer implementation would use a
// move-only version of std::any since shared_ptr is expensive.
template <class T>
std::any moveAny(T&& obj)
{
  return std::make_shared<std::remove_reference_t<T>>(std::move(obj));
}

// A reusable GPU buffer of cluster groups to insert in or remove from the
// renderable data. These are passed in a queue from the streaming thread to the
// render thread. They will not be added to the queue until ready, so no
// synchronization in that direction is needed and the render thread will not be
// interrupted.
class GroupModsList
{
public:
  GroupModsList(ResourceAllocator* allocator, uint32_t maxLoadUnloads);
  shaders::StreamGroupModsList write(ResourceAllocator*                    allocator,
                                     std::span<const shaders::LoadGroup>   loads,
                                     std::span<const shaders::UnloadGroup> unloads,
                                     std::span<const ClusteredMesh>        meshes,
                                     std::span<const uint32_t>             meshInstanceCounts,
                                     VkCommandBuffer                       cmd);

  // Called on the render thread to patch pointers (meshPointers) to cluster
  // groups. These may include writing pointers to new groups or zeroing
  // pointers to those just unloaded. May return objects to free after the
  // commands have been completed and the semaphore has been signalled.
  // NOTE: unloadGarbageSemaphore is only valid when unloadGarbage is not empty
  void modifyGroups(ModifyGroupsProgram&          program,
                    vkobj::Buffer<shaders::Mesh>& meshPointers,
                    vkobj::SemaphoreValue         promisedSubmitSemaphoreState,
                    VkCommandBuffer               cmd,
                    uint64_t&                     totalResidentClusters,  // TODO: clean up too many params
                    uint64_t&                     totalResidentInstanceClusters);

  size_t memoryUsage() const { return m_loads.size_bytes() + m_unloads.size_bytes(); }

private:
  vkobj::Buffer<shaders::LoadGroup>    m_loads;
  vkobj::Buffer<shaders::UnloadGroup>  m_unloads;
  std::optional<vkobj::SemaphoreValue> m_reuseSemaphore;  // render queue to streaming thread sync
  shaders::StreamGroupModsList         m_groupModsList;
  int32_t                              m_clusterCountDelta         = 0;
  int32_t                              m_instanceClusterCountDelta = 0;
};

enum Result : uint32_t
{
  eSuccess,       // keep loading/unloading
  eSkip,          // out of memory, skip and hopefully we get an unload
  eStopAndRetry,  // pipeline full, retry later
};

// A class to manage two queues:
// 1. Top level requests. This may fill up, and drain over time. If a
//    load/unload is no longer needed, the request may be skipped within this
//    queue. Order is not important.
// 2. Dependency expanded requests. Loads/unloads must happen in order and
//    cannot be skipped.
class RequestDependencyPipeline
{
public:
  RequestDependencyPipeline(size_t globalGroupCount)
      : m_globalGroupsExpected(globalGroupCount, false)
      , m_globalGroupsNeeded(globalGroupCount, false)
      , m_globalGroupsLoaded(globalGroupCount, false)
  {
  }

  // Insert a batch of requests from the render thread
  void queueRequests(std::span<const shaders::GroupRequest> requests);

  // Extract a single batch of loads/unloads, making sure cluster groups are
  // always loaded and unloaded in order of dependencies.
  // TODO: while an individaul RequestList cannot have a load and unload for
  // the same page, m_topLevelRequests can. It is then possible that a batch
  // is created with both a load and an unload for the same page. This could
  // result in cracks in a mesh.
  void dequeueLoadUnloadBatch(const Scene&                                     scene,
                              const std::function<Result(uint32_t, uint32_t)>& emitLoad,
                              const std::function<Result(uint32_t, uint32_t)>& emitUnload);

  uint32_t pendingRequests() const { return m_pendingRequests; }

private:
  // Depth first search. Related to topological sort
  Result loadGroupDependenciesRecursive(offset_span<offset_span<uint32_t>>               meshGroupGeneratedGroups,
                                        uint32_t                                         meshGroupOffset,
                                        uint32_t                                         meshIndex,
                                        uint32_t                                         meshGroupIndex,
                                        const std::function<Result(uint32_t, uint32_t)>& emitLoad);
  Result unloadGroupDependenciesRecursive(offset_span<offset_span<uint32_t>>               meshGroupGeneratedGroups,
                                          offset_span<offset_span<uint32_t>>               meshGroupGeneratingGroups,
                                          uint32_t                                         meshGroupOffset,
                                          uint32_t                                         meshIndex,
                                          uint32_t                                         meshGroupIndex,
                                          const std::function<Result(uint32_t, uint32_t)>& emitUnload);

  std::vector<bool> m_globalGroupsExpected;  // Allow new request to shortcut queued requests in m_topLevelRequests
  std::vector<bool> m_globalGroupsNeeded;  // Pin top level requests so that an orphaned dependency doesn't unload them
  std::vector<bool> m_globalGroupsLoaded;  // Keep track of what load events have actually been issued
  std::unordered_set<uint32_t> m_batchUnloads;  // Don't reload dependencies if they were unloaded in the same batch
  std::deque<shaders::GroupRequest> m_topLevelRequests;
  uint32_t                          m_pendingRequests = 0;
};

}  // namespace streaming

inline std::ostream& operator<<(std::ostream& os, const shaders::GroupRequest& x)
{
  os << "GroupRequest{";
  os << "globalGroup " << x.decoded.globalGroup << " ";
  os << "load " << x.decoded.load;
  os << "}";
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const shaders::LoadGroup& x)
{
  os << "LoadGroup{";
  os << "meshIndex " << x.meshIndex << " ";
  os << "groupIndex " << x.groupIndex << " ";
  os << "groupData " << x.groupData;
  os << "}";
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const shaders::UnloadGroup& x)
{
  os << "UnloadGroup{";
  os << "meshIndex " << x.meshIndex << " ";
  os << "groupIndex " << x.groupIndex;
  os << "}";
  return os;
}
