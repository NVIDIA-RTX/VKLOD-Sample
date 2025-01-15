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

#include <lod_streaming_jobs.hpp>
#include <lod_streaming_scene.hpp>
#include <nvh/alignment.hpp>
#include <sample_vulkan_objects.hpp>
#include <vulkan/vulkan_core.h>

namespace streaming {

// Common call to facilitate both sync and async streaming
// TODO: inline - sync streaming was removed
static Result makeLoad(ResourceAllocator* allocator,
                       vkobj::ByteBuffer& memoryPoolBuffer,
                       PoolAllocator&     memoryPool,
                       const Scene&       scene,
                       uint32_t           meshIndex,
                       uint32_t           groupIndex,
                       uint32_t           maxLoads,
                       uint32_t           maxClustersPerBuild,
                       LoadUnloadBatch&   batch,
                       VkCommandBuffer    cmd)
{
  // Simple out-of-memory case, where we're above a high water mark
  // of 80% memory usage. This saves having to estimate and reserve
  // additional CLAS memory that will be needed to complete the
  // load.
  // TODO: estimate and reserve CLAS allocations to avoid hard coded 80%
  // TODO: this can deadlock on launch if there is not enough memory for all
  // root groups
  if(memoryPool.bytesAllocated() + memoryPool.internalFragmentation() > (memoryPool.size() * 80) / 100)
  {
    return Result::eSkip;
  }

  // Abort if adding this the next job would exceed maxClusters
  uint32_t groupClusterCount = scene.meshes[meshIndex].groupClusterRanges[groupIndex].count;
  if(batch.loads.size() + 1 > maxLoads || batch.totalClusters + groupClusterCount > maxClustersPerBuild)
  {
    assert(batch.totalClusters > 0);  // Check if even a single load would load too many clusters
    return Result::eStopAndRetry;
  }

  batch.newGeomertries.push_back(
      ClusterGroupGeometryVk(allocator, memoryPoolBuffer, memoryPool, scene.meshes[meshIndex], groupIndex, cmd));
  ClusterGroupGeometryVk& geometry = batch.newGeomertries.back();

  // Prepare pointers to the new geometry. This doubles as input for
  // building the CLAS and buffers to pass to the render thread to fetch
  // geometry when raytracing.
  // NOTE: the addresses array shaders::ClusterGroup::clasAddressesAddress
  // points to is not populated yet!
  uint32_t loadGroupIndex = uint32_t(batch.loads.size());
  batch.loadClusterLoadGroupsHost.insert(batch.loadClusterLoadGroupsHost.end(), groupClusterCount, loadGroupIndex);
  batch.loadGroupClusterOffsetsHost.push_back(batch.totalClusters);
  batch.totalClusters += groupClusterCount;
  batch.loads.push_back(shaders::LoadGroup{
      .groupData =
          shaders::ClusterGroup{
              .clusterGeometryAddressesAddress = geometry.clusterGeometryAddressesAddress(),
              .clusterGeneratingGroupsAddress  = geometry.clusterGeneratingGroupsAddress(),
              .clasAddressesAddress            = geometry.clasAddressesAddress(),
              .clusterCount                    = groupClusterCount,
              .padding_                        = 0,
          },
      .meshIndex  = meshIndex,
      .groupIndex = groupIndex,
  });
  assert(batch.loads.back().groupData.clusterGeometryAddressesAddress);
  assert(batch.loads.back().groupData.clusterGeneratingGroupsAddress);
  assert(batch.loads.back().groupData.clasAddressesAddress);
  assert(batch.loads.back().groupData.clusterCount);
  return Result::eSuccess;
}

inline VkBuildAccelerationStructureFlagBitsKHR defaultClasBuildFlags()
{
  // alt: VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR
  return VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
}

BatchWithBuiltCLAS::BatchWithBuiltCLAS(ResourceAllocator* allocator,
                                       const Scene&       scene,
                                       uint32_t           maxLoadUnloads,
                                       uint32_t           maxGroupsPerBuild,
                                       uint32_t           maxClustersPerBuild,
                                       uint32_t           positionTruncateBits)
    : batch(allocator, maxLoadUnloads)
    , clasStaging(allocator, scene, maxGroupsPerBuild, maxClustersPerBuild, positionTruncateBits, defaultClasBuildFlags())
{
}

StreamingSceneVk::StreamingSceneVk(ResourceAllocator* allocator, SampleGlslCompiler &glslCompiler, VkDeviceSize geometryBufferBytes, VkCommandPool initPool, vkobj::TimelineQueue& initQueue, const Scene& scene, SceneVK& sceneVk, bool requiresClas, uint32_t streamingTransferQueueFamilyIndex, VkQueue streamingTransferQueue)
      : m_geometryLoaderContext(allocator->getDevice(), allocator->getPhysicalDevice(), allocator->getMemoryAllocator(), streamingTransferQueueFamilyIndex, streamingTransferQueue)
      , m_maxClustersPerBuild(MaxGroupsPerBuild * scene.counts.maxClustersPerGroup)
      , m_requiresClas(requiresClas)
      , m_memoryPoolBuffer(allocator, 
        geometryBufferBytes,
         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
      , m_memoryPool(m_memoryPoolBuffer.address(), geometryBufferBytes)
      , m_pipeline0Requests{
          RequestList{allocator, MaxRequests, initPool, initQueue.queue},
          RequestList{allocator, MaxRequests, initPool, initQueue.queue},
          RequestList{allocator, MaxRequests, initPool, initQueue.queue},
      }
      , m_pipeline1Geometry{
          LoadUnloadBatch{allocator, MaxLoadUnloads},
          LoadUnloadBatch{allocator, MaxLoadUnloads},
      }
      , m_pipeline2GeometryCLAS{
          BatchWithBuiltCLAS{allocator, scene, MaxLoadUnloads, MaxGroupsPerBuild, m_maxClustersPerBuild, PositionTruncateBits},
          BatchWithBuiltCLAS{allocator, scene, MaxLoadUnloads, MaxGroupsPerBuild, m_maxClustersPerBuild, PositionTruncateBits},
      }
      , m_requestsProgram(allocator->getDevice(), glslCompiler)
      , m_modifyGroupsProgram(allocator->getDevice(), glslCompiler)
      , m_fillClasInputProgram(allocator->getDevice(), glslCompiler)
      , m_packClasProgram(allocator->getDevice(), glslCompiler)
{
  for(const RequestList& item : m_pipeline0Requests.storage())
    m_staticStagingMemory += item.memoryUsage();
  for(const LoadUnloadBatch& item : m_pipeline1Geometry.storage())
    m_staticStagingMemory += item.memoryUsage();
  for(const BatchWithBuiltCLAS& item : m_pipeline2GeometryCLAS.storage())
    m_staticStagingMemory += item.memoryUsage();

  // Reset all streaming data so that this object can recreated while the
  // SceneVK persists
  {
    vkobj::ImmediateCommandBuffer cmd(allocator->getDevice(), initPool, initQueue.queue);
    sceneVk.cmdResetStreaming(allocator, scene, cmd);
  }

  // Start the background thread
  m_geometryLoaderThread = std::jthread(&StreamingSceneVk::geometryLoaderEntrypoint, this, std::make_unique<const Scene>(scene));

  // Stream in the root pages
  flush(allocator, initPool, initQueue, scene, sceneVk.allGroupNeededFlags, sceneVk.meshPointers,
        sceneVk.totalResidentClusters, sceneVk.totalResidentInstanceClusters);
}

StreamingSceneVk::~StreamingSceneVk()
{
  m_running = false;

  // Unblock any producer/consumer waits
  m_pipeline0Requests.cancel();
  m_pipeline1Geometry.cancel();
  m_pipeline2GeometryCLAS.cancel();
}

// Called on the render thread
void StreamingSceneVk::makeRequests(vkobj::Buffer<uint8_t>& groupNeededFlags,
                                    vkobj::SemaphoreValue   promisedSubmitSemaphoreState,
                                    VkCommandBuffer         cmd)
{
  m_pipeline0Requests.tryProduce([&](streaming::RequestList& requests) {
    requests.gather(m_requestsProgram, groupNeededFlags, promisedSubmitSemaphoreState, cmd);
    m_workCounter.addRequestWork(1);
  });
}

void StreamingSceneVk::buildClasBatch(ResourceAllocator* allocator, vkobj::SemaphoreValue promisedSubmitSemaphoreState, bool block, VkCommandBuffer cmd)
{
  m_pipeline1Geometry.maybeConsume(block, [&](LoadUnloadBatch& in) -> bool {
    // It is possible that an empty batch is produced if there is no more
    // memory.
    if(in.loads.empty() && in.unloads.empty())
      return true;

    // Build the batch of CLASes into a staging buffer on the render thread.
    // They are then compacted, but we first need to allocate the final block of
    // memory, so we wait for the build to complete in a background thread
    // before allocating and compacting.
    return m_pipeline2GeometryCLAS.tryProduce([&](BatchWithBuiltCLAS& out) {
      if(in.totalClusters > 0 && m_requiresClas)
      {
        out.clasStaging.buildClas(allocator, m_fillClasInputProgram, m_packClasProgram, in.uploadedMods,
                                  in.loadClusterLoadGroupsHost, in.loadGroupClusterOffsetsHost, in.totalClusters, cmd);
      }
      out.clasBuildDone = promisedSubmitSemaphoreState;
      std::swap(in, out.batch);  // cycle objects between intermediate queues
    });
  });
}

// Called on the render thread
bool StreamingSceneVk::modifyGroups(ResourceAllocator*            allocator,
                                    const Scene&                  scene,
                                    vkobj::Buffer<shaders::Mesh>& meshPointers,
                                    std::vector<ClusterGroupVk>&  unloadGarbage,
                                    vkobj::SemaphoreValue         unloadGarbageSemaphore,
                                    bool                          block,
                                    VkCommandBuffer               cmd,
                                    uint64_t& totalResidentClusters,  // TODO: clean up too many params
                                    uint64_t& totalResidentInstanceClusters)
{
  return m_pipeline2GeometryCLAS.maybeConsume(block, [&](BatchWithBuiltCLAS& in) -> bool {
    m_workCounter.addResult(-1);

    // Compact the built CLAS
    if(in.batch.totalClusters > 0 && m_requiresClas)
    {
      if(!in.clasBuildDone.wait(m_geometryLoaderContext.device, 0))
        return false;  // This batch is not ready yet. Try again next frame.
      in.clasStaging.compactClas(allocator, m_memoryPool, in.clasBuildDone, m_packClasProgram, in.batch.uploadedMods,
                                 in.batch.totalClusters, cmd, in.newClases);
      assert(in.batch.newGeomertries.size() == in.newClases.size());
    }

    // Keep track of streamed allocations until we unload them and move any
    // unloaded allocations into unloadGarbage.
    loadUnloadGroupAllocations(scene, in.batch, in.newClases, unloadGarbage);

    // Insert new geometry and CLAS pointers into the scene state on the GPU and
    // remove unloaded ones.
    in.batch.groupModsList.modifyGroups(m_modifyGroupsProgram, meshPointers, unloadGarbageSemaphore, cmd,
                                        totalResidentClusters, totalResidentInstanceClusters);
    return true;
  });
}

void StreamingSceneVk::flush(ResourceAllocator*            allocator,
                             VkCommandPool                 pool,
                             vkobj::TimelineQueue&         queue,
                             const Scene&                  scene,
                             vkobj::Buffer<uint8_t>&       groupNeededFlags,
                             vkobj::Buffer<shaders::Mesh>& meshPointers,
                             uint64_t&                     totalResidentClusters,  // TODO: clean up too many params
                             uint64_t&                     totalResidentInstanceClusters)
{
  // Loop until there are no more changes from streaming requests. This is
  // particularly important for uploading initial root cluster groups so each
  // instance has something to render, even if low detail. This may take
  // multiple iterations if all requests don't fit in one
  // streaming::RequestList.
  for(;;)
  {
    // Inject top level requests.
    {
      vkobj::ImmediateCommandBuffer cmd(allocator->getDevice(), pool, queue, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
      makeRequests(groupNeededFlags, queue.nextSubmitValue(), cmd);
    }

    // Wait for modify groups. If the requests produced no results we're done.
    // This is particularly difficult to capture since makeRequests() always
    // creates an object but we won't know if it will result in any groups being
    // modified until requests (if any) are expanded to dependencies. We also
    // need to keep calling modifyGroups() due to the limited number of
    // intermediate buffers.
    if(!m_workCounter.waitForResult())
      break;

    if(m_requiresClas)
    {
      vkobj::ImmediateCommandBuffer cmd(allocator->getDevice(), pool, queue,
                                        VK_PIPELINE_STAGE_2_TRANSFER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                                            | VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_NV);
      buildClasBatch(allocator, queue.nextSubmitValue(), true, cmd);
    }

    // Update pointers to geometry and CLAS
    {
      vkobj::ImmediateCommandBuffer cmd(allocator->getDevice(), pool, queue,
                                        VK_PIPELINE_STAGE_2_TRANSFER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                                            | VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_NV);
      std::vector<ClusterGroupVk>   unloadGarbage;
      vkobj::SemaphoreValue unloadGarbageSemaphore = queue.nextSubmitValue();  // don't reuse GPU buffers until this is signalled
      modifyGroups(allocator, scene, meshPointers, unloadGarbage, unloadGarbageSemaphore, true, cmd,
                   totalResidentClusters, totalResidentInstanceClusters);
      assert(unloadGarbage.empty());  // should only be loading new groups
    }
  }

  // There should be no more work left in the streaming pipeline. These checks
  // are not made atomically and thus are not thread safe but it's better than
  // nothing for a sanity check
  assert(m_pipeline0Requests.promisedEmpty() && m_pipeline1Geometry.promisedEmpty() && m_pipeline2GeometryCLAS.promisedEmpty());

#if 0
  BufferDownloader downloader(initContext.device, 0 /* DANGER: hard coded */, initContext.allocator->getStaging());
  rangeSummaryVk(std::cerr << "Meshes: ", meshPointers) << "\n";
#endif
}

void StreamingSceneVk::geometryLoaderEntrypoint(std::unique_ptr<const Scene> scene)
{
  // Keep staging memory around, otherwise continuous allocations will block the
  // render thread and cause a lot of stutter.
  m_geometryLoaderContext.allocator->getStaging()->setFreeUnusedOnRelease(false);

  // Temporary storage to pull from
  std::vector<shaders::GroupRequest> topLevelGroupRequests;

  // Top level request queue. This provides two features:
  // 1. filtering requests that cancel (e.g. a quick load+unload)
  // 2. maintaining group dependencies
  int32_t                              requestsConsumed   = 0;
  bool                                 requestsQueueEmpty = true;
  streaming::RequestDependencyPipeline requestDependenciesPipeline(scene->meshGroupOffsets.back());
  while(m_running)
  {
    // Fetch streaming requests from the render thread. Wait for them if there
    // is no more streaming work to be done, otherwise process one batch of the
    // remaining work in the queue.
    bool waitForRequests = (m_pendingRequests == 0);
    m_pipeline0Requests.consume(waitForRequests, [&](streaming::RequestList& requests) {
      ++requestsConsumed;  // delay marking work as done until subsequent work is tracked
      requests.download(m_geometryLoaderContext.allocator, m_geometryLoaderContext.commandPool,
                        m_geometryLoaderContext.queue, topLevelGroupRequests);
      requestDependenciesPipeline.queueRequests(topLevelGroupRequests);
      m_pendingRequests = requestDependenciesPipeline.pendingRequests();
    });

    // Artificially increment m_workCounter while there is work in the queue.
    // This facilitates StreamingSceneVk::flush() checking for completion.
    if(m_pendingRequests != 0 && requestsQueueEmpty)
    {
      requestsQueueEmpty = false;
      m_workCounter.addRequestWork(1);
    }

    // Extract and load geometry for one batch of group modifications from the
    // request pipeline. If the next stage of the pipeline is full, keep looping.
    // It's more important to clear m_pipeline0Requests from the render thread.
    // TODO: this could result in spinning/polling; ideally we could wait on
    // both m_pipeline0Requests or m_pipeline1Geometry.
    if(m_pendingRequests != 0)
    {
      if(m_requiresClas)
      {
        m_pipeline1Geometry.tryProduce([&](LoadUnloadBatch& batch) {
          m_workCounter.addResult(1);
          batch.clear();  // object reuse
          loadGeometryBatch(*scene, requestDependenciesPipeline, batch);
        });
      }
      else
      {
        // HACK: rasterization path. Using BatchWithBuiltCLAS, allocations and
        // all, even though it's not initialized with any CLAS
        m_pipeline2GeometryCLAS.tryProduce([&](BatchWithBuiltCLAS& out) {
          m_workCounter.addResult(1);
          out.batch.clear();  // object reuse
          loadGeometryBatch(*scene, requestDependenciesPipeline, out.batch);
        });
      }
    }

    // Decrement m_workCounter once the queue becomes empty again
    if(m_pendingRequests == 0 && !requestsQueueEmpty)
    {
      requestsQueueEmpty = true;
      m_workCounter.addRequestWork(-1);
    }

    m_workCounter.addRequestWork(-requestsConsumed);  // Mark work as complete only after producing
    requestsConsumed = 0;

    // Mark any allocated staging buffers for reuse
    VkDeviceSize allocatedSize, usedSize;
    m_geometryLoaderContext.allocator->getStaging()->getUtilization(allocatedSize, usedSize);
    m_geometryLoaderMemory = allocatedSize;
    m_geometryLoaderContext.allocator->finalizeAndReleaseStaging();
  }
}

void StreamingSceneVk::loadGeometryBatch(const Scene& scene, streaming::RequestDependencyPipeline& requests, LoadUnloadBatch& batch)
{
  vkobj::BuildingCommandBuffer cmd(m_geometryLoaderContext.device, m_geometryLoaderContext.commandPool,
                                   VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

  // Callback to load a cluster group. A callback is easier as we can abort the
  // batch at any time, e.g. when the batch is full or when out of memory.
  auto load = [this, &scene, batchPtr = &batch, cmdPtr = &cmd](uint32_t meshIndex, uint32_t groupIndex) -> Result {
    auto& batch = *batchPtr;
    auto& cmd   = *cmdPtr;
    return makeLoad(m_geometryLoaderContext.allocator, m_memoryPoolBuffer, m_memoryPool, scene, meshIndex, groupIndex,
                    MaxLoadUnloads, m_maxClustersPerBuild, batch, cmd);
  };

  // Callback for unloading a cluster group.
  auto unload = [&batch](uint32_t meshIndex, uint32_t groupIndex) -> Result {
    if(batch.unloads.size() + 1 > MaxLoadUnloads)
      return Result::eStopAndRetry;
    batch.unloads.push_back(shaders::UnloadGroup{
        .meshIndex  = meshIndex,
        .groupIndex = groupIndex,
    });
    return Result::eSuccess;
  };

  // Create one batch of loads/unloads, expanding dependencies as needed
  requests.dequeueLoadUnloadBatch(scene, load, unload);
  m_pendingRequests = requests.pendingRequests();  // Update the remaining requests
  assert(batch.loads.size() <= MaxLoadUnloads);
  assert(batch.unloads.size() <= MaxLoadUnloads);
  if(!batch.loads.empty() || !batch.unloads.empty())
  {
    // Upload per-group geometry pointers, which are passed back to the
    // render thread
    batch.uploadedMods = batch.groupModsList.write(m_geometryLoaderContext.allocator, batch.loads, batch.unloads,
                                                   scene.meshes, scene.meshInstanceCounts, cmd);
  }

  // Submit and wait for completion. The next consumer is the render thread,
  // to build CLASes. We could pass it a semaphore but we don't want to block
  // it so waiting here is best.
  vkobj::ReadyCommandBuffer transferCmd(std::move(cmd));
  transferCmd.submit(m_geometryLoaderContext.queue);
  vkQueueWaitIdle(m_geometryLoaderContext.queue);

#if 0
  BufferDownloader downloader(context.device, 1 /* DANGER: hard coded */,
                              context.allocator->getStaging());
  rangeSummary(std::cerr << "loads: ", batch.loads) << "\n";
#endif
}

void StreamingSceneVk::loadUnloadGroupAllocations(const Scene&                 scene,
                                                  LoadUnloadBatch&             batch,
                                                  std::vector<PoolMemory>&     newClases,
                                                  std::vector<ClusterGroupVk>& unloadGarbage)
{
  // Extract unloaded groups so the memory is released after they are
  // extracted from the rendering data structures
  for(shaders::UnloadGroup& unloadGroup : batch.unloads)
  {
    uint32_t globalIndex = scene.meshGroupOffsets[unloadGroup.meshIndex] + unloadGroup.groupIndex;
    assert(m_groups.count(globalIndex) == 1);
    unloadGarbage.emplace_back(std::move(m_groups.extract(globalIndex).mapped()));
  }

  // Store the newGeomertries and newClases allocations in m_groups until
  // they get unloaded.
  for(size_t loadIndex = 0; loadIndex < batch.loads.size(); ++loadIndex)
  {
    shaders::LoadGroup& loadGroup   = batch.loads[loadIndex];
    uint32_t            globalIndex = scene.meshGroupOffsets[loadGroup.meshIndex] + loadGroup.groupIndex;
    static_assert(std::is_move_constructible_v<ClusterGroupVk>);
    static_assert(std::is_move_assignable_v<ClusterGroupVk>);

    // Branch for ray tracing with CLASes built during streaming or
    // rasterization that doesn't need them
    if(m_requiresClas)
    {
      m_groups.emplace(globalIndex, ClusterGroupVk{std::move(batch.newGeomertries[loadIndex]), std::move(newClases[loadIndex])});
    }
    else
    {
      m_groups.emplace(globalIndex, ClusterGroupVk{std::move(batch.newGeomertries[loadIndex])});
    }
  }
  newClases.clear();
}

}  // namespace streaming
