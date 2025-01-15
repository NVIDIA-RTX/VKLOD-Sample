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
#include <sample_vulkan_objects.hpp>
#include <shaders/shaders_scene.h>
#include <vulkan/vulkan_core.h>

namespace streaming {

RequestList::RequestList(ResourceAllocator* allocator, uint32_t maxRequests, VkCommandPool initPool, VkQueue initQueue)
    : m_maxRequests(maxRequests)
    , m_requests(allocator,
                 maxRequests,
                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
{
  // Initialize m_requestCounts contents
  // TODO: would be nice to consolidate these waits, but would need temp storage for the command buffer
  vkobj::ImmediateCommandBuffer cmd(allocator->getDevice(), initPool, initQueue);
  shaders::StreamRequestCounts  init{
       .requestsCount = 0,
       .requestsSize  = maxRequests,
  };
  m_requestCounts = vkobj::Buffer(allocator, std::vector{init},
                                  VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, cmd);
  memoryBarrier(cmd, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
}

void RequestList::gather(MakeRequestsProgram&    program,
                         vkobj::Buffer<uint8_t>& groupNeededFlags,
                         vkobj::SemaphoreValue   promisedSubmitSemaphoreState,
                         VkCommandBuffer         cmd)
{
  shaders::BuildRequestsConstants constants{
      .groupNeededFlagsAddress    = groupNeededFlags.address(),
      .streamRequestCountsAddress = m_requestCounts.address(),
      .requestsAddress            = m_requests.address(),
      .groupCount                 = uint32_t(groupNeededFlags.size()),
  };
  memoryBarrier(cmd, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  vkCmdPushConstants(cmd, program.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants), &constants);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, program.pipeline);
  vkCmdDispatch(cmd, div_ceil(uint32_t(groupNeededFlags.size()), uint32_t(STREAM_WORKGROUP_SIZE)), 1, 1);
  memoryBarrier(cmd, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  m_readySemaphore = promisedSubmitSemaphoreState;
}

void RequestList::download(ResourceAllocator* allocator, VkCommandPool pool, VkQueue queue, std::vector<shaders::GroupRequest>& result)
{
  result.clear();
  nvvk::StagingMemoryManager* smm = allocator->getStaging();

  // First download the request list count
  const uint32_t* countPtr;
  {
    vkobj::BuildingCommandBuffer cmd(allocator->getDevice(), pool, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    countPtr = smm->cmdFromBufferT<uint32_t>(cmd, m_requestCounts, offsetof(shaders::StreamRequestCounts, requestsCount),
                                             sizeof(shaders::StreamRequestCounts::requestsCount));
    memoryBarrier(cmd, VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                  VK_PIPELINE_STAGE_TRANSFER_BIT);
    vkCmdFillBuffer(cmd, m_requestCounts, offsetof(shaders::StreamRequestCounts, requestsCount),
                    sizeof(shaders::StreamRequestCounts::requestsCount), 0);  // zero the count for buffer reuse
    vkobj::ReadyCommandBuffer recordedCmd(std::move(cmd));
    std::array                waits{m_readySemaphore.submitInfo(VK_PIPELINE_STAGE_2_TRANSFER_BIT)};
    recordedCmd.submitAfter(queue, waits);
    NVVK_CHECK(vkQueueWaitIdle(queue));
  }

  // requestsCount may overshoot due to parallel atomics
  uint32_t count = std::min(m_maxRequests, *countPtr);

  // Then download the data, if there is any
  if(count)
  {
    std::span<const shaders::GroupRequest> requests;
    {
      vkobj::ImmediateCommandBuffer cmd(allocator->getDevice(), pool, queue);
      requests = {smm->cmdFromBufferT<shaders::GroupRequest>(cmd, m_requests, 0, sizeof(shaders::GroupRequest) * count), count};
    };
    result.resize(count);
    std::ranges::copy(requests, result.begin());
  }
#if 0
  rangeSummary(std::cerr << "Requests: ", result) << "\n";
#endif
}

GroupModsList::GroupModsList(ResourceAllocator* allocator, uint32_t maxLoadUnloads)
    : m_loads(allocator,
              maxLoadUnloads,
              VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
    , m_unloads(allocator,
                maxLoadUnloads,
                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
{
  // Suballocate request list
  m_groupModsList.loadGroupsAddress   = m_loads.address();
  m_groupModsList.unloadGroupsAddress = m_unloads.address();
}

shaders::StreamGroupModsList GroupModsList::write(ResourceAllocator*                    allocator,
                                                  std::span<const shaders::LoadGroup>   loads,
                                                  std::span<const shaders::UnloadGroup> unloads,
                                                  std::span<const ClusteredMesh>        meshes,
                                                  std::span<const uint32_t>             meshInstanceCounts,
                                                  VkCommandBuffer                       cmd)
{
  // Make sure this object is not still being read from by the GPU after a
  // previous call to modifyGroups()
  if(m_reuseSemaphore)
    m_reuseSemaphore->wait(allocator->getDevice());

  // Write the counts, which will be used to dispatch compute threads
  m_groupModsList.loadGroupCount   = uint32_t(loads.size());
  m_groupModsList.unloadGroupCount = uint32_t(unloads.size());

  // Write the LoadGroup and UnloadGroup arrays that contain references to
  // streamed data that must be added or removed
  nvvk::StagingMemoryManager* smm = allocator->getStaging();
  cmdToArray(*smm, cmd, loads, m_loads);
  cmdToArray(*smm, cmd, unloads, m_unloads);

  // Record running totals of what was loaded and unloaded. This may be used to
  // determine worst case allocations needed
  m_clusterCountDelta         = 0;
  m_instanceClusterCountDelta = 0;
  for(const shaders::LoadGroup& load : loads)
  {
    assert(load.groupData.clusterCount == meshes[load.meshIndex].groupClusterRanges[load.groupIndex].count);
    m_clusterCountDelta += load.groupData.clusterCount;
    m_instanceClusterCountDelta += load.groupData.clusterCount * meshInstanceCounts[load.meshIndex];
  }
  for(const shaders::UnloadGroup& unload : unloads)
  {
    uint32_t groupClusterCount = meshes[unload.meshIndex].groupClusterRanges[unload.groupIndex].count;
    m_clusterCountDelta -= groupClusterCount;
    m_instanceClusterCountDelta -= groupClusterCount * meshInstanceCounts[unload.meshIndex];
    assert(unload.groupIndex != uint32_t(meshes[unload.meshIndex].groupGeneratedGroups.size() - 1));  // should never unload root pages
  }

  // Debugging - check there are no duplicates between loads/unloads
#if !defined(NDEBUG)
  std::set<std::pair<uint32_t, uint32_t>> loadids;
  std::set<std::pair<uint32_t, uint32_t>> unloadids;
  for(const shaders::LoadGroup& load : loads)
    loadids.insert({load.meshIndex, load.groupIndex});
  for(const shaders::UnloadGroup& unload : unloads)
    unloadids.insert({unload.meshIndex, unload.groupIndex});
  for(const shaders::LoadGroup& load : loads)
    assert(unloadids.count({load.meshIndex, load.groupIndex}) == 0);
  for(const shaders::UnloadGroup& unload : unloads)
    assert(loadids.count({unload.meshIndex, unload.groupIndex}) == 0);
#endif

  // Returns the uploaded structures, which contain cluster geometry and can be used
  // to fill CLAS build input structures
  return m_groupModsList;
}

void GroupModsList::modifyGroups(ModifyGroupsProgram&          program,
                                 vkobj::Buffer<shaders::Mesh>& meshPointers,
                                 vkobj::SemaphoreValue         promisedSubmitSemaphoreState,
                                 VkCommandBuffer               cmd,
                                 uint64_t&                     totalResidentClusters,
                                 uint64_t&                     totalResidentInstanceClusters)
{
  // Update running totals
  totalResidentClusters += m_clusterCountDelta;
  totalResidentInstanceClusters += m_instanceClusterCountDelta;

  // Launch compute kernels to process loads and unloads
  memoryBarrier(cmd, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  if(m_groupModsList.loadGroupCount)
  {
    shaders::StreamGroupsConstants loadConstants{
        .meshesAddress = meshPointers.address(),
        .mods          = m_groupModsList,
        .load          = 1u,
    };
    vkCmdPushConstants(cmd, program.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(loadConstants), &loadConstants);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, program.pipeline);
    vkCmdDispatch(cmd, div_ceil(m_groupModsList.loadGroupCount, uint32_t(STREAM_WORKGROUP_SIZE)), 1, 1);
  }
  if(m_groupModsList.unloadGroupCount)
  {
    shaders::StreamGroupsConstants unloadConstants{
        .meshesAddress = meshPointers.address(),
        .mods          = m_groupModsList,
        .load          = 0,
    };
    vkCmdPushConstants(cmd, program.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(unloadConstants), &unloadConstants);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, program.pipeline);
    vkCmdDispatch(cmd, div_ceil(m_groupModsList.unloadGroupCount, uint32_t(STREAM_WORKGROUP_SIZE)), 1, 1);
  }
  memoryBarrier(cmd, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

  // Record the command buffer's semaphore so this object doesn't get written to
  // before the above compute shaders are finished
  m_reuseSemaphore = promisedSubmitSemaphoreState;
}

void RequestDependencyPipeline::queueRequests(std::span<const shaders::GroupRequest> requests)
{
  // Keep track of the most recent state at the back of the queue so short
  // pulses can be ignored
  for(const shaders::GroupRequest& request : requests)
  {
    m_globalGroupsExpected[request.decoded.globalGroup] = bool(request.decoded.load);
    if(m_globalGroupsNeeded[request.decoded.globalGroup] != bool(request.decoded.load))
      m_pendingRequests++;
    else
      m_pendingRequests--;
  }

  // Add all the requests to the main queue. This is allowed to grow
  // indefinitely
  m_topLevelRequests.insert(m_topLevelRequests.end(), requests.begin(), requests.end());
}

void RequestDependencyPipeline::dequeueLoadUnloadBatch(const Scene&                                     scene,
                                                       const std::function<Result(uint32_t, uint32_t)>& emitLoad,
                                                       const std::function<Result(uint32_t, uint32_t)>& emitUnload)
{
  m_batchUnloads.clear();
  while(!m_topLevelRequests.empty())
  {
    // Get the next top level request, but don't consume it just yet
    shaders::GroupRequest request = m_topLevelRequests.front();

    // Skip processing requests made outdated by newer ones from
    // queueRequests().
    if(m_globalGroupsExpected[request.decoded.globalGroup] != bool(request.decoded.load))
    {
      m_topLevelRequests.pop_front();
      // m_pendingRequests should already be adjusted for this
      continue;
    }

    // With the above filter we now also need to ignore inevitable duplicate
    // requests that are still in the queue
    if(m_globalGroupsNeeded[request.decoded.globalGroup] == bool(request.decoded.load))
    {
      m_topLevelRequests.pop_front();
      // m_pendingRequests should already be adjusted for this
      continue;
    }

    // TODO: avoid upper_bound binary search
    auto     offsetIt        = std::ranges::upper_bound(scene.meshGroupOffsets, request.decoded.globalGroup) - 1;
    uint32_t meshGroupOffset = *offsetIt;
    uint32_t meshIndex       = uint32_t(offsetIt - scene.meshGroupOffsets.begin());
    uint32_t meshGroupIndex  = request.decoded.globalGroup - *offsetIt;
    Result   result;
    if(request.decoded.load)
    {
      result = loadGroupDependenciesRecursive(scene.meshes[meshIndex].groupGeneratedGroups, meshGroupOffset, meshIndex,
                                              meshGroupIndex, emitLoad);

      // Pin the page to prevent automatically unloading it due to a lost
      // dependency. This is still done even for Result::eSkip.
      m_globalGroupsNeeded[request.decoded.globalGroup] = true;
    }
    else
    {
      // Un-pin this group as needed to allow it and dependencies to unload
      m_globalGroupsNeeded[request.decoded.globalGroup] = false;
      result = unloadGroupDependenciesRecursive(scene.meshes[meshIndex].groupGeneratedGroups, scene.meshes[meshIndex].groupGeneratingGroups,
                                                meshGroupOffset, meshIndex, meshGroupIndex, emitUnload);
    }

    if(result == Result::eSuccess || result == Result::eSkip)
    {
      m_topLevelRequests.pop_front();
      --m_pendingRequests;
    }
    else
    {
      assert(result == Result::eStopAndRetry);

      // Reset m_globalGroupsNeeded so the retry passes the initial filter
      m_globalGroupsNeeded[request.decoded.globalGroup] = !bool(request.decoded.load);
      break;
    }
  }
}

Result RequestDependencyPipeline::loadGroupDependenciesRecursive(offset_span<offset_span<uint32_t>> meshGroupGeneratedGroups,
                                                                 uint32_t meshGroupOffset,
                                                                 uint32_t meshIndex,
                                                                 uint32_t meshGroupIndex,
                                                                 const std::function<Result(uint32_t, uint32_t)>& emitLoad)
{
  Result result = Result::eSuccess;

  // Nothing to do if the group is already loaded
  if(m_globalGroupsLoaded[meshGroupOffset + meshGroupIndex])
    return result;

  // Load the group dependencies first to guarantee dependency order.
  for(uint32_t dependency : meshGroupGeneratedGroups[meshGroupIndex])
  {
    result = loadGroupDependenciesRecursive(meshGroupGeneratedGroups, meshGroupOffset, meshIndex, dependency, emitLoad);
    if(result != Result::eSuccess)
      return result;
  }

  // Don't emit a load if just unloaded in this batch. Top level pulse requests
  // like this are filtered out by m_globalGroupsExpected, but a dependency can
  // be unloaded and immediately reloaded by another top level request. The
  // reverse cannot happen as top level requests are pinned.
  if(m_batchUnloads.count(meshGroupOffset + meshGroupIndex) > 0)
    return Result::eStopAndRetry;

  // Emit load op for this group
  result = emitLoad(meshIndex, meshGroupIndex);
  if(result == Result::eSuccess)
    m_globalGroupsLoaded[meshGroupOffset + meshGroupIndex] = true;
  return result;
}

Result RequestDependencyPipeline::unloadGroupDependenciesRecursive(offset_span<offset_span<uint32_t>> meshGroupGeneratedGroups,
                                                                   offset_span<offset_span<uint32_t>> meshGroupGeneratingGroups,
                                                                   uint32_t meshGroupOffset,
                                                                   uint32_t meshIndex,
                                                                   uint32_t meshGroupIndex,
                                                                   const std::function<Result(uint32_t, uint32_t)>& emitUnload)
{
  Result result = Result::eSuccess;

  // Cannot implicitly unload directly requested group
  if(m_globalGroupsNeeded[meshGroupOffset + meshGroupIndex])
    return result;

  // Abort if something depends on this group
  for(auto& groupDependentOnThis : meshGroupGeneratingGroups[meshGroupIndex])
  {
    if(m_globalGroupsLoaded[meshGroupOffset + groupDependentOnThis])
      return result;
  }

  // Emit unload op for this group if it is currently loaded
  if(m_globalGroupsLoaded[meshGroupOffset + meshGroupIndex])
    result = emitUnload(meshIndex, meshGroupIndex);

  // Search for orphaned dependencies and unload them even if this group was
  // already unloaded. This is to support Result::eStopAndRetry.
  if(result == Result::eSuccess)
  {
    m_globalGroupsLoaded[meshGroupOffset + meshGroupIndex] = false;
    m_batchUnloads.insert(meshGroupOffset + meshGroupIndex);

    // Check if any dependencies can now be unloaded. Recurse after to guarantee
    // dependency order.
    for(uint32_t dependency : meshGroupGeneratedGroups[meshGroupIndex])
    {
      result = unloadGroupDependenciesRecursive(meshGroupGeneratedGroups, meshGroupGeneratingGroups, meshGroupOffset,
                                                meshIndex, dependency, emitUnload);
      if(result != Result::eSuccess)
        return result;
    }
  }

  return result;
}

}  // namespace streaming
