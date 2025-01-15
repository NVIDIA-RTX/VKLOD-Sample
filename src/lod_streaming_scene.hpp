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
#include <condition_variable>
#include <lod_streaming_jobs.hpp>
#include <mutex>
#include <sample_allocation.hpp>
#include <sample_glsl_compiler.hpp>
#include <sample_vulkan_objects.hpp>
#include <shaders/shaders_scene.h>
#include <shaders/shaders_stream.h>
#include <thread>

namespace streaming {

// Internal job for the queue between the geometry loader and CLAS builder. This
// object is built over time and its allocations are reused.
struct LoadUnloadBatch
{
  LoadUnloadBatch(ResourceAllocator* allocator, uint32_t maxLoadUnloads)
      : groupModsList(allocator, maxLoadUnloads)
  {
  }
  size_t memoryUsage() const { return groupModsList.memoryUsage(); }

  // Initialized over time (delayed CLAS addresses written and unload garbage)
  // TODO: the mods list allocations are kept through the pipeline to increase
  // streaming bandwidth, but this would be good to optimize away
  streaming::GroupModsList     groupModsList;
  shaders::StreamGroupModsList uploadedMods;

  uint32_t                            totalClusters = 0;
  std::vector<shaders::LoadGroup>     loads;
  std::vector<shaders::UnloadGroup>   unloads;
  std::vector<ClusterGroupGeometryVk> newGeomertries;

  // for per-cluster shader threads to find their load group index
  std::vector<uint32_t> loadClusterLoadGroupsHost;
  std::vector<uint32_t> loadGroupClusterOffsetsHost;

  // Reusing objects is bad, but reusing the std::vector allocations will save
  // perf
  void clear()
  {
    totalClusters = 0;
    loads.clear();
    unloads.clear();
    newGeomertries.clear();
    loadClusterLoadGroupsHost.clear();
    loadGroupClusterOffsetsHost.clear();
  }
};

// Queue object to hold geometry buffers and CLAS before CLAS compaction and
// final insertion into the scene's geometry pointers
struct BatchWithBuiltCLAS
{
  BatchWithBuiltCLAS(ResourceAllocator* allocator,
                     const Scene&       scene,
                     uint32_t           maxLoadUnloads,
                     uint32_t           maxGroupsPerBuild,
                     uint32_t           maxClustersPerBuild,
                     uint32_t           positionTruncateBits);
  size_t                  memoryUsage() const { return batch.memoryUsage() + clasStaging.memoryUsage(); }
  LoadUnloadBatch         batch;
  vkobj::SemaphoreValue   clasBuildDone;
  ClasStaging             clasStaging;
  std::vector<PoolMemory> newClases;
};

// Streaming involves processing requests from the GPU in various queues.
// Requests may be expanded into many due to dependencies but may also result in
// no changes. For example, when clusters are already streamed in due to other
// dependencies. WorkCounter exists to check whether work remains or is
// available to process without blocking forever. It counts two buckets:
// requests, which may expand or filter into nothing, and results which are
// guaranteed output.
class WorkCounter
{
public:
  void addRequestWork(int32_t count)
  {
    std::lock_guard lk(mutex);
    m_requestWork += count;
    if(m_requestWork == 0)
      cv.notify_all();
  }

  void addResult(int32_t count)
  {
    std::lock_guard lk(mutex);
    if(m_resultCount == 0)
      cv.notify_all();
    m_resultCount += count;
  }

  // Waits for requests to be processed. Returns true when a result will be
  // emitted. Returns false when requests are processed with no results.
  bool waitForResult()
  {
    // Wait for either a result or work to run try (no more work and no results)
    std::unique_lock lk(mutex);
    cv.wait(lk, [this] { return m_resultCount != 0 || (m_requestWork == 0 && m_resultCount == 0); });
    return m_resultCount > 0;
  }

  bool busy() const
  {
    std::lock_guard lk(mutex);
    return m_resultCount != 0 && m_requestWork != 0;
  }

private:
  int32_t                 m_requestWork = 0;
  int32_t                 m_resultCount = 0;
  mutable std::mutex      mutex;
  std::condition_variable cv;
};

// Background streaming threads, their temporary resources and synchronized
// queues to pass jobs
class StreamingSceneVk
{
public:
#if 0
  static constexpr uint32_t MaxRequests          = 1;
  static constexpr uint32_t MaxLoadUnloads       = 1;
  static constexpr uint32_t PositionTruncateBits = 0;
  static constexpr uint32_t MaxGroupsPerBuild    = 1;
#else
  static constexpr uint32_t MaxRequests          = 256;
  static constexpr uint32_t MaxLoadUnloads       = 256;
  static constexpr uint32_t PositionTruncateBits = 0;
  static constexpr uint32_t MaxGroupsPerBuild    = 256;
#endif

  StreamingSceneVk(ResourceAllocator*    allocator,
                   SampleGlslCompiler&   glslCompiler,
                   VkDeviceSize          geometryBufferBytes,
                   VkCommandPool         initPool,
                   vkobj::TimelineQueue& initQueue,
                   const Scene&          scene,
                   SceneVK&              sceneVk,
                   bool                  requiresClas,
                   uint32_t              streamingTransferQueueFamilyIndex,
                   VkQueue               streamingTransferQueue);

  ~StreamingSceneVk();

  // Called on the render thread
  void makeRequests(vkobj::Buffer<uint8_t>& groupNeededFlags, vkobj::SemaphoreValue promisedSubmitSemaphoreState, VkCommandBuffer cmd);

  // Called on the render thread. Do not pass block=true unless
  // m_workCounter.waitForResult() returns true.
  void buildClasBatch(ResourceAllocator* allocator, vkobj::SemaphoreValue promisedSubmitSemaphoreState, bool block, VkCommandBuffer cmd);

  // Called on the render thread. Returns true if there were any groups
  // modifications made. Do not pass block=true unless
  // m_workCounter.waitForResult() returns true.
  bool modifyGroups(ResourceAllocator*            allocator,
                    const Scene&                  scene,
                    vkobj::Buffer<shaders::Mesh>& meshPointers,
                    std::vector<ClusterGroupVk>&  unloadGarbage,
                    vkobj::SemaphoreValue         unloadGarbageSemaphore,
                    bool                          block,
                    VkCommandBuffer               cmd,
                    uint64_t&                     totalResidentClusters,  // TODO: clean up too many params
                    uint64_t&                     totalResidentInstanceClusters);

  VkDeviceSize bytesAllocated() const { return m_memoryPool.bytesAllocated(); }
  VkDeviceSize internalFragmentation() const { return m_memoryPool.internalFragmentation(); }
  uint32_t     pendingRequests() const { return m_pendingRequests; }
  size_t       stagingMemoryUsage() const { return m_geometryLoaderMemory + m_staticStagingMemory; }
  bool         requiresClas() const { return m_requiresClas; }

  // Synchronously fulfill all streaming requests. Called on the render thread
  void flush(ResourceAllocator*            allocator,
             VkCommandPool                 pool,
             vkobj::TimelineQueue&         queue,
             const Scene&                  scene,
             vkobj::Buffer<uint8_t>&       groupNeededFlags,
             vkobj::Buffer<shaders::Mesh>& meshPointers,
             uint64_t&                     totalResidentClusters,  // TODO: clean up too many params
             uint64_t&                     totalResidentInstanceClusters);

private:
  void geometryLoaderEntrypoint(std::unique_ptr<const Scene> scene);  // pointer to war coverity large pass-by-value warning
  void loadGeometryBatch(const Scene& scene, streaming::RequestDependencyPipeline& requests, LoadUnloadBatch& batch);
  void loadUnloadGroupAllocations(const Scene&                 scene,
                                  LoadUnloadBatch&             batch,
                                  std::vector<PoolMemory>&     newClases,
                                  std::vector<ClusterGroupVk>& unloadGarbage);

  vkobj::Context    m_geometryLoaderContext;
  const uint32_t    m_maxClustersPerBuild;
  bool              m_requiresClas = false;  // HACK: skip building CLAS for rasterization
  vkobj::ByteBuffer m_memoryPoolBuffer;
  PoolAllocator     m_memoryPool;

  // The first intermediate data and queue of the streaming pipeline - GPU
  // cluster group requests to download in geometryLoaderEntrypoint(). The
  // streaming thread, m_geometryLoaderThread, puts these requests (which may be
  // empty) into its own internal queue, expands their dependencies and uploads
  // batches of geometry to the GPU.
  ProducerConsumer<RequestList, 3> m_pipeline0Requests;

  // Once uploaded, m_geometryLoaderThread hands batches of geometry back to the
  // render thread. For raytracing, cluster acceleration structures (CLAS) must
  // be built before the renderer can use the geometry. This is performed on the
  // render thread to give it more control over where in the frame this work is
  // performed. It could also happen on a dedicated compute queue, but high load
  // may introduce frame stutter at indeterminate times. The cost of this is
  // higher streaming latency and fixed throughput per frame.
  ProducerConsumer<LoadUnloadBatch, 2> m_pipeline1Geometry;

  // The CLAS batch is built into staging memory on the render thread, before
  // being compacted into its final allocation. This ultimately saves memory,
  // but introduces another sync in the pipeline because the variable-sized
  // allocation is made host side. Hence, a third queue is needed. The
  // allocation, pointer transfer and compaction happen on the render thread
  // too, but they're fairly quick so no background thread is used.
  // TODO: GPU-side memory pool allocator to skip host side allocation
  ProducerConsumer<BatchWithBuiltCLAS, 2> m_pipeline2GeometryCLAS;

  // Track where async jobs are in the pipeline and allow waiting for work to
  // complete to facilitate flush()
  WorkCounter m_workCounter;

  std::unordered_map<uint32_t, ClusterGroupVk> m_groups;  // Streaming geometry
  streaming::MakeRequestsProgram               m_requestsProgram;
  streaming::ModifyGroupsProgram               m_modifyGroupsProgram;
  streaming::FillClasInputProgram              m_fillClasInputProgram;
  streaming::PackClasProgram                   m_packClasProgram;
  std::atomic<bool>                            m_running              = true;
  std::atomic<uint32_t>                        m_pendingRequests      = 0;
  std::atomic<size_t>                          m_geometryLoaderMemory = 0;
  size_t                                       m_staticStagingMemory  = 0;
  std::jthread                                 m_geometryLoaderThread;
};

}  // namespace streaming
