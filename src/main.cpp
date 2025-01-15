/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
// See README.md for docs

#include <acceleration_structures.hpp>
#include <algorithm>
#include <any>
#include <cstddef>
#include <cstdlib>
#include <debug_range_summary.hpp>
#include <filesystem>
#include <glm/glm.hpp>
#include <glm/gtc/random.hpp>
#include <imgui.h>
#include <imgui/imgui_camera_widget.h>
#include <imgui/imgui_helper.h>
#include <implot.h>
#include <lod_streaming_scene.hpp>
#include <memory>
#include <nvclusterlod/nvclusterlod_hierarchy.h>
#include <nvh/commandlineparser.hpp>
#include <nvh/fileoperations.hpp>
#include <nvh/primitives.hpp>
#include <nvvk/debug_util_vk.hpp>
#include <nvvk/dynamicrendering_vk.hpp>
#include <nvvk/images_vk.hpp>
#include <nvvk/memallocator_vma_vk.hpp>
#include <nvvkhl/alloc_vma.hpp>
#include <nvvkhl/application.hpp>
#include <nvvkhl/element_camera.hpp>
#include <nvvkhl/element_gui.hpp>
#include <nvvkhl/element_nvml.hpp>
#include <nvvkhl/element_profiler.hpp>
#include <nvvkhl/gbuffer.hpp>
#include <renderer_common.hpp>
#include <renderer_rasterize.hpp>
#include <renderer_raytrace.hpp>
#include <sample_app_element.hpp>
#include <sample_camera_paths.hpp>
#include <sample_glsl_compiler.hpp>
#include <sample_raytracing_objects.hpp>
#include <sample_vulkan_objects.hpp>
#include <scene.hpp>
#include <third_party/imgui/backends/imgui_impl_vulkan.h>
#include <type_traits>
#include <vector>
#include <vulkan/vulkan_core.h>

using streaming::moveAny;
namespace fs = std::filesystem;

constexpr const char* INSTALL_SUBDIRECTORY = "GLSL_" PROJECT_NAME;

fs::path getRendercachePath(const fs::path& cacheDir, const fs::path& gltfPath)
{
  return cacheDir / ("rendercache_" + gltfPath.filename().string() + ".dat");
}

template <class T, size_t N>
class PlotSamples : public std::array<T, N>
{
public:
  PlotSamples() { push(T(0)); }  // implot only renders if there are at least two samples
  T&   first() { return (*this)[pivot()]; }
  T&   last() { return (*this)[(m_size + N - 1) % N]; }
  void push(T v)
  {
    (*this)[m_size % N] = v;
    ++m_size;
  }
  size_t pivot() const { return m_size < N ? 0 : m_size % N; }
  size_t size() const { return std::min(m_size, N); }

private:
  size_t m_size = 0;
};

struct Config
{
  int            rendererIndex                 = 0;
  int            streamingBufferSizeMB         = 1024;  // 1GB
  bool           invalidateNextLoadRendercache = false;
  bool           streamingStress               = false;  // random camera position each frame
  SceneLodConfig sceneLodConfig;
};

std::unique_ptr<streaming::StreamingSceneVk> makeStreaming(ResourceAllocator*    allocator,
                                                           SampleGlslCompiler&   glslCompiler,
                                                           nvvkhl::Application*  app,
                                                           Scene&                scene,
                                                           SceneVK&              sceneVk,
                                                           vkobj::TimelineQueue& initQueueState,
                                                           VkDeviceSize          geometryBufferBytes,
                                                           bool                  requiresClas)
{
  return std::make_unique<streaming::StreamingSceneVk>(allocator, glslCompiler, geometryBufferBytes,
                                                       app->getCommandPool(), initQueueState, scene, sceneVk, requiresClas,
                                                       app->getQueue(2).familyIndex /* transfer */, app->getQueue(2).queue /* transfer */
  );
}

// Scene container to guarantee lifetime and destruction order
struct RenderableScene
{
  RenderableScene(const fs::path&       gltfPath,
                  const fs::path&       cachePath,
                  const Config&         config,
                  bool                  invalidateCache,
                  ResourceAllocator*    allocator,
                  VkCommandPool         initCommandPool,
                  vkobj::TimelineQueue& initQueue,
                  SampleGlslCompiler&   glslCompiler,
                  nvvkhl::Application*  app,
                  bool                  requiresClas)
      : mapping{gltfPath, cachePath, config.sceneLodConfig, invalidateCache}
      , view{*mapping.data}
      , vk{allocator, view, initCommandPool, initQueue.queue}
      , streaming(makeStreaming(allocator, glslCompiler, app, view, vk, initQueue, VkDeviceSize(config.streamingBufferSizeMB) << 20, requiresClas))
  {
  }
  SceneFile mapping;  // file mapping objects
  Scene     view;     // pointers into the file mapping
  SceneVK   vk;       // GPU data, mostly uploaded from SceneMapping

  // Background threads and "groups" of geometry and acceleration structures
  std::unique_ptr<streaming::StreamingSceneVk> streaming;
};

// Lazy renderer container with a name and factory function. This saves inlining
// create() function whenever we recreate the renderer.
struct Renderer
{
  using CreateFunc = std::function<std::unique_ptr<RendererInterface>()>;
  RendererInterface* get()
  {
    if(!instance)
      instance = create();
    return instance.get();
  }
  std::string                        name;
  bool                               requiresClas;  // HACK: flag to skip building CLAS for rasterization
  CreateFunc                         create;
  std::unique_ptr<RendererInterface> instance;
};

SynchronizedMemAllocator makeMemAllocator(nvvkhl::Application* app)
{
  return SynchronizedMemAllocator(std::make_unique<VMAMemAllocator>(VmaAllocatorCreateInfo{
      .flags                       = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
      .physicalDevice              = app->getPhysicalDevice(),
      .device                      = app->getDevice(),
      .preferredLargeHeapBlockSize = 0,
      .pAllocationCallbacks        = nullptr,
      .pDeviceMemoryCallbacks      = nullptr,
      .pHeapSizeLimit              = nullptr,
      .pVulkanFunctions            = nullptr,
      .instance                    = app->getInstance(),
      .vulkanApiVersion = VK_API_VERSION_1_2,  // Not 1.3. See "# vma" in nvpro_core/third_party/CMakeLists.txt
      .pTypeExternalMemoryHandleTypes = nullptr,
  }));
}

class Sample
{
public:
  Sample(nvvkhl::Application* app, glm::uvec2 vpSize, const std::filesystem::path& gltfPath, const std::filesystem::path& cacheDir)
      : m_app(app)  // held for making temp command buffers
      , m_memAlloc(makeMemAllocator(app))
      , m_context(m_app->getDevice(),
                  m_app->getPhysicalDevice(),
                  &m_memAlloc,
                  m_app->getQueue(0).familyIndex,
                  m_app->getQueue(0).queue)
      , m_queueStates(m_app)
      , m_glslCompiler{
        // Shader include directories for permutations of relative exe paths
        INSTALL_SUBDIRECTORY, // search after cmake --install
        "shaders", // this project's shaders, debugging from the source directory (canonical)
        NVPRO_CORE_DIR, // nvpro_core shaders
        PROJECT_RELDIRECTORY "shaders", // and again if CWD is not the source directory
        PROJECT_NVPRO_CORE_RELDIRECTORY}
      , m_framebuffer(std::make_unique<Framebuffer>(m_context.allocator, m_glslCompiler, vpSize))
      , m_scene{gltfPath,
                getRendercachePath(cacheDir, gltfPath),
                m_config,
                false,
                m_context.allocator,
                m_context.commandPool,
                m_queueStates.primary,
                m_glslCompiler,
                m_app,
                true}
      , m_rendererCommon{m_context.allocator, m_glslCompiler, m_context.commandPool, app->getQueue(0).queue, m_scene.vk}
      , m_profiler(std::make_shared<nvvkhl::ElementProfiler>())
      , m_cameraPaths(std::make_shared<CameraPathsElement>())
      , m_cacheDir(cacheDir)
  {
    // The profiler is special because we need access to it from within the
    // 'Sample' class, so it's created and added here.
    // Similarly, frames can be recorded when playing camera animation
    m_app->addElement(m_profiler);
    m_app->addElement(m_cameraPaths);

    // Initial camera view for the scene loaded
    float sceneSize    = glm::length(m_scene.view.worldAABB.max - m_scene.view.worldAABB.min);
    float sceneSizeDec = powf(10.0f, ceilf(log10f(sceneSize)));
    CameraManip.setClipPlanes({sceneSizeDec * 0.001f, sceneSizeDec * 20.0f});
    CameraManip.setLookat(m_scene.view.worldAABB.center() + glm::vec3{-0.2F, 0.4F, 0.8F} * sceneSize,
                          m_scene.view.worldAABB.center(), {0.0F, 1.0F, 0.0F});
    CameraManip.setFov(80.0f);

    // Populate the lazily-created list of renderers
    m_renderers.push_back(Renderer{
        .name         = "Raytrace Clusters",
        .requiresClas = true,
        .create =
            [this]() {
              return std::make_unique<RaytraceRenderer>(
                  RenderInitParams{
                      .context      = m_context,
                      .glslCompiler = m_glslCompiler,
                      .common       = m_rendererCommon,
                      .scene        = m_scene.view,
                      .sceneVk      = m_scene.vk,
                      .framebuffer  = *m_framebuffer,
                  },
                  m_rendererRaytraceConfig);
            },
        .instance = {},
    });
    m_renderers.push_back(Renderer{
        .name         = "Rasterize Clusters",
        .requiresClas = false,
        .create =
            [this]() {
              return std::make_unique<RasterizeRenderer>(m_context.allocator, m_glslCompiler, m_context.commandPool,
                                                         m_app->getQueue(0).familyIndex, m_context.queue,
                                                         m_rendererCommon, m_scene.view, m_scene.vk, *m_framebuffer);
            },
        .instance = {},
    });

    // Keep staging memory around, otherwise continuous allocations will block the
    // render thread and cause a lot of stutter.
    m_context.allocator->getStaging()->setFreeUnusedOnRelease(false);
    m_context.allocator->getStaging()->freeUnused();
  }

  void load(const std::filesystem::path& filename)
  {
    // Synchronize for a simple way to make sure no GPU objects are still in use
    NVVK_CHECK(vkQueueWaitIdle(m_context.queue));

    // Renderers are all scene dependent and need to be reset. This avoids some
    // complexity at the cost of recreating shaders and pipeline objects
    // unnecessarily.
    for(auto& r : m_renderers)
      r.instance.reset();

    // Load the new scene. This may take a while
    RenderableScene(std::move(m_scene)); // Free the current scene first
    m_scene = RenderableScene{
        filename,
        getRendercachePath(m_cacheDir, filename),
        m_config,
        m_config.invalidateNextLoadRendercache,
        m_context.allocator,
        m_context.commandPool,
        m_queueStates.primary,
        m_glslCompiler,
        m_app,
        m_renderers[m_config.rendererIndex].requiresClas,
    };
    m_config.invalidateNextLoadRendercache = false;
    m_rendererCommon.m_frameAccumIndex     = 0;  // reset framebuffer accumulation

    // Clear the initial scene loading staging memory
    m_context.allocator->getStaging()->freeUnused();
  }

  void renderUI()
  {
    {
      // Setting menu
      ImGui::Begin("Settings");

      if(ImGui::Button("Load Scene"))
      {
        std::string filename = NVPSystem::windowOpenFileDialog(m_app->getWindowHandle(), "Pick scene file",
                                                               "Supported (glTF 2.0)|*.gltf;*.glb;"
                                                               "|All|*.*");
        if(!filename.empty())
          load(filename);
      }

      ImGuiH::CameraWidget();

      using namespace ImGuiH;

      std::vector<const char*> rendererNames;
      for(auto& r : m_renderers)
        rendererNames.push_back(r.name.c_str());
      ImGui::Combo("Renderer", &m_config.rendererIndex, rendererNames.data(), int(rendererNames.size()));

      ImGui::Text("Streaming");
      ImGui::Text("Resident Clusters: %zu\n", (size_t)m_scene.vk.totalResidentClusters);
      ImGui::Text("Resident Clusters x Instances: %zu\n", (size_t)m_scene.vk.totalResidentInstanceClusters);
      float streamingMemory = float(m_scene.streaming->bytesAllocated()) / 1024.0f / 1024.0f;
      if(streamingMemory != m_memoryUsageHistory.last())
        m_memoryUsageHistory.push(streamingMemory);
      if(ImPlot::BeginPlot("##memory_usage", ImVec2(-1, 150 * ImGui::GetIO().FontGlobalScale)))
      {
        ImPlot::SetupAxis(ImAxis_X1, nullptr, ImPlotAxisFlags_NoDecorations);
        ImPlot::SetupAxisLimits(ImAxis_X1, 0, static_cast<double>(m_memoryUsageHistory.size()), ImGuiCond_Always);
        ImPlot::SetupAxis(ImAxis_Y1, "Memory (MiB)", ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_LockMin);
        ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, 100000.0);
        ImPlot::SetAxes(ImAxis_X1, ImAxis_Y1);
        auto plot = [](auto name, auto& series) {
          ImPlot::PlotLine(name, series.data(), static_cast<int>(series.size()), 1.0f, 0.0f, ImPlotShadedFlags_None,
                           static_cast<int>(series.pivot()), sizeof(*series.data()));
        };
        plot("Streaming", m_memoryUsageHistory);
        ImPlot::EndPlot();
      }
      ImGui::Text("Pending Requests: %u", m_scene.streaming->pendingRequests());
      ImGui::Text("Inner Fragmentation: %.1f MiB", float(m_scene.streaming->internalFragmentation() >> 10) / 1024.0f);
      VkDeviceSize stagingAllocatedSize, stagingUsedSize;
      m_context.allocator->getStaging()->getUtilization(stagingAllocatedSize, stagingUsedSize);
      ImGui::Text("Staging memory: %.1f MiB", float((stagingAllocatedSize + m_scene.streaming->stagingMemoryUsage()) >> 10) / 1024.0f);
      bool remakeStreaming = false;
      PropertyEditor::begin();
      auto clusterSizeSlider = [&] {
        return ImGui::SliderInt("Streaming Buffer Size", (int*)&m_config.streamingBufferSizeMB, 100, 4000, "%d",
                                ImGuiSliderFlags_AlwaysClamp);
      };
      remakeStreaming = PropertyEditor::entry("Streaming Buffer Size", clusterSizeSlider) || remakeStreaming;
      PropertyEditor::entry("Stress Test", [&] { return ImGui::Checkbox("Stress Test", &m_config.streamingStress); });
      PropertyEditor::end();
      if(remakeStreaming || m_scene.streaming->requiresClas() != m_renderers[m_config.rendererIndex].requiresClas)
      {
        NVVK_CHECK(vkQueueWaitIdle(m_context.queue));

        // Garbage may reference the memory pool owned by m_scene.streaming and
        // must be cleared now. Could make its allocator a weak_ptr.
        m_garbage = {};

        m_scene.streaming.reset();
        m_scene.streaming = makeStreaming(m_context.allocator, m_glslCompiler, m_app, m_scene.view, m_scene.vk,
                                          m_queueStates.primary, VkDeviceSize(m_config.streamingBufferSizeMB) << 20,
                                          m_renderers[m_config.rendererIndex].requiresClas);
        m_rendererCommon.m_frameAccumIndex = 0;
      }

      if(m_rendererCommon.uiLod(m_scene.view, *m_framebuffer))
        m_rendererCommon.m_frameAccumIndex = 0;  // reset framebuffer accumulation

      // Renderer-specific UI. Returns true if options that require the renderer to be recreated were changed.
      bool needRecreate = false, resetFrameAccumulation = false;
      m_renderers[m_config.rendererIndex].get()->ui(needRecreate, resetFrameAccumulation);
      if(needRecreate)
        m_renderers[m_config.rendererIndex].instance.reset();
      if(resetFrameAccumulation)
        m_rendererCommon.m_frameAccumIndex = 0;

      ImGui::Separator();

      ImGui::Text("Level of Detail (Mesh)");
      PropertyEditor::begin();
      PropertyEditor::entry("Cluster Size", [&] {
        return ImGui::SliderInt("Cluster Size", (int*)&m_config.sceneLodConfig.clusterSize, 8, 256);
      });
      PropertyEditor::entry("Cluster Group Size", [&] {
        return ImGui::SliderInt("Cluster Group Size", (int*)&m_config.sceneLodConfig.clusterGroupSize, 2, 32);
      });
      PropertyEditor::entry("Decimation Factor", [&] {
        return ImGui::SliderFloat("Decimation Factor", &m_config.sceneLodConfig.lodLevelDecimationFactor, 0.1f, 0.9f);
      });
      PropertyEditor::end();
      if(ImGui::Button("Rebuild LOD"))
      {
        m_config.invalidateNextLoadRendercache = true;
        load(m_scene.mapping.path.string().c_str());
      }

      if(m_rendererCommon.uiSky())
        m_rendererCommon.m_frameAccumIndex = 0;  // reset framebuffer accumulation
    }

    {  // Rendering Viewport
      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
      ImGui::Begin("Viewport");

      // Display the G-Buffer image
      ImGui::Image(m_framebuffer->colorDescriptorSet(), ImGui::GetContentRegionAvail());

      ImGui::End();
      ImGui::PopStyleVar();
    }
  }

  void render(VkCommandBuffer cmd)
  {
    // Free unused allocations after all work that references them has finished.
    // E.g. check if the submit in GroupModsList::modifyGroups() has finished
    // before freeing streaming memory pool allocations, calling the
    // streaming::ClusterGroupVk destructor
    {
      nvvk::ProfilerVK::Section timer(*m_profiler, "Free Garbage", cmd);
      while(!m_garbage.empty() && m_garbage.front().semaphoreState.wait(m_context.device, 0 /* no wait */))
      {
        if(m_garbage.front().stagingSetId)
        {
          // Release the allocator's staging memory for reuse now that command
          // buffers referencing it have finished.
          m_context.allocator->getStaging()->releaseResourceSet(*m_garbage.front().stagingSetId);
        }
        m_garbage.pop();
      }
    }

    // Streaming stress test with a random camera position each frame
    if(m_config.streamingStress)
    {
      float sceneSize = glm::length(m_scene.view.worldAABB.max - m_scene.view.worldAABB.min);
      CameraManip.setLookat(m_scene.view.worldAABB.center() + (glm::vec3{-0.2F, 0.4F, 0.8F} + glm::ballRand(1.0f)) * sceneSize,
                            m_scene.view.worldAABB.center(), {0.0F, 1.0F, 0.0F});
    }

    bool reloadShaders = false;

    // Press "R" to reload traversal and rendering shaders
    if(ImGui::IsKeyPressed(ImGui::GetKeyIndex(ImGuiKey_R)))
    {
      // Make sure no GPU objects are in use
      NVVK_CHECK(vkQueueWaitIdle(m_context.queue));

      // Reload traversal shaders by re-creating the common renderer
      // TODO: recreate only shader modules and pipelines?
      m_rendererCommon =
          RendererCommon{m_context.allocator, m_glslCompiler, m_context.commandPool, m_context.queue, m_scene.vk};

      // This happens anyway, but for completeness, reset framebuffer
      // accumulation
      m_rendererCommon.m_frameAccumIndex = 0;

      // Reload renderer shaders
      for(auto& r : m_renderers)
        r.instance.reset();

      reloadShaders = true;
    }

    if(reloadShaders || m_scene.streaming->requiresClas() != m_renderers[m_config.rendererIndex].requiresClas)
    {
      NVVK_CHECK(vkQueueWaitIdle(m_context.queue));

      // Garbage may reference the memory pool owned by m_scene.streaming and
      // must be cleared now. Could make its allocator a weak_ptr.
      m_garbage = {};

      m_scene.streaming.reset();
      m_scene.streaming = makeStreaming(m_context.allocator, m_glslCompiler, m_app, m_scene.view, m_scene.vk,
                                        m_queueStates.primary, VkDeviceSize(m_config.streamingBufferSizeMB) << 20,
                                        m_renderers[m_config.rendererIndex].requiresClas);
    }

    // Build one batch of cluster acceleration structures for the streaming
    // pipeline.
    {
      nvvk::ProfilerVK::Section timer(*m_profiler, "Build CLAS", cmd);
      m_scene.streaming->buildClasBatch(m_context.allocator, m_queueStates.primary.nextSubmitValue(), false, cmd);
    }

    // Insert scene geometry streaming from the streaming thread before
    // traversal to compute LOD. Keep any unloaded pages around for another
    // frame since commands referencing them may still be in flight.
    {
      nvvk::ProfilerVK::Section              timer(*m_profiler, "Modify Groups", cmd);
      std::vector<streaming::ClusterGroupVk> garbage;
      vkobj::SemaphoreValue garbageSemaphore = m_queueStates.primary.nextSubmitValue();  // don't reuse GPU buffers until this is signalled
      m_scene.streaming->modifyGroups(m_context.allocator, m_scene.view, m_scene.vk.meshPointers, garbage, garbageSemaphore,
                                      false, cmd, m_scene.vk.totalResidentClusters, m_scene.vk.totalResidentInstanceClusters);
      if(!garbage.empty())
      {
        m_garbage.emplace();
        m_garbage.back().streamingGarbage = std::move(garbage);
        m_garbage.back().semaphoreState   = garbageSemaphore;
      }
    }

    // Render the scene. This includes traversing LOD hierarchies to create
    // per-instance LODs. If the renderer is a raytracer, acceleration
    // structures are built during this process. TODO: some rendering commands
    // get recorded into 'cmd' but submitted by the nvvkhl Application so the
    // order here is counterintuitive!
    {
      RenderParams renderParams{m_context, m_rendererCommon, *m_framebuffer, *m_profiler, m_garbage, m_queueStates};
      m_rendererCommon.cmdUpdateParams(*m_framebuffer, nvh::CameraManipulator::Singleton(), m_scene.view.worldAABB, cmd);
      m_renderers[m_config.rendererIndex].get()->render(renderParams, m_scene.vk, cmd);
    }

    // Gather any new geometry requests from traversal and send them to the
    // streaming thread. Results will be picked up next frame or when next
    // available.
    {
      nvvk::ProfilerVK::Section timer(*m_profiler, "Make Requests", cmd);
      m_scene.streaming->makeRequests(m_scene.vk.allGroupNeededFlags, m_queueStates.primary.nextSubmitValue(), cmd);
    }

    // Record the current batch of staging buffers to reuse once transfer
    // commands have finished
    m_garbage.emplace();
    m_garbage.back().semaphoreState = m_queueStates.primary.nextSubmitValue();
    m_garbage.back().stagingSetId   = m_context.allocator->getStaging()->finalizeResourceSet();


    // Finish the last save before starting the next
    if(m_lastFrameSave.joinable())
    {
      m_lastFrameSave.join();
    }
    #if 1
    if(m_cameraPaths->saveFrames())
    {
      auto saveThread = std::jthread([this, after = m_queueStates.primary.nextSubmitValue()]() {
        after.wait(m_context.device);
        // From nvvkhl::Application::saveScreenShot()
        VkImage        dstImage;
        VkDeviceMemory dstImageMemory;
        {
          vkobj::ImmediateCommandBuffer cmd(m_context.device, m_context.commandPool, m_context.queue);
          m_app->imageToRgba8Linear(cmd, m_context.device, m_context.physicalDevice, m_framebuffer->gbuffer().getColorImage(0),
                                    m_framebuffer->gbuffer().getSize(), dstImage, dstImageMemory);
        }
        m_app->saveImageToFile(m_context.device, dstImage, dstImageMemory, m_framebuffer->gbuffer().getSize(),
                               fmt::format("frame_{:04}.png", m_cameraPaths->animationFrame()), 98);
        vkUnmapMemory(m_context.device, dstImageMemory);
        vkFreeMemory(m_context.device, dstImageMemory, nullptr);
        vkDestroyImage(m_context.device, dstImage, nullptr);
      });
      m_lastFrameSave = std::move(saveThread);
    }
    #endif

    {
      VkSemaphoreSubmitInfo signalSubmitInfo = m_queueStates.primary.submitInfoAndAdvance(VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT);
      m_app->addSignalSemaphore(signalSubmitInfo);
    }
  }

  void resize(const glm::uvec2& vpSize)
  {
    // Sync to make sure any in-flight framebuffer access has completed. An
    // alternative would be to just move the current framebuffer onto m_garbage.
    NVVK_CHECK(vkQueueWaitIdle(m_context.queue));

    m_framebuffer = std::make_unique<Framebuffer>(m_context.allocator, m_glslCompiler, vpSize);

    for(auto& renderer : m_renderers)
      if(renderer.instance)
        renderer.instance->updatedFrambuffer(m_context.allocator, *m_framebuffer);

    m_rendererCommon.m_frameAccumIndex = 0;  // gbuffer is recreated so reset the accumulation count
  }

private:
  nvvkhl::Application*                     m_app = nullptr;
  Config                                   m_config;
  SynchronizedMemAllocator                 m_memAlloc;
  vkobj::Context                           m_context;
  TimelineQueueContainer                   m_queueStates;
  SampleGlslCompiler                       m_glslCompiler;
  std::unique_ptr<Framebuffer>             m_framebuffer;
  RenderableScene                          m_scene;
  PlotSamples<float, 20>                   m_memoryUsageHistory;
  RendererCommon                           m_rendererCommon;
  std::vector<Renderer>                    m_renderers;
  RaytraceConfig                           m_rendererRaytraceConfig;
  std::shared_ptr<nvvkhl::ElementProfiler> m_profiler;
  std::shared_ptr<CameraPathsElement>      m_cameraPaths;
  std::queue<Garbage>                      m_garbage;
  fs::path                                 m_cacheDir;
  std::jthread                             m_lastFrameSave;
};

int main(int argc, char** argv)
{
  try
  {
    // Cluster accelration structure not currently supported by validation layers
    bool enableValidation = false;

    // Parse command line arguments
    fs::path                 exeDirectoryPath   = nvh::getExecutablePath().parent_path();
    std::vector<std::string> defaultSearchPaths = {
        fs::absolute(exeDirectoryPath / PROJECT_DOWNLOAD_RELDIRECTORY).string(),  // regular build
        fs::absolute(exeDirectoryPath / "media").string(),                        // install build
    };
    std::string gltfPath  = nvh::findFile("bunny_v2/bunny.gltf", defaultSearchPaths);
    std::string cacheDir  = fs::current_path().string();  // or fs::temp_directory_path()?
    bool        printHelp = false;
    nvh::CommandLineParser args("vk_continuous_lod_clusters - a vulkan sample to demo continuous level of detail with ray tracing");
    args.addArgument({"-m", "--mesh"}, &gltfPath, "Mesh filename (*.gltf)");
    args.addArgument({"-c", "--cache-dir"}, &cacheDir, "Directory to keep render cache files. Default is CWD, " + cacheDir);
    args.addArgument({"--validate"}, &enableValidation,
                     "Enable validation layers (may break depending on VK_NV_cluster_acceleration_structure support)");
    args.addArgument({"-h", "--help"}, &printHelp, "Print Help");
    if(!args.parse(argc, argv) || printHelp || gltfPath.empty())
    {
      args.printHelp();
      return printHelp ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    nvvkhl::ApplicationCreateInfo spec;
    spec.name  = PROJECT_NAME " Example";
    spec.vSync = false;

    nvvk::ContextCreateInfo vkSetup;
    vkSetup          = nvvk::ContextCreateInfo(enableValidation);
    vkSetup.apiMajor = 1;
    vkSetup.apiMinor = 3;

    vkSetup.addDeviceExtension(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
    VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructure{
        .sType                              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
        .pNext                              = nullptr,
        .accelerationStructure              = VK_FALSE,
        .accelerationStructureCaptureReplay = VK_FALSE,
        .accelerationStructureIndirectBuild = VK_FALSE,
        .accelerationStructureHostCommands  = VK_FALSE,
        .descriptorBindingAccelerationStructureUpdateAfterBind = VK_FALSE,
    };
    vkSetup.addDeviceExtension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, false, &accelerationStructure);  // To build acceleration structures
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingPipeline{
        .sType              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR,
        .pNext              = nullptr,
        .rayTracingPipeline = VK_FALSE,
        .rayTracingPipelineShaderGroupHandleCaptureReplay      = VK_FALSE,
        .rayTracingPipelineShaderGroupHandleCaptureReplayMixed = VK_FALSE,
        .rayTracingPipelineTraceRaysIndirect                   = VK_FALSE,
        .rayTraversalPrimitiveCulling                          = VK_FALSE,
    };
    vkSetup.addDeviceExtension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, false, &rayTracingPipeline);  // To use vkCmdTraceRaysKHR
    vkSetup.addDeviceExtension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);  // Required by ray tracing pipeline
    vkSetup.addDeviceExtension(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
    VkPhysicalDeviceRayTracingPositionFetchFeaturesKHR rayTracingPositionFetch{
        .sType                   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_POSITION_FETCH_FEATURES_KHR,
        .pNext                   = nullptr,
        .rayTracingPositionFetch = VK_FALSE,
    };
    vkSetup.addDeviceExtension(VK_KHR_RAY_TRACING_POSITION_FETCH_EXTENSION_NAME, false, &rayTracingPositionFetch);
    vkSetup.addDeviceExtension(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME, false);
#ifndef NDEBUG
    if(enableValidation)
    {
      vkSetup.addInstanceExtension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME, false);  // for debugPrintfEXT, causes nvvk::Context::initDebugUtils() to be called
      vkSetup.addDeviceExtension(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME, false);  // for debugPrintfEXT
    }
#endif

    VkPhysicalDeviceMeshShaderFeaturesNV meshShaderStructure{
        .sType      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_NV,
        .pNext      = nullptr,
        .taskShader = VK_FALSE,
        .meshShader = VK_FALSE,
    };
    vkSetup.addDeviceExtension(VK_NV_MESH_SHADER_EXTENSION_NAME, false, &meshShaderStructure);

    // Add cluster acceleration structure extension
    static VkPhysicalDeviceClusterAccelerationStructureFeaturesNV clusterAccelerationStructureFeatures = {
        .sType                        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CLUSTER_ACCELERATION_STRUCTURE_FEATURES_NV,
        .pNext                        = nullptr,
        .clusterAccelerationStructure = VK_TRUE,
    };
    vkSetup.addDeviceExtension(VK_NV_CLUSTER_ACCELERATION_STRUCTURE_EXTENSION_NAME, false,
                               &clusterAccelerationStructureFeatures, VK_NV_CLUSTER_ACCELERATION_STRUCTURE_SPEC_VERSION);

    // Required for GPU buffer download with
    // nvvk::StagingMemoryManager::cmdFromAddressNV()
    static VkPhysicalDeviceCopyMemoryIndirectFeaturesNV copyMemoryIndirectFeatures = {
        .sType        = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COPY_MEMORY_INDIRECT_FEATURES_NV,
        .pNext        = nullptr,
        .indirectCopy = VK_FALSE,
    };
    vkSetup.addDeviceExtension(VK_NV_COPY_MEMORY_INDIRECT_EXTENSION_NAME, false, &copyMemoryIndirectFeatures);

    // Surface extensions
    nvvkhl::addSurfaceExtensions(vkSetup.instanceExtensions);
    vkSetup.addDeviceExtension(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

    // Create the vulkan context
    nvvk::Context vkctx;
    if(!vkctx.init(vkSetup))
    {
      return EXIT_FAILURE;
    }

    // Check the extension exists
    if(clusterAccelerationStructureFeatures.clusterAccelerationStructure != VK_TRUE)
    {
      LOGE("ERROR: The Cluster Acceleration Structure feature is not supported by the loaded vulkan implementation");
      return EXIT_FAILURE;
    }

    // Fill app context parameters
    spec.instance       = vkctx.m_instance;
    spec.device         = vkctx.m_device;
    spec.physicalDevice = vkctx.m_physicalDevice;
    spec.queues         = {
        {vkctx.m_queueGCT.familyIndex, vkctx.m_queueGCT.queueIndex, vkctx.m_queueGCT.queue},
        {vkctx.m_queueC.familyIndex, vkctx.m_queueC.queueIndex, vkctx.m_queueC.queue},
        {vkctx.m_queueT.familyIndex, vkctx.m_queueT.queueIndex, vkctx.m_queueT.queue},
    };

    // UI default docking
    spec.dockSetup = [](ImGuiID viewportID) {
      ImGuiID settingID = ImGui::DockBuilderSplitNode(viewportID, ImGuiDir_Right, 0.2F, nullptr, &viewportID);
      ImGui::DockBuilderDockWindow("Settings", settingID);
    };

    {
      // Create the application
      nvvkhl::Application app(spec);

      // Add all application elements
      app.addElement(std::make_shared<nvvkhl::ElementCamera>());
      app.addElement(std::make_shared<nvvkhl::ElementDefaultMenu>());         // Menu / Quit
      app.addElement(std::make_shared<nvvkhl::ElementDefaultWindowTitle>());  // Window title info
      app.addElement(std::make_shared<nvvkhl::ElementNvml>());
      app.addElement(std::make_shared<SampleAppElement<Sample>>(std::filesystem::path(gltfPath), cacheDir));
      app.run();
    }
  }
  catch(const std::exception& e)
  {
    // Catch-all case. Anything is fatal.
    LOGE("Exception thrown: %s\n", e.what());
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
