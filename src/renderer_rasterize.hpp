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
#include <lod_traverser.hpp>
#include <nvvk/pipeline_vk.hpp>
#include <nvvk/sbtwrapper_vk.hpp>
#include <renderer_common.hpp>
#include <sample_glsl_compiler.hpp>
#include <shaders/rasterize_device_host.h>

struct RasterizeConfig
{
  shaders::RasterizeConfig shaders                = {.lodVisualization = 0};
  uint32_t                 maxDrawableClusterBits = 20;
};

class RasterizeRenderer : public RendererInterface
{
public:
  RasterizeRenderer(ResourceAllocator*    allocator,
                    SampleGlslCompiler&   glslCompiler,
                    VkCommandPool         initPool,
                    uint32_t              initQueueFamilyIndex,
                    VkQueue               initQueue,
                    const RendererCommon& common,
                    const Scene&          scene,
                    const SceneVK&        sceneVk,
                    Framebuffer&          framebuffer);


  virtual void updatedFrambuffer(ResourceAllocator* allocator, Framebuffer& framebuffer) override;
  virtual void render(const RenderParams& params, const SceneVK& sceneVk, VkCommandBuffer cmd) override;
  virtual void ui(bool& recreateRenderer, bool& resetFrameAccumulation) override;
  virtual bool requiresCLAS() const override { return false; }

private:
  static const uint64_t MAX_CYCLES = 4;

  RasterizeConfig m_config;

  struct Drawing
  {
    vkobj::Pipeline pipeline;

    nvvk::DescriptorSetContainer bindings;

  } m_drawing;

  struct DrawingData
  {
    vkobj::Buffer<shaders::RasterizeConstants> constants;

    vkobj::Buffer<shaders::DrawCluster>           drawClusters;
    vkobj::Buffer<shaders::DrawMeshTasksIndirect> drawIndirect;
    vkobj::Buffer<shaders::DrawStats>             drawStats;
    vkobj::Buffer<shaders::DrawStats>             drawStatsHostVisible;
    vkobj::BufferMapping<shaders::DrawStats>      drawStatsMapping;
  } m_drawingData;

  struct Traversal
  {
    vkobj::Pipeline traversePipeline;
    vkobj::Pipeline traverseInitPipeline;
    vkobj::Pipeline traverseVerifyPipeline;

    nvvk::DescriptorSetContainer bindings;
  } m_traversal;

  struct TraversalData
  {
    vkobj::Buffer<shaders::TraversalConstants> constants;
    vkobj::Buffer<shaders::EncodedNodeJob>     nodeQueue;
    vkobj::Buffer<shaders::EncodedClusterJob>  clusterQueue;
    vkobj::Buffer<shaders::JobStatus>          jobStatus;
  } m_traversalData;

  uint64_t m_frame = 0;

  void initDrawingPipeline(VkDevice              device,
                           SampleGlslCompiler&   glslCompiler,
                           const RendererCommon& common,
                           const Scene&          scene,
                           const SceneVK&        sceneVk,
                           Framebuffer&          framebuffer);

  void initTraversalPipeline(VkDevice device, SampleGlslCompiler& glslCompiler, Framebuffer& framebuffer);

  void resizeTraversalData(const RenderParams& params, const SceneVK& sceneVk, VkCommandBuffer cmd);
  void resizeDrawingData(const RenderParams& params);
};
