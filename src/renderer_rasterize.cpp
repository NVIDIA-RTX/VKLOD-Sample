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

#include <nvvk/profiler_vk.hpp>
#include <renderer_rasterize.hpp>
#include <sample_vulkan_objects.hpp>
#include <scene.hpp>
#include <shaders/dh_bindings.h>


RasterizeRenderer::RasterizeRenderer(ResourceAllocator*    allocator,
                                     SampleGlslCompiler&   glslCompiler,
                                     VkCommandPool         initPool,
                                     uint32_t              initQueueFamilyIndex,
                                     VkQueue               initQueue,
                                     const RendererCommon& common,
                                     const Scene&          scene,
                                     const SceneVK&        sceneVk,
                                     Framebuffer&          framebuffer)
{
  std::ignore = initQueueFamilyIndex;

  VkDevice device = allocator->getDevice();

  // key drawing data
  {
    m_drawingData.constants = vkobj::Buffer<shaders::RasterizeConstants>(allocator, 1, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                                                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    m_drawingData.drawClusters =
        vkobj::Buffer<shaders::DrawCluster>(allocator, size_t(1) << m_config.maxDrawableClusterBits,
                                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    m_drawingData.drawIndirect = vkobj::Buffer<shaders::DrawMeshTasksIndirect>(
        allocator, 1, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    m_drawingData.drawStats =
        vkobj::Buffer<shaders::DrawStats>(allocator, 1, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    m_drawingData.drawStatsHostVisible =
        vkobj::Buffer<shaders::DrawStats>(allocator, MAX_CYCLES,
                                          VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    m_drawingData.drawStatsMapping = {m_drawingData.drawStatsHostVisible};
  }

  // key traversal data
  {
    m_traversalData.constants = vkobj::Buffer<shaders::TraversalConstants>(allocator, 1, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                                                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // Allocate only the node queue and job status buffer. The Cluster queue is
    // allocated just before use.
    m_traversalData.nodeQueue =
        vkobj::Buffer<shaders::EncodedNodeJob>(allocator, sceneVk.counts.maxTotalInstanceNodes,
                                               VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    m_traversalData.jobStatus =
        vkobj::Buffer<shaders::JobStatus>(allocator, 1, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    vkobj::ImmediateCommandBuffer initCmd(allocator->getDevice(), initPool, initQueue);
    vkCmdFillBuffer(initCmd, m_traversalData.nodeQueue, 0, m_traversalData.nodeQueue.size_bytes(), 0);
  }

  initDrawingPipeline(device, glslCompiler, common, scene, sceneVk, framebuffer);
  initTraversalPipeline(device, glslCompiler, framebuffer);
}

static uint32_t getBlockCount(uint32_t targetThreadCount, uint32_t blockSize)
{
  return (targetThreadCount + blockSize - 1) / blockSize;
}

void RasterizeRenderer::render(const RenderParams& params, const SceneVK& sceneVk, VkCommandBuffer cmd)
{
  VkMemoryBarrier memoryBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};

  //////////////////////////////////////////////////////////////////////////

  // handle resizes of scratch buffers due to ui settings / streaming

  resizeDrawingData(params);
  resizeTraversalData(params, sceneVk, cmd);

  //////////////////////////////////////////////////////////////////////////

  // setup basics

  shaders::RasterizeConstants rasterSceneInfo = {
      .instancesAddress           = sceneVk.instances.address(),
      .meshesAddress              = sceneVk.meshPointers.address(),
      .drawClustersAddress        = m_drawingData.drawClusters.address(),
      .drawStatsAddress           = m_drawingData.drawStats.address(),
      .config                     = m_config.shaders,
      .frame                      = 0,
      .errorOverDistanceThreshold = params.common.m_traversalParams.errorOverDistanceThreshold,
  };

  vkCmdUpdateBuffer(cmd, m_drawingData.constants, 0, sizeof(rasterSceneInfo), &rasterSceneInfo);

  // Zero draw indirect
  vkCmdFillBuffer(cmd, m_drawingData.drawIndirect, 0, sizeof(shaders::DrawMeshTasksIndirect), 0);
  // Zero draw stats
  vkCmdFillBuffer(cmd, m_drawingData.drawStats, 0, sizeof(shaders::DrawStats), 0);

  // Execute the traversal shader
  shaders::TraversalConstants traversalConstants = {
      .traversalParams              = params.common.m_traversalParams,
      .meshesAddress                = sceneVk.meshPointers.address(),
      .instancesAddress             = sceneVk.instances.address(),
      .nodeQueueAddress             = m_traversalData.nodeQueue.address(),
      .clusterQueueAddress          = m_traversalData.clusterQueue.address(),
      .jobStatusAddress             = m_traversalData.jobStatus.address(),
      .blasInputAddress             = vkobj::DeviceAddress<shaders::ClusterBLASInfoNV>(0),  // not relevant to raster
      .blasInputClustersAddress     = vkobj::DeviceAddress<uint64_t>(0),                    // not relevant to raster
      .drawClustersAddress          = m_drawingData.drawClusters.address(),
      .drawMeshTasksIndirectAddress = m_drawingData.drawIndirect.address(),
      .drawStatsAddress             = m_drawingData.drawStats.address(),
      .meshInstances                = vkobj::DeviceAddress<shaders::MeshInstances>(0),
      .sortingMeshInstances         = vkobj::DeviceAddress<shaders::SortingMeshInstances>(0),
      .nodeQueueSize                = uint32_t(m_traversalData.nodeQueue.size()),
      .clusterQueueSize             = uint32_t(m_traversalData.clusterQueue.size()),
      .itemsSize                    = uint32_t(sceneVk.instances.size()),
      .drawClustersSize             = uint32_t(m_drawingData.drawClusters.size()),
  };

  vkCmdUpdateBuffer(cmd, m_traversalData.constants, 0, sizeof(traversalConstants), &traversalConstants);

  // Zero the job queue
  vkCmdFillBuffer(cmd, m_traversalData.jobStatus, 0, sizeof(shaders::JobStatus), 0);


  //////////////////////////////////////////////////////////////////////////

  // first do traversal

  {
    nvvk::ProfilerVK::Section sec = params.profiler.timeRecurring("Traverse Scene LOD", cmd);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_traversal.bindings.getPipeLayout(), 0, 1,
                            m_traversal.bindings.getSets(), 0, nullptr);

    memoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                         &memoryBarrier, 0, nullptr, 0, nullptr);

    // sets up node and job queue so that every instance enqueues its lod root node.
    // outputs:
    //  m_traversalData.jobStatus
    //  m_traversalData.nodeQueue

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_traversal.traverseInitPipeline);
    vkCmdDispatch(cmd, getBlockCount(uint32_t(sceneVk.instances.size()), TRAVERSAL_WORKGROUP_SIZE), 1, 1);

    memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                         &memoryBarrier, 0, nullptr, 0, nullptr);

    // primary traversal over the instance's lod nodes.
    // reads:
    //   scene data: meshes, instances, cluster groups ... data
    // modifies:
    //  m_traversalData.jobStatus
    //  m_traversalData.nodeQueue
    //  m_traversalData.clusterQueue
    //
    //  state related to streaming:
    //    sceneVk.allGroupNeededFlags contains a big array over streaming info for all meshes
    //
    // outputs:
    //  m_traversalData.drawClusters
    //  m_traversalData.drawIndirect

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_traversal.traversePipeline);
    vkCmdDispatch(cmd, getBlockCount(4096u, TRAVERSAL_WORKGROUP_SIZE), 1, 1);

    memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                         &memoryBarrier, 0, nullptr, 0, nullptr);

    // sets up the indirect draw call
    // also clamps the number of generated draw calls to stay within limit
    // modifies:
    //  m_traversalData.drawIndirect
    //  m_traversalData.drawStats

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_traversal.traverseVerifyPipeline);
    vkCmdDispatch(cmd, 1, 1, 1);

    // graphics must wait
    memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_MESH_SHADER_BIT_NV | VK_PIPELINE_STAGE_TASK_SHADER_BIT_NV | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                         0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);
  }

  //////////////////////////////////////////////////////////////////////////

  // then draw

  {
    nvvk::ProfilerVK::Section sec = params.profiler.timeRecurring("Rasterize", cmd);

    VkClearValue colorClear{.color = {0.0F, 0.0F, 0.0F, 1.0F}};
    VkClearValue depthClear{.depthStencil = {1.0F, 0}};

    VkRenderingAttachmentInfo colorAttachment = {
        .sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView   = params.framebuffer.gbuffer().getColorImageView(0),
        .imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
        .loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp     = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue  = colorClear,
    };

    VkRenderingAttachmentInfo depthStencilAttachment{
        .sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView   = params.framebuffer.gbuffer().getDepthImageView(),
        .imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
        .loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp     = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue  = depthClear,
    };

    // Dynamic rendering information: color and depth attachments
    VkRenderingInfo renderingInfo{
        .sType                = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .flags                = VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT,
        .renderArea           = {{0, 0}, params.framebuffer.gbuffer().getSize()},
        .layerCount           = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments    = &colorAttachment,
        .pDepthAttachment     = &depthStencilAttachment,
    };

    vkCmdBeginRendering(cmd, &renderingInfo);

    VkExtent2D imageSize = params.framebuffer.size();

    VkViewport vp;
    vp.minDepth = 0;
    vp.maxDepth = 1.0f;
    vp.x        = 0;
    vp.y        = 0;
    vp.width    = float(imageSize.width);
    vp.height   = float(imageSize.height);

    VkRect2D scissor;
    scissor.extent = imageSize;
    scissor.offset = {0, 0};

    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_drawing.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_drawing.bindings.getPipeLayout(), 0, 1,
                            m_drawing.bindings.getSets(), 0, nullptr);


    // reads:
    //  m_traversalData.drawClusters
    //  m_traversalData.drawIndirect
    //  (as well as instances, meshes... data )

    vkCmdDrawMeshTasksIndirectNV(cmd, m_drawingData.drawIndirect, 0, 1, uint32_t(sizeof(shaders::DrawMeshTasksIndirect)));

    vkCmdEndRendering(cmd);
  }

  {
    nvvk::ProfilerVK::Section sec = params.profiler.timeRecurring("Read & HiZ", cmd);

    // barrier for copying stats to host
    VkBufferMemoryBarrier bufferBarrier = {};
    bufferBarrier.sType                 = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bufferBarrier.buffer                = m_drawingData.drawStats;
    bufferBarrier.size                  = m_drawingData.drawStats.size_bytes();
    bufferBarrier.srcAccessMask         = VK_ACCESS_SHADER_WRITE_BIT;
    bufferBarrier.dstAccessMask         = VK_ACCESS_TRANSFER_READ_BIT;

    // barrier for hiz generation
    VkImageMemoryBarrier imageBarrier            = {};
    imageBarrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    imageBarrier.image                           = params.framebuffer.gbuffer().getDepthImage();
    imageBarrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_DEPTH_BIT;
    imageBarrier.subresourceRange.baseMipLevel   = 0;
    imageBarrier.subresourceRange.levelCount     = VK_REMAINING_MIP_LEVELS;
    imageBarrier.subresourceRange.baseArrayLayer = 0;
    imageBarrier.subresourceRange.layerCount     = VK_REMAINING_ARRAY_LAYERS;

    imageBarrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    imageBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    imageBarrier.oldLayout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    imageBarrier.newLayout     = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_MESH_SHADER_BIT_NV | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1,
                         &bufferBarrier, 1, &imageBarrier);

    // copy back statistics to host visible buffer
    {
      VkBufferCopy copy;
      copy.dstOffset = sizeof(shaders::DrawStats) * (m_frame % MAX_CYCLES);
      copy.size      = sizeof(shaders::DrawStats);
      copy.srcOffset = 0;

      vkCmdCopyBuffer(cmd, m_drawingData.drawStats, m_drawingData.drawStatsHostVisible, 1, &copy);
    }

    // update hiz
    if(!params.common.m_frameIndex || !params.common.m_config.lockLodCamera)
    {
      params.framebuffer.hiz().cmdUpdateHiz(cmd, params.framebuffer.hizUpdate(), uint32_t(0));
    }

    // barrier to transition depth image back to rendering state

    imageBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    imageBarrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    imageBarrier.oldLayout     = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL;
    imageBarrier.newLayout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, 0, 0,
                         nullptr, 0, nullptr, 1, &imageBarrier);
  }

  m_frame++;
}

void RasterizeRenderer::resizeTraversalData(const RenderParams& params, const SceneVK& sceneVk, VkCommandBuffer cmd)
{
  if(!m_traversalData.clusterQueue || sceneVk.totalResidentInstanceClusters < m_traversalData.clusterQueue.size() / 3
     || sceneVk.totalResidentInstanceClusters > m_traversalData.clusterQueue.size())
  {
    size_t newClusterQueueSize = (sceneVk.totalResidentInstanceClusters * 3) / 2;
    LOGI("Reallocating traversal cluster queue: %zu\n", newClusterQueueSize);

    {
      params.garbage.emplace();
      params.garbage.back().semaphoreState = params.queueStates.primary.nextSubmitValue();  // will be signalled after rendering the current frame
      //params.garbage.back().traversalBuffers = {std::move(m_traversalData.clusterQueue)};
      params.garbage.back().buffers.push_back(std::move(m_traversalData.clusterQueue.moveByteBuffer()));
    }

    m_traversalData.clusterQueue =
        vkobj::Buffer<shaders::EncodedClusterJob>(params.context.allocator, newClusterQueueSize,
                                                  VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vkCmdFillBuffer(cmd, m_traversalData.clusterQueue, 0, m_traversalData.clusterQueue.size_bytes(), 0);
  }
}

void RasterizeRenderer::resizeDrawingData(const RenderParams& params)
{
  size_t newSize = size_t(1) << m_config.maxDrawableClusterBits;

  if(newSize != m_drawingData.drawClusters.size())
  {
    {
      params.garbage.emplace();
      params.garbage.back().semaphoreState = params.queueStates.primary.nextSubmitValue();  // will be signalled after rendering the current frame
      params.garbage.back().buffers.push_back(std::move(m_drawingData.drawClusters.moveByteBuffer()));
    }

    m_drawingData.drawClusters =
        vkobj::Buffer<shaders::DrawCluster>(params.context.allocator, newSize,
                                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  }
}

void RasterizeRenderer::updatedFrambuffer(ResourceAllocator* allocator, Framebuffer& framebuffer)
{
  VkWriteDescriptorSet write = m_traversal.bindings.makeWrite(0, shaders::BTraversalHiZTex, &framebuffer.hizFar());
  vkUpdateDescriptorSets(allocator->getDevice(), 1u, &write, 0, nullptr);
}

void RasterizeRenderer::initDrawingPipeline(VkDevice              device,
                                            SampleGlslCompiler&   glslCompiler,
                                            const RendererCommon& common,
                                            const Scene&          scene,
                                            const SceneVK&        sceneVk,
                                            Framebuffer&          framebuffer)
{
  // rasterization shaders

  shaderc::CompileOptions options = glslCompiler.defaultOptions();
  options.AddMacroDefinition("CLUSTER_VERTEX_COUNT", std::to_string(scene.counts.maxClusterVertexCount));
  options.AddMacroDefinition("CLUSTER_TRIANGLE_COUNT", std::to_string(scene.counts.maxClusterTriangleCount));

  vkobj::ShaderModule meshModule = reloadUntilCompiling(device, glslCompiler, "rasterize.mesh.glsl",
                                                        shaderc_shader_kind::shaderc_glsl_mesh_shader, &options);

  vkobj::ShaderModule fragmentModule = reloadUntilCompiling(device, glslCompiler, "rasterize.frag.glsl",
                                                            shaderc_shader_kind::shaderc_glsl_fragment_shader, &options);

  // descriptor set / binding

  {
    nvvk::DescriptorSetContainer& binding = m_drawing.bindings;

    VkShaderStageFlags shaderStageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_MESH_BIT_NV;

    binding.init(device);
    binding.addBinding(shaders::BRasterFrameInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, shaderStageFlags);
    binding.addBinding(shaders::BRasterConstants, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, shaderStageFlags);
    binding.addBinding(shaders::BRasterSkyParams, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, shaderStageFlags);
    binding.addBinding(shaders::BRasterTextures, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                       uint32_t(sceneVk.textureDescriptors.size()), shaderStageFlags);
    binding.initLayout();
    binding.initPipeLayout();
    binding.initPool(1);

    VkDescriptorBufferInfo            frameInfoDesc{common.m_bFrameInfo, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo            constantsDesc{m_drawingData.constants, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo            skyParamsDesc{common.m_bSkyParams, 0, VK_WHOLE_SIZE};
    std::vector<VkWriteDescriptorSet> writes;
    writes.emplace_back(binding.makeWrite(0, shaders::BRasterFrameInfo, &frameInfoDesc));
    writes.emplace_back(binding.makeWrite(0, shaders::BRasterConstants, &constantsDesc));
    writes.emplace_back(binding.makeWrite(0, shaders::BRasterSkyParams, &skyParamsDesc));
    if(sceneVk.textureDescriptors.size())
      writes.emplace_back(binding.makeWriteArray(0, shaders::BRasterTextures, sceneVk.textureDescriptors.data()));
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
  }

  // pipeline

  {
    nvvk::GraphicsPipelineState graphicsState = {};

    graphicsState.inputAssemblyState.topology  = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    graphicsState.rasterizationState.cullMode  = (VK_CULL_MODE_BACK_BIT);
    graphicsState.rasterizationState.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    graphicsState.depthStencilState.depthTestEnable       = VK_TRUE;
    graphicsState.depthStencilState.depthWriteEnable      = VK_TRUE;
    graphicsState.depthStencilState.depthCompareOp        = VK_COMPARE_OP_LESS;
    graphicsState.depthStencilState.depthBoundsTestEnable = VK_FALSE;
    graphicsState.depthStencilState.stencilTestEnable     = VK_FALSE;
    graphicsState.depthStencilState.minDepthBounds        = 0.0f;
    graphicsState.depthStencilState.maxDepthBounds        = 1.0f;

    graphicsState.multisampleState.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkFormat colorFormat = framebuffer.gbuffer().getColorFormat();
    VkFormat depthFormat = framebuffer.gbuffer().getDepthFormat();

    VkPipelineRenderingCreateInfo pipelineRendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR};
    pipelineRendering.colorAttachmentCount    = 1;
    pipelineRendering.pColorAttachmentFormats = &colorFormat;
    pipelineRendering.depthAttachmentFormat   = depthFormat;

    nvvk::GraphicsPipelineGenerator graphicsGen(device, m_drawing.bindings.getPipeLayout(), pipelineRendering, graphicsState);
    graphicsGen.addShader(meshModule, VK_SHADER_STAGE_MESH_BIT_NV);
    graphicsGen.addShader(fragmentModule, VK_SHADER_STAGE_FRAGMENT_BIT);

    VkPipeline pipeline = graphicsGen.createPipeline();
    m_drawing.pipeline  = vkobj::Pipeline(device, std::move(pipeline));
  }
}

void RasterizeRenderer::initTraversalPipeline(VkDevice device, SampleGlslCompiler& glslCompiler, Framebuffer& framebuffer)
{
  // traversal shaders

  shaderc::CompileOptions options = glslCompiler.defaultOptions();
  options.AddMacroDefinition("IS_RASTERIZATION", "1");
  options.AddMacroDefinition("TRAVERSE_PER_INSTANCE", "1");

  vkobj::ShaderModule traverseModule       = reloadUntilCompiling(device, glslCompiler, "traverse.comp.glsl",
                                                                  shaderc_shader_kind::shaderc_glsl_compute_shader, &options);
  vkobj::ShaderModule traverseInitModule   = reloadUntilCompiling(device, glslCompiler, "traverse_init.comp.glsl",
                                                                  shaderc_shader_kind::shaderc_glsl_compute_shader, &options);
  vkobj::ShaderModule traverseVerifyModule = reloadUntilCompiling(device, glslCompiler, "traverse_verify.comp.glsl",
                                                                  shaderc_shader_kind::shaderc_glsl_compute_shader, &options);

  // binding
  {
    nvvk::DescriptorSetContainer& binding = m_traversal.bindings;

    binding.init(device);
    binding.addBinding(shaders::BTraversalConstants, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    binding.addBinding(shaders::BTraversalHiZTex, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    binding.initLayout();
    binding.initPipeLayout();
    binding.initPool(1);

    VkDescriptorBufferInfo constantsDesc{m_traversalData.constants, 0, VK_WHOLE_SIZE};

    std::vector<VkWriteDescriptorSet> writes;
    writes.emplace_back(binding.makeWrite(0, shaders::BTraversalConstants, &constantsDesc));
    writes.emplace_back(binding.makeWrite(0, shaders::BTraversalHiZTex, &framebuffer.hizFar()));
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
  }

  // pipeline
  {
    VkComputePipelineCreateInfo computeCreateInfo = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};

    computeCreateInfo.stage = {
        .sType               = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext               = nullptr,
        .flags               = 0,
        .stage               = VK_SHADER_STAGE_COMPUTE_BIT,
        .pName               = "main",
        .pSpecializationInfo = nullptr,
    };

    computeCreateInfo.layout = m_traversal.bindings.getPipeLayout();

    VkPipeline pipeline;
    computeCreateInfo.stage.module = traverseModule;
    vkCreateComputePipelines(device, nullptr, 1, &computeCreateInfo, nullptr, &pipeline);
    m_traversal.traversePipeline = vkobj::Pipeline(device, std::move(pipeline));

    computeCreateInfo.stage.module = traverseInitModule;
    vkCreateComputePipelines(device, nullptr, 1, &computeCreateInfo, nullptr, &pipeline);
    m_traversal.traverseInitPipeline = vkobj::Pipeline(device, std::move(pipeline));

    computeCreateInfo.stage.module = traverseVerifyModule;
    vkCreateComputePipelines(device, nullptr, 1, &computeCreateInfo, nullptr, &pipeline);
    m_traversal.traverseVerifyPipeline = vkobj::Pipeline(device, std::move(pipeline));
  }
}

void RasterizeRenderer::ui(bool&, bool&)
{

  ImGui::Text("Rasterization");
  ImGuiH::InputIntClamped("Draw Clusters (bits)", &m_config.maxDrawableClusterBits, 0, 28, 1, 1, ImGuiInputTextFlags_EnterReturnsTrue);

  shaders::DrawStats drawStats =
      m_frame > MAX_CYCLES ? m_drawingData.drawStatsMapping.span()[(m_frame + 1) % MAX_CYCLES] : shaders::DrawStats{};

  ImGui::Text("Max Draw Clusters %d", uint32_t(m_drawingData.drawClusters.size()));
  ImGui::Text("Requested Draw Clusters %d (%f pct)", drawStats.requestedClusterCount,
              double(drawStats.requestedClusterCount) * 100.0 / double(m_drawingData.drawClusters.size()));
  ImGui::Text("Rastered Triangles %d", drawStats.triangleCount);

#if 0
  static bool asFloat = false;
  ImGui::Checkbox("as float", &asFloat);
  for(uint32_t i = 0; i < 128; i++)
  {
    if(asFloat)
      ImGui::Text("Debug %2d - %f", i, *((float*)&drawStats.debug[i]));
    else
      ImGui::Text("Debug %2d - %10d", i, drawStats.debug[i]);
  }
#endif
}
