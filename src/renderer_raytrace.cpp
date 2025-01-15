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

#include <nvvk/profiler_vk.hpp>
#include <renderer_raytrace.hpp>
#include <sample_vulkan_objects.hpp>
#include <scene.hpp>
#include <shaders/dh_bindings.h>

PathtracingPipeline::PathtracingPipeline(SampleGlslCompiler&                glslCompiler,
                                         ResourceAllocator*                 allocator,
                                         uint32_t                           queueGCT,
                                         std::vector<VkDescriptorSetLayout> descriptorSetLayouts)
{
  nvvk::DebugUtil dutil(allocator->getDevice());

  vkobj::ShaderModule shaderRaygen = reloadUntilCompiling(allocator->getDevice(), glslCompiler, "pathtrace.rgen.glsl",
                                                          shaderc_shader_kind::shaderc_glsl_raygen_shader);
  vkobj::ShaderModule shaderMiss   = reloadUntilCompiling(allocator->getDevice(), glslCompiler, "pathtrace.rmiss.glsl",
                                                          shaderc_shader_kind::shaderc_glsl_miss_shader);
  vkobj::ShaderModule shaderClosestHit = reloadUntilCompiling(allocator->getDevice(), glslCompiler, "pathtrace.rchit.glsl",
                                                              shaderc_shader_kind::shaderc_glsl_closesthit_shader);
  std::vector<VkPipelineShaderStageCreateInfo> shaderStages{
      {
          .sType               = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .pNext               = nullptr,
          .flags               = 0,
          .stage               = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
          .module              = shaderRaygen,
          .pName               = "main",
          .pSpecializationInfo = nullptr,
      },
      {
          .sType               = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .pNext               = nullptr,
          .flags               = 0,
          .stage               = VK_SHADER_STAGE_MISS_BIT_KHR,
          .module              = shaderMiss,
          .pName               = "main",
          .pSpecializationInfo = nullptr,
      },
      {
          .sType               = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .pNext               = nullptr,
          .flags               = 0,
          .stage               = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
          .module              = shaderClosestHit,
          .pName               = "main",
          .pSpecializationInfo = nullptr,
      },
  };
  for([[maybe_unused]] auto& stage : shaderStages)
    assert(stage.module != VK_NULL_HANDLE);
  dutil.setObjectName(shaderStages[0].module, "Raygen");
  dutil.setObjectName(shaderStages[1].module, "Miss");
  dutil.setObjectName(shaderStages[2].module, "Closest Hit");

  std::vector<VkRayTracingShaderGroupCreateInfoKHR> shadingGroups{
      {.sType                           = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
       .pNext                           = nullptr,
       .type                            = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
       .generalShader                   = 0 /* "raygen" shaderStages[0] */,
       .closestHitShader                = VK_SHADER_UNUSED_KHR,
       .anyHitShader                    = VK_SHADER_UNUSED_KHR,
       .intersectionShader              = VK_SHADER_UNUSED_KHR,
       .pShaderGroupCaptureReplayHandle = nullptr},
      {.sType                           = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
       .pNext                           = nullptr,
       .type                            = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
       .generalShader                   = 1 /* "miss" shaderStages[1] */,
       .closestHitShader                = VK_SHADER_UNUSED_KHR,
       .anyHitShader                    = VK_SHADER_UNUSED_KHR,
       .intersectionShader              = VK_SHADER_UNUSED_KHR,
       .pShaderGroupCaptureReplayHandle = nullptr},
      {.sType                           = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
       .pNext                           = nullptr,
       .type                            = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR,
       .generalShader                   = VK_SHADER_UNUSED_KHR,
       .closestHitShader                = 2 /* "closest hit" shaderStages[2] */,
       .anyHitShader                    = VK_SHADER_UNUSED_KHR,
       .intersectionShader              = VK_SHADER_UNUSED_KHR,
       .pShaderGroupCaptureReplayHandle = nullptr}};

  // Push constants - small struct to upload in the command buffer each frame
  VkPushConstantRange pushConstantRange{VK_SHADER_STAGE_ALL, 0, sizeof(shaders::PathtraceConstant)};

  VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{
      .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pNext                  = nullptr,
      .flags                  = 0,
      .setLayoutCount         = static_cast<uint32_t>(descriptorSetLayouts.size()),
      .pSetLayouts            = descriptorSetLayouts.data(),
      .pushConstantRangeCount = 1,
      .pPushConstantRanges    = &pushConstantRange,
  };
  m_pipelineLayout = vkobj::PipelineLayout(allocator->getDevice(), pipelineLayoutCreateInfo);
  dutil.DBG_NAME(m_pipelineLayout);

  // Enable cluster acceleration structures in the pipeline
  VkRayTracingPipelineClusterAccelerationStructureCreateInfoNV pipelineClusterAccelerationStructureCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CLUSTER_ACCELERATION_STRUCTURE_CREATE_INFO_NV,
      .pNext = nullptr,
      .allowClusterAccelerationStructure = true};

  // Assemble the shader stages and recursion depth info into the ray tracing pipeline
  VkRayTracingPipelineCreateInfoKHR pipelineCreateInfo{
      .sType                        = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
      .pNext                        = &pipelineClusterAccelerationStructureCreateInfo,
      .flags                        = 0,
      .stageCount                   = static_cast<uint32_t>(shaderStages.size()),
      .pStages                      = shaderStages.data(),
      .groupCount                   = static_cast<uint32_t>(shadingGroups.size()),
      .pGroups                      = shadingGroups.data(),
      .maxPipelineRayRecursionDepth = PATHTRACE_MAX_RECURSION_DEPTH,
      .pLibraryInfo                 = nullptr,
      .pLibraryInterface            = nullptr,
      .pDynamicState                = nullptr,
      .layout                       = m_pipelineLayout,
      .basePipelineHandle           = VK_NULL_HANDLE,
      .basePipelineIndex            = 0,
  };
  VkPipeline pipeline;
  NVVK_CHECK(vkCreateRayTracingPipelinesKHR(allocator->getDevice(), {}, {}, 1, &pipelineCreateInfo, nullptr, &pipeline));
  m_pipeline = vkobj::Pipeline(allocator->getDevice(), std::move(pipeline));
  dutil.DBG_NAME(m_pipeline);

  // Requesting ray tracing properties
  VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtPipelineProperties{
      .sType                              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR,
      .pNext                              = nullptr,
      .shaderGroupHandleSize              = 0,
      .maxRayRecursionDepth               = 0,
      .maxShaderGroupStride               = 0,
      .shaderGroupBaseAlignment           = 0,
      .shaderGroupHandleCaptureReplaySize = 0,
      .maxRayDispatchInvocationCount      = 0,
      .shaderGroupHandleAlignment         = 0,
      .maxRayHitAttributeSize             = 0,
  };
  VkPhysicalDeviceProperties2 prop2{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &rtPipelineProperties, .properties = {}};
  vkGetPhysicalDeviceProperties2(allocator->getPhysicalDevice(), &prop2);

  // Create utilities to create BLAS/TLAS and the Shader Binding Table (SBT)
  m_sbt = std::make_unique<SBT>();
  m_sbt->setup(allocator->getDevice(), queueGCT, allocator, rtPipelineProperties);
  m_sbt->create(m_pipeline, pipelineCreateInfo);
}

std::unique_ptr<nvvk::DescriptorSetContainer> PathtracingPipeline::makeDescriptorSet(VkDevice device, uint32_t textureCount)
{
  // This descriptor set, holds the top level acceleration structure and the output image
  auto result = std::make_unique<nvvk::DescriptorSetContainer>(device);
  result->addBinding(BRtTlas, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_ALL);
  result->addBinding(BRtOutImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
  result->addBinding(BRtFrameInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
  result->addBinding(BRtSkyParam, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
  result->addBinding(BRtTextures, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, textureCount, VK_SHADER_STAGE_ALL);
  result->initLayout();
  result->initPool(1);
  return result;
}

void PathtracingPipeline::writeDescriptorSet(VkDevice                               device,
                                             VkAccelerationStructureKHR             tlas,
                                             VkImageView                            framebuffer,
                                             VkBuffer                               frameInfo,
                                             VkBuffer                               skyParams,
                                             std::span<const VkDescriptorImageInfo> textures,
                                             nvvk::DescriptorSetContainer&          descriptorSet)
{
  // Write path tracing shader descriptor set
  VkWriteDescriptorSetAccelerationStructureKHR tlasDesc{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
                                                        .pNext                      = nullptr,
                                                        .accelerationStructureCount = 1,
                                                        .pAccelerationStructures    = &tlas};
  VkDescriptorImageInfo             framebufferColorDesc{{}, framebuffer, VK_IMAGE_LAYOUT_GENERAL};
  VkDescriptorBufferInfo            frameInfoDesc{frameInfo, 0, VK_WHOLE_SIZE};
  VkDescriptorBufferInfo            skyParamsDesc{skyParams, 0, VK_WHOLE_SIZE};
  std::vector<VkWriteDescriptorSet> writes;
  writes.emplace_back(descriptorSet.makeWrite(0, BRtTlas, &tlasDesc));
  writes.emplace_back(descriptorSet.makeWrite(0, BRtOutImage, &framebufferColorDesc));
  writes.emplace_back(descriptorSet.makeWrite(0, BRtFrameInfo, &frameInfoDesc));
  writes.emplace_back(descriptorSet.makeWrite(0, BRtSkyParam, &skyParamsDesc));
  if(textures.size())
    writes.emplace_back(descriptorSet.makeWriteArray(0, BRtTextures, textures.data()));
  vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

void PathtracingPipeline::writeDescriptorSetFramebuffer(VkDevice device, VkImageView framebuffer, nvvk::DescriptorSetContainer& descriptorSet)
{
  VkDescriptorImageInfo framebufferColorDesc{{}, framebuffer, VK_IMAGE_LAYOUT_GENERAL};
  VkWriteDescriptorSet  write = descriptorSet.makeWrite(0, BRtOutImage, &framebufferColorDesc);
  vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
}

void PathtracingPipeline::trace(VkCommandBuffer                   cmd,
                                std::vector<VkDescriptorSet>      descriptorSets,
                                const shaders::PathtraceConstant& pushConstant,
                                glm::uvec2                        vpSize) const
{
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_pipelineLayout, 0,
                          static_cast<uint32_t>(descriptorSets.size()), descriptorSets.data(), 0, nullptr);
  vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(pushConstant), &pushConstant);
  const auto& regions = m_sbt->getRegions();
  vkCmdTraceRaysKHR(cmd, regions.data(), &regions[1], &regions[2], &regions[3], vpSize.x, vpSize.y, 1);
}


RaytraceRenderer::RaytraceRenderer(const RenderInitParams& params, RaytraceConfig& config)
    : m_config(config)
    , m_rtBinding(PathtracingPipeline::makeDescriptorSet(params.context.allocator->getDevice(),
                                                         uint32_t(params.sceneVk.textures.size())))
    , m_rtPipeline(params.glslCompiler, params.context.allocator, params.context.queueFamilyIndex, {m_rtBinding->getLayout()})
{
  VkAccelerationStructureKHR tlas = VK_NULL_HANDLE;
  if(m_config.perInstanceTraversal)
  {
    m_lodInstanceTraverser = LodInstanceTraverser{params.context.allocator,
                                                  params.glslCompiler,
                                                  params.context.commandPool,
                                                  params.context.queue,
                                                  params.context.queueFamilyIndex,
                                                  params.scene,
                                                  params.sceneVk};
    tlas                   = m_lodInstanceTraverser->tlas();
  }
  else
  {
    m_lodMeshTraverser = LodMeshTraverser{params.context.allocator,
                                          params.glslCompiler,
                                          params.context.commandPool,
                                          params.context.queue,
                                          params.context.queueFamilyIndex,
                                          params.scene,
                                          params.sceneVk};
    tlas               = m_lodMeshTraverser->tlas();
  }
  PathtracingPipeline::writeDescriptorSet(params.context.allocator->getDevice(), tlas, params.framebuffer.colorView(),
                                          params.common.m_bFrameInfo, params.common.m_bSkyParams,
                                          params.sceneVk.textureDescriptors, *m_rtBinding);
}

void RaytraceRenderer::updatedFrambuffer(ResourceAllocator* allocator, Framebuffer& framebuffer)
{
  PathtracingPipeline::writeDescriptorSetFramebuffer(allocator->getDevice(), framebuffer.colorView(), *m_rtBinding);
}

void RaytraceRenderer::render(const RenderParams& params, const SceneVK& sceneVk, VkCommandBuffer cmd)
{
  // Scene traversal
  {
    params.garbage.emplace();
    params.garbage.back().semaphoreState = params.queueStates.primary.nextSubmitValue();

    assert(bool(m_lodInstanceTraverser) ^ bool(m_lodMeshTraverser));  // must be one but not both
    if(m_lodInstanceTraverser)
      params.garbage.back().traversalBuffers =
          m_lodInstanceTraverser->traverseAndBuildBVH(params.context.allocator, params.common.m_traversalParams,
                                                      sceneVk, params.profiler, cmd);
    if(m_lodMeshTraverser)
      params.garbage.back().traversalBuffers =
          m_lodMeshTraverser->traverseAndBuildBVH(params.context.allocator, params.common.m_traversalParams, sceneVk,
                                                  params.profiler, cmd);
  }

  float errorOverDistanceThreshold =
      nvclusterlodErrorOverDistance(params.common.m_config.lodTargetPixelError, glm::radians(CameraManip.getFov()),
                                    float(params.framebuffer.size().height));

  shaders::PathtraceConstant pushConstant{
      .instancesAddress           = sceneVk.instances.address(),
      .meshesAddress              = sceneVk.meshPointers.address(),
      .config                     = m_config.shaders,
      .frame                      = params.common.m_frameAccumIndex++,
      .errorOverDistanceThreshold = errorOverDistanceThreshold,
  };

  (void)pushConstant;

  // Ray trace
  {
    nvvk::ProfilerVK::Section timer(params.profiler, "Ray Trace", cmd);
    m_rtPipeline.trace(cmd, {m_rtBinding->getSet()}, pushConstant,
                       {params.framebuffer.size().width, params.framebuffer.size().height});
  }
}

void RaytraceRenderer::ui(bool& recreateRenderer, bool& resetFrameAccumulation)
{
  ImGui::Text("Ray Tracing");

  if(m_lodInstanceTraverser)
  {
    ImGui::Text("Memory: Traverse %s  BLAS %s  TLAS %s", formatBytes(m_lodInstanceTraverser->traversalMemory()).c_str(),
                formatBytes(m_lodInstanceTraverser->blasDeviceMemory()).c_str(),
                formatBytes(m_lodInstanceTraverser->tlasDeviceMemory()).c_str());
  }
  if(m_lodMeshTraverser)
  {
    ImGui::Text("Memory: Traverse %s  BLAS %s  TLAS %s", formatBytes(m_lodMeshTraverser->traversalMemory()).c_str(),
                formatBytes(m_lodMeshTraverser->blasDeviceMemory()).c_str(),
                formatBytes(m_lodMeshTraverser->tlasDeviceMemory()).c_str());
  }

  using namespace ImGuiH;
  PropertyEditor::begin();
  resetFrameAccumulation = false;
  recreateRenderer       = false;
  recreateRenderer       = recreateRenderer | PropertyEditor::entry("Per-Instance Traversal", [&] {
                       return ImGui::Checkbox("Per-Instance Traversal", reinterpret_cast<bool*>(&m_config.perInstanceTraversal));
                     });
  resetFrameAccumulation = resetFrameAccumulation | PropertyEditor::entry("Pathtrace", [&] {
                             return ImGui::Checkbox("Pathtrace", reinterpret_cast<bool*>(&m_config.shaders.pathtrace));
                           });
  resetFrameAccumulation = resetFrameAccumulation | PropertyEditor::entry("Subpixel Samples", [&] {
                             return ImGui::SliderInt("Subpixel Samples", &m_config.shaders.sampleCountPixel, 1, 32);
                           });
  ImGui::BeginDisabled(!m_config.shaders.pathtrace);
  resetFrameAccumulation =
      resetFrameAccumulation | PropertyEditor::entry("Pathtrace Depth", [&] {
        return ImGui::SliderInt("Pathtrace Depth", &m_config.shaders.maxDepth, 1, PATHTRACE_MAX_RECURSION_DEPTH);
      });
  ImGui::EndDisabled();
  ImGui::BeginDisabled(m_config.shaders.pathtrace);
  resetFrameAccumulation = resetFrameAccumulation | PropertyEditor::entry("Ambient Occlusion Samples", [&] {
                             return ImGui::SliderInt("Ambient Occlusion Samples", &m_config.shaders.sampleCountAO, 1, 32);
                           });
  resetFrameAccumulation = resetFrameAccumulation | PropertyEditor::entry("Ambient Occlusion Radius", [&] {
                             return ImGui::SliderFloat("Ambient Occlusion Radius", &m_config.shaders.aoRadius, 0.0f, 1000.0f);
                           });
  ImGui::EndDisabled();
  PropertyEditor::end();

  const char* visualizeItems[] = VISUALIZE_ENUM_NAMES;
  if(ImGui::BeginCombo("LOD Visualization", visualizeItems[m_config.shaders.lodVisualization]))
  {
    for(int32_t i = 0; i < int32_t(std::size(visualizeItems)); i++)
    {
      bool isSelected = (m_config.shaders.lodVisualization == i);
      if(ImGui::Selectable(visualizeItems[i], isSelected))
      {
        m_config.shaders.lodVisualization = i;
        resetFrameAccumulation            = true;
      }
      if(isSelected)
        ImGui::SetItemDefaultFocus();
    }
    ImGui::EndCombo();
  }
}
