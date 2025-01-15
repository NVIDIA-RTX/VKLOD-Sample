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

#include <renderer_common.hpp>
#include <sample_vulkan_objects.hpp>

RendererCommon::RendererCommon(ResourceAllocator* allocator, SampleGlslCompiler&, VkCommandPool, VkQueue, const SceneVK&)
    : m_traversalParams(initialTraversalParams())
    , m_skyParams(nvvkhl_shaders::initSimpleSkyParameters())
    , m_bFrameInfo(allocator, 1, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
    , m_bSkyParams(allocator, 1, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
{
  // Tone down the sky intensity relative to the sun
  m_skyParams.horizonColor *= 0.6;
  m_skyParams.groundColor *= 0.6;
  m_skyParams.skyColor *= 0.6;
  m_skyParams.groundColor = m_skyParams.horizonColor;

  // Brighter yellow sun
  m_skyParams.lightRadiance = glm::vec3(1.0f, 0.8f, 0.5f) * 2.0f;
}

void RendererCommon::cmdUpdateParams(Framebuffer& framebuffer, nvh::CameraManipulator& camera, AABB sceneAabb, VkCommandBuffer cmd)
{
  auto  imageSize       = framebuffer.size();
  float viewAspectRatio = float(imageSize.width) / float(imageSize.height);

  // Update the uniform buffer containing frame info
  shaders::FrameParams frameInfo{};
  const auto&          clip = camera.getClipPlanes();
  frameInfo.view            = camera.getMatrix();
  frameInfo.proj            = glm::perspectiveRH_ZO(glm::radians(camera.getFov()), viewAspectRatio, clip.x, clip.y);
  frameInfo.proj[1][1] *= -1;
  frameInfo.projInv         = glm::inverse(frameInfo.proj);
  frameInfo.viewInv         = glm::inverse(frameInfo.view);
  frameInfo.viewProj        = frameInfo.proj * frameInfo.view;
  frameInfo.camPos          = camera.getEye();
  frameInfo.fogHeightOffset = sceneAabb.max.y;
  frameInfo.fogHeightScale  = sceneAabb.max.y - sceneAabb.min.y;
  vkCmdUpdateBuffer(cmd, m_bFrameInfo, 0, sizeof(frameInfo), &frameInfo);

  // Reset m_frameAccumIndex if the camera changed
  if(frameInfo.view != m_lastFrameInfo.view || frameInfo.proj != frameInfo.proj)
    m_frameAccumIndex = 0;

  // Update the sky
  vkCmdUpdateBuffer(cmd, m_bSkyParams, 0, sizeof(m_skyParams), &m_skyParams);

  // first frame use same
  if(!m_frameIndex)
    m_lastFrameInfo = frameInfo;

  // Update traversal parameters
  if(!m_frameIndex || !m_config.lockLodCamera)
  {
    m_traversalParams.viewTransform = frameInfo.view;
    m_traversalParams.hizViewProj   = m_lastFrameInfo.viewProj;
  }

  float errorOverDistanceThreshold =
      nvclusterlodErrorOverDistance(m_config.lodTargetPixelError, glm::radians(camera.getFov()), float(imageSize.height));

  m_traversalParams.errorOverDistanceThreshold = errorOverDistanceThreshold;
  m_traversalParams.distanceToUNorm32 =
      float(double(std::numeric_limits<uint32_t>::max()) / double(glm::length(sceneAabb.max - sceneAabb.min)));
  m_traversalParams.useOcclusion      = m_config.useOcclusion ? 1 : 0;
  m_traversalParams.hizSizeFactors    = framebuffer.hizFarFactors();
  m_traversalParams.hizSizeMax        = framebuffer.hizFarMax();
  m_traversalParams.hizViewport       = glm::vec2(framebuffer.size().width, framebuffer.size().height);

  m_lastFrameInfo = frameInfo;
  m_frameIndex++;
}

bool RendererCommon::uiLod(const Scene&, const Framebuffer&)
{
  using namespace ImGuiH;
  bool changed = false;

  ImGui::Text("Level of Detail (View)");
  PropertyEditor::begin();
  changed = PropertyEditor::entry(
                "Max. Pixel Error",
                [&] { return ImGui::SliderFloat("Max. Pixel Error", &m_config.lodTargetPixelError, 0.5f, 1000.0f); })
            || changed;
  PropertyEditor::entry("Lock Camera", [&] { return ImGui::Checkbox("Lock Camera", &m_config.lockLodCamera); });
  PropertyEditor::entry("Use Occlusion", [&] { return ImGui::Checkbox("Use Occlusion", &m_config.useOcclusion); });
  PropertyEditor::end();

  return changed;
}

bool RendererCommon::uiSky()
{
  using namespace ImGuiH;
  bool changed = false;
  ImGui::Text("Sun Orientation");
  PropertyEditor::begin();
  glm::vec3 dir                = m_skyParams.directionToLight;
  changed                      = ImGuiH::azimuthElevationSliders(dir, false) || changed;
  m_skyParams.directionToLight = dir;
  PropertyEditor::end();
  ImGui::End();
  return changed;
}

void Framebuffer::deinit()
{
  m_hiz.deinitUpdateViews(m_hizUpdate);
  m_allocator->destroy(m_imgHizFar);
}

glm::vec4 Framebuffer::hizFarFactors() const
{
  glm::vec4 vec;
  m_hizUpdate.farInfo.getShaderFactors(glm::value_ptr(vec));
  return vec;
}

float Framebuffer::hizFarMax() const
{
  return m_hizUpdate.farInfo.getSizeMax();
}

void Framebuffer::initHiz(glm::uvec2 vpSize)
{
  m_hiz.setupUpdateInfos(m_hizUpdate, vpSize.x, vpSize.y, s_depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);

  // hiz
  VkImageCreateInfo hizImageInfo = {};
  hizImageInfo.sType             = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  hizImageInfo.imageType         = VK_IMAGE_TYPE_2D;
  hizImageInfo.format            = m_hizUpdate.farInfo.format;
  hizImageInfo.extent.width      = m_hizUpdate.farInfo.width;
  hizImageInfo.extent.height     = m_hizUpdate.farInfo.height;
  hizImageInfo.mipLevels         = m_hizUpdate.farInfo.mipLevels;
  hizImageInfo.extent.depth      = 1;
  hizImageInfo.arrayLayers       = 1;
  hizImageInfo.samples           = VK_SAMPLE_COUNT_1_BIT;
  hizImageInfo.tiling            = VK_IMAGE_TILING_OPTIMAL;
  hizImageInfo.usage             = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
  hizImageInfo.flags             = 0;
  hizImageInfo.initialLayout     = VK_IMAGE_LAYOUT_UNDEFINED;


  m_imgHizFar = m_allocator->createImage(hizImageInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  m_hizUpdate.sourceImage = gbuffer().getDepthImage();
  m_hizUpdate.farImage    = m_imgHizFar.image;
  m_hizUpdate.nearImage   = VK_NULL_HANDLE;

  m_hiz.initUpdateViews(m_hizUpdate);
  m_hiz.updateDescriptorSet(m_hizUpdate, 0);

  // initial resource transitions

  // fixme gbuffer and this should do it differently
  nvvk::CommandPool cpool(m_allocator->getDevice(), 0);
  VkCommandBuffer   cmd = cpool.createCommandBuffer();
  nvvk::cmdBarrierImageLayout(cmd, m_imgHizFar.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
  cpool.submitAndWait(cmd);
}
