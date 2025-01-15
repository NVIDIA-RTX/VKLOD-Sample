/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <filesystem>
#include <glm/glm.hpp>
#include <nvvkhl/application.hpp>
#include <optional>

// An AppElement to manage the lifetime of Sample, creating it only after a
// viewport with non-zero size exists. Currently all arguments must be stored in
// this object. A nicer alternative would be a callback/lambda to create the
// sample.
template <class AttachedElement>
class SampleAppElement : public nvvkhl::IAppElement
{
public:
  SampleAppElement(const std::filesystem::path& gltfPath, const std::filesystem::path& cacheDir)
      : m_gltfPath(gltfPath)
      , m_cacheDir(cacheDir)
  {
  }

private:
  virtual void onAttach(nvvkhl::Application* app) override
  {
    m_app = app;
    if(m_vpSize.x != 0)
      m_element.emplace(app, m_vpSize, m_gltfPath, m_cacheDir);
  }
  virtual void onDetach() override
  {
    // nvvkhl::Application may still have frame jobs in flight, which depend on
    // GPU objects owned by m_element
    vkDeviceWaitIdle(m_app->getDevice());
    m_element.reset();
    m_app = nullptr;
  }
  virtual void onResize(uint32_t width, uint32_t height) override
  {
    m_vpSize = {width, height};
    if(m_app && !m_element)
      m_element.emplace(m_app, m_vpSize, m_gltfPath, m_cacheDir);
    else if(m_element)
      m_element->resize(m_vpSize);
  }
  virtual void onUIRender() override
  {
    if(m_element)
      m_element->renderUI();
  }
  virtual void onRender(VkCommandBuffer cmd) override
  {
    if(m_element)
      m_element->render(cmd);
  }
  virtual void onFileDrop(const char* filename) override
  {
    if(m_element)
      m_element->load(filename);
  }
  std::filesystem::path          m_gltfPath;
  std::filesystem::path          m_cacheDir;
  nvvkhl::Application*           m_app    = nullptr;
  glm::uvec2                     m_vpSize = {0, 0};
  std::optional<AttachedElement> m_element;
};
