/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <glm/ext/quaternion_trigonometric.hpp>
#include <glm/geometric.hpp>
#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>
#include <imgui.h>
#include <imgui/imgui_helper.h>
#include <nvh/cameramanipulator.hpp>
#include <nvvkhl/application.hpp>
#include <third_party/tinygltf/json.hpp>
#include <vector>

namespace camera_paths {

// GLM to json
template <class T>
nlohmann::json make_json(const T& value)
{
  nlohmann::json j;
  for(int i = 0; i < T::length(); ++i)
    j.push_back(value[i]);
  return j;
}

// GLM from json
template <class T>
T json_get(const nlohmann::json& j)
{
  if((int)j.size() != T::length())
    throw std::runtime_error("JSON array size does not match GLM type");
  T value{};
  for(int i = 0; i < T::length(); ++i)
    value[i] = j[i].get<typename T::value_type>();
  return value;
}

// All attributes to interpolate over time. Could be nicer to allow separate
// timelines for each attribute.
struct Keyframe
{
  glm::vec3              position;
  glm::quat              rotation;
  float                  distance      = 1.0f;
  float                  fov           = 80.0f;
  float                  durationScale = 1.0f;  // TODO: should be stored outside as time between keyframes
  static Keyframe        fromJSON(const nlohmann::json& keyframe);
  static Keyframe        fromCamera(const nvh::CameraManipulator& camera);
  void                   toCamera(nvh::CameraManipulator& camera);
  nlohmann::json         toJSON() const;
};

// An array of keyframes, a function to interpolate between them and UI controls
struct CameraPath
{
  std::string           name;
  std::vector<Keyframe> keyframes;
  float                 duration = 30.0f;

  int   selectedIndex  = -1;
  float m_seekPosition = 0.0f;
  float cpScale        = 2.0f / 3.0f;

  CameraPath(const char* newName);
  CameraPath(const std::string& newName);
  explicit CameraPath(const nlohmann::json& ipath);
  nlohmann::json toJSON() const;
  Keyframe       interpolate(float t);
  void           onUIRender();
};

}  // namespace camera_paths

// A window to create and edit multiple camera paths. Paths are read from a json
// file when created and saved when destroyed.
class CameraPathsElement : public nvvkhl::IAppElement
{
public:
  static constexpr const char* WindowName                       = "Camera Paths";
  static constexpr const char* PathsFilename                    = "camera_paths.json";
  CameraPathsElement(const CameraPathsElement& other)           = delete;
  CameraPathsElement operator=(const CameraPathsElement& other) = delete;
  CameraPathsElement();
  ~CameraPathsElement();
  bool saveFrames() const { return m_saveFrames && m_cameraAnimating; }
  int  animationFrame() const { return m_cameraAnimationFrame; }

private:
  using time_point = std::chrono::high_resolution_clock::time_point;
  using duration   = std::chrono::high_resolution_clock::duration;
  virtual void                          onUIRender() override;
  virtual void                          onRender(VkCommandBuffer) override;
  virtual void                          onUIMenu() override;
  int                                   m_cameraPathIndex = -1;
  std::vector<camera_paths::CameraPath> m_cameraPaths;
  int                                   m_cameraPositionIndex  = -1;
  int                                   m_cameraAnimationFrame = 0;
  bool                                  m_saveFrames           = false;
  bool                                  m_cameraAnimating      = false;
  time_point                            m_cameraAnimateLastFrame;
  duration                              m_cameraAnimatePosition = duration::zero();
  bool                                  m_showWindow            = false;
};
