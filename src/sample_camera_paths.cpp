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

#include <fstream>
#include <numeric>
#include <ranges>
#include <sample_camera_paths.hpp>

namespace camera_paths {

Keyframe Keyframe::fromJSON(const nlohmann::json& keyframe)
{
  return {
      .position      = json_get<glm::vec3>(keyframe["position"]),
      .rotation      = json_get<glm::quat>(keyframe["rotation"]),
      .distance      = keyframe["distance"].get<float>(),
      .fov           = keyframe["fov"].get<float>(),
      .durationScale = keyframe["duration_scale"].get<float>(),
  };
}

Keyframe Keyframe::fromCamera(const nvh::CameraManipulator& camera)
{
  glm::vec3 direction = camera.getCenter() - camera.getEye();
  return {
      .position = camera.getEye(),
      .rotation = glm::quatLookAtRH(glm::normalize(direction), camera.getUp()),
      .distance = glm::length(direction),
      .fov      = const_cast<nvh::CameraManipulator&>(camera).getFov(),
  };
}

void Keyframe::toCamera(nvh::CameraManipulator& camera)
{
  glm::vec3 eye     = position;
  glm::vec3 forward = rotation * glm::vec3(0.0f, 0.0f, -1.0f);
  glm::vec3 center  = eye + forward * distance;
  glm::vec3 up      = rotation * glm::vec3(0.0f, 1.0f, 0.0f);
  // Use +Y = up if it's close enough to keep the checkbox in the UI checked
  if(fabs(glm::dot(glm::cross(up, forward), glm::vec3(0.0f, 1.0f, 0.0f))) < 0.0001f)
    up = glm::vec3(0.0f, 1.0f, 0.0f);
  printf("%f\n", fabs(glm::dot(glm::cross(up, forward), glm::vec3(0.0f, 1.0f, 0.0f))));
  camera.setLookat(eye, center, up, true);
  camera.setFov(fov);
}
nlohmann::json Keyframe::toJSON() const
{
  return {
      {"position", {position.x, position.y, position.z}},
      {"rotation", {rotation.x, rotation.y, rotation.z, rotation.w}},
      {"distance", distance},
      {"fov", fov},
      {"duration_scale", durationScale},
  };
}

CameraPath::CameraPath(const char* newName)
    : CameraPath(std::string(newName))
{
}

CameraPath::CameraPath(const std::string& newName)
    : name(newName)
    , keyframes({Keyframe::fromCamera(nvh::CameraManipulator::Singleton())})
    , selectedIndex(0)
{
}

CameraPath::CameraPath(const nlohmann::json& ipath)
{
  name     = ipath["name"].get<std::string>();
  duration = ipath["duration"].get<float>();
  for(const auto& keyframe : ipath["keyframes"])
    keyframes.push_back(Keyframe::fromJSON(keyframe));
}

nlohmann::json CameraPath::toJSON() const
{
  nlohmann::json jkeyframes;
  for(const auto& cameraPosition : keyframes)
  {
    jkeyframes.push_back(cameraPosition.toJSON());
  }
  return nlohmann::json{
      {"name", name},
      {"duration", duration},
      {"keyframes", jkeyframes},
  };
}

Keyframe CameraPath::interpolate(float t)
{
  assert(keyframes.size() >= 2);

  // Find the current set of keyframes
  size_t             keyframeCount = keyframes.size();
  std::vector<float> runningDurationScale(keyframeCount);
  auto               durationScales = keyframes | std::views::transform(&Keyframe::durationScale);
  std::exclusive_scan(durationScales.begin(), durationScales.end(), runningDurationScale.begin(), 0.0f);
  float  durationScaleT = t * runningDurationScale.back();
  size_t currentIndex =
      size_t(std::distance(runningDurationScale.begin(), std::ranges::upper_bound(runningDurationScale, durationScaleT) - 1));
  float localT = (durationScaleT - runningDurationScale[currentIndex]) / keyframes[currentIndex].durationScale;

  // Clamp to the array bounds and give them names
  const Keyframe& kBefore = keyframes[currentIndex == 0 ? 0 : currentIndex - 1];
  const Keyframe& kStart  = keyframes[currentIndex];
  const Keyframe& kEnd    = keyframes[std::min<size_t>(currentIndex + 1, keyframeCount - 1)];
  const Keyframe& kAfter  = keyframes[std::min<size_t>(currentIndex + 2, keyframeCount - 1)];

  // Interpolate keyframe positions
  glm::vec3 before = kBefore.position;
  glm::vec3 start  = kStart.position;
  glm::vec3 end    = kEnd.position;
  glm::vec3 after  = kAfter.position;

  // Max. dist should be at most half way between start and end, at which point
  // the middle control points could overlap
  float cpDist = cpScale * 0.5f;

  // Interpolate position using a piecewise Bezier spline
  // Optionally limit the control point distance
#if 1
  auto      makelength  = [](glm::vec3 v, float l) { return v * (l / std::max(0.0001f, glm::length(v))); };
  float     cpStartSize = std::min(glm::length(start - before) * cpDist, glm::length(end - start) * cpDist);
  float     cpEndSize   = std::min(glm::length(end - after) * cpDist, glm::length(end - start) * cpDist);
  glm::vec3 cpStart     = start + makelength(end - before, cpStartSize);
  glm::vec3 cpEnd       = end + makelength(start - after, cpEndSize);
#else
  glm::vec3 cpStart = start + (end - before) * (cpDist * 0.5f);
  glm::vec3 cpEnd   = end + (start - after) * (cpDist * 0.5f);
#endif
  auto bezierInterp = [](const glm::vec3& p0, const glm::vec3& p1, const glm::vec3& p2, const glm::vec3& p3, float t) {
    float u   = 1.0f - t;
    float tt  = t * t;
    float uu  = u * u;
    float uuu = uu * u;
    float ttt = tt * t;
    return (uuu * p0) + (3.0f * uu * t * p1) + (3.0f * u * tt * p2) + (ttt * p3);
  };
  glm::vec3 interpPosition = bezierInterp(start, cpStart, cpEnd, end, localT);

  // Interpolate keyframe rotations
  glm::quat beforeQ = kBefore.rotation;
  glm::quat startQ  = kStart.rotation;
  glm::quat endQ    = kEnd.rotation;
  glm::quat afterQ  = kAfter.rotation;

  // Interpolate rotations using a piecewise Bezier spline
  // Optionally limit the control point angular distance
#if 1
  // TODO: way too much normalization happening here
  auto angleDiffQ = [](glm::quat q1, glm::quat q2) {
    if(glm::dot(q1, q2) < 0.0f)
      q2 = -q2;
    return fabs(glm::angle(glm::normalize(q2 * glm::conjugate(q1))));
  };
  float     cpStartAngle      = std::min(angleDiffQ(startQ, beforeQ) * cpDist, angleDiffQ(endQ, startQ) * cpDist);
  float     cpEndAngle        = std::min(angleDiffQ(endQ, afterQ) * cpDist, angleDiffQ(endQ, startQ) * cpDist);
  float     cpStartAngleRatio = cpStartAngle / std::max(0.0001f, angleDiffQ(beforeQ, endQ));
  float     cpEndAngleRatio   = cpEndAngle / std::max(0.0001f, angleDiffQ(afterQ, startQ));
  glm::quat cpStartQ = glm::normalize(glm::slerp(startQ, startQ * endQ * glm::conjugate(beforeQ), cpStartAngleRatio));
  glm::quat cpEndQ   = glm::normalize(glm::slerp(endQ, endQ * startQ * glm::conjugate(afterQ), cpEndAngleRatio));
#else
  glm::quat cpStartQ = glm::normalize(glm::slerp(startQ, startQ * endQ * glm::conjugate(beforeQ), (cpDist * 0.5f)));
  glm::quat cpEndQ   = glm::normalize(glm::slerp(endQ, endQ * startQ * glm::conjugate(afterQ), (cpDist * 0.5f)));
#endif
  auto bezierInterpRecursive = [](const auto& p0, const auto& p1, const auto& p2, const auto& p3, float t, auto lerpFunc) {
    auto p01  = glm::normalize(lerpFunc(p0, p1, t));
    auto p12  = glm::normalize(lerpFunc(p1, p2, t));
    auto p23  = glm::normalize(lerpFunc(p2, p3, t));
    auto p012 = glm::normalize(lerpFunc(p01, p12, t));
    auto p123 = glm::normalize(lerpFunc(p12, p23, t));
    return glm::normalize(lerpFunc(p012, p123, t));
  };
  glm::quat interpQuat = bezierInterpRecursive(startQ, cpStartQ, cpEndQ, endQ, localT, glm::slerp<float, glm::defaultp>);

  // Reuse the keyframe struct to return interpolated values
  return {
      .position = interpPosition,
      .rotation = interpQuat,
      .distance = glm::mix(kStart.distance, kEnd.distance, localT),
      .fov      = glm::mix(kStart.fov, kEnd.fov, localT),
  };
}

void CameraPath::onUIRender()
{
  // Edit the camera path name and other properties
  auto resize = [](ImGuiInputTextCallbackData* data) -> int {
    if(data->EventFlag == ImGuiInputTextFlags_CallbackResize)
    {
      auto& str = *static_cast<std::string*>(data->UserData);
      str.resize(data->BufTextLen);
      data->Buf = str.data();
    }
    return 0;
  };
  ImGui::InputText("Name", name.data(), name.capacity() + 1, ImGuiInputTextFlags_CallbackResize, resize, &name);
  ImGui::SliderFloat("Animation Duration", &duration, 0.1f, 60.0f, "%.1f seconds");
  if(ImGui::SliderFloat("Jump to position", &m_seekPosition, 0.0f, 1.0f))
  {
    interpolate(m_seekPosition).toCamera(nvh::CameraManipulator::Singleton());
  }
  if(ImGui::SliderFloat("Bezier Control Point Distance", &cpScale, 0.0f, 1.0f))
  {
    interpolate(m_seekPosition).toCamera(nvh::CameraManipulator::Singleton());
  }

  // Display the list of camera positions
  for(size_t i = 0; i < keyframes.size(); ++i)
  {
    ImGui::PushID(static_cast<int>(i));
    ImGui::SetNextItemAllowOverlap();
    if(ImGui::Selectable(("P " + glm::to_string(keyframes[i].position)).c_str(), selectedIndex == static_cast<int>(i)))
    {
      selectedIndex = static_cast<int>(i);
    }

    // Allow reordering
    if(i > 0)
    {
      ImGui::SameLine();
      if(ImGui::Button("Up"))
      {
        std::swap(keyframes[i], keyframes[i - 1]);
        if(selectedIndex == static_cast<int>(i))
          selectedIndex--;
        else if(selectedIndex == static_cast<int>(i - 1))
          selectedIndex++;
      }
    }
    if(i < keyframes.size() - 1)
    {
      ImGui::SameLine();
      if(ImGui::Button("Down"))
      {
        std::swap(keyframes[i], keyframes[i + 1]);
        if(selectedIndex == static_cast<int>(i))
          selectedIndex++;
        else if(selectedIndex == static_cast<int>(i + 1))
          selectedIndex--;
      }
    }
    ImGui::PopID();
  }

  // Add a new camera position
  if(ImGui::Button("Add Camera Position"))
  {
    keyframes.insert(keyframes.begin() + selectedIndex + 1, Keyframe::fromCamera(nvh::CameraManipulator::Singleton()));
    selectedIndex += 1;
  }
  if(ImGui::Button("Delete Camera Position"))
  {
    if(selectedIndex >= 0 && selectedIndex < static_cast<int>(keyframes.size()))
    {
      keyframes.erase(keyframes.begin() + selectedIndex);
      if(selectedIndex >= static_cast<int>(keyframes.size()))
      {
        selectedIndex = static_cast<int>(keyframes.size()) - 1;
      }
    }
  }

  // Load and write the current camera manipulator to the selected camera position
  if(selectedIndex >= 0 && selectedIndex < static_cast<int>(keyframes.size()))
  {
    if(ImGui::Button("Save"))
      keyframes[selectedIndex] = Keyframe::fromCamera(nvh::CameraManipulator::Singleton());
    ImGui::SameLine();
    if(ImGui::Button("Load to Camera"))
      keyframes[selectedIndex].toCamera(nvh::CameraManipulator::Singleton());
  }

  // Display details of the selected camera position
  if(selectedIndex >= 0 && selectedIndex < static_cast<int>(keyframes.size()))
  {
    Keyframe& selectedCamera = keyframes[selectedIndex];
    ImGui::Text("Selected Camera Position:");
    ImGui::InputFloat3("Position", &selectedCamera.position[0]);
    ImGui::InputFloat4("Rotation", &selectedCamera.rotation[0]);
    ImGui::InputFloat("Duration Scale", &selectedCamera.durationScale);
  }
}

}  // namespace camera_paths

CameraPathsElement::CameraPathsElement()
{
  if(std::filesystem::exists(PathsFilename))
    for(auto ipath : nlohmann::json::parse(std::ifstream{PathsFilename}))
      m_cameraPaths.emplace_back(ipath);
  m_cameraPathIndex = m_cameraPaths.empty() ? -1 : 0;
}

CameraPathsElement::~CameraPathsElement()
{
  nlohmann::json outputJson;
  for(const auto& path : m_cameraPaths)
    outputJson.push_back(path.toJSON());
  std::ofstream outFile(PathsFilename);
  outFile << outputJson.dump(4);  // Write JSON with 4-space indentation
  outFile.close();
}

void CameraPathsElement::onUIRender()
{
  if(!m_showWindow)
    return;

  // Opening the window
  if(!ImGui::Begin(WindowName, &m_showWindow))
  {
    ImGui::End();
    return;
  }

  // Dropdown for selecting item
  if(ImGui::BeginCombo("Item", m_cameraPathIndex == -1 ? "None" : m_cameraPaths[m_cameraPathIndex].name.c_str()))
  {
    for(int i = 0; i < (int)m_cameraPaths.size(); i++)
    {
      bool isSelected = (i == m_cameraPathIndex);
      if(ImGui::Selectable(m_cameraPaths[i].name.c_str(), isSelected))
      {
        m_cameraPathIndex = i;
      }
      if(isSelected)
      {
        ImGui::SetItemDefaultFocus();
      }
    }
    ImGui::EndCombo();
  }

  // Button to create a new camera path
  if(ImGui::Button("Create New Path"))
  {
    m_cameraPaths.push_back(camera_paths::CameraPath("My Camera Path"));
    m_cameraPathIndex = static_cast<int>(m_cameraPaths.size()) - 1;  // Select the newly created path
  }

  // If an item is selected, display its edit fields
  if(m_cameraPathIndex != -1)
  {
    // Add checkbox for saving frames
    ImGui::Checkbox("Save Frames", &m_saveFrames);

    ImGui::BeginDisabled(m_cameraPaths[m_cameraPathIndex].keyframes.size() < 2);
    // Add buttons to start/stop camera animation
    if(ImGui::Button("Play"))
    {
      m_cameraAnimating        = true;
      m_cameraAnimateLastFrame = std::chrono::high_resolution_clock::now();
      m_cameraAnimatePosition  = duration::zero();
      m_cameraAnimationFrame   = 0;
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    if(ImGui::Button("Stop"))
    {
      m_cameraAnimating = false;
    }

    // Display the keyframe editor
    m_cameraPaths[m_cameraPathIndex].onUIRender();
  }

  ImGui::End();  // m_showWindow
}

void CameraPathsElement::onRender(VkCommandBuffer)
{
  if(m_cameraAnimating && m_cameraPathIndex != -1)
  {
    // Compute the current camera position in the animation
#if 0
      auto now = std::chrono::high_resolution_clock::now();
      m_cameraAnimatePosition += now - m_cameraAnimateLastFrame;
      m_cameraAnimateLastFrame = now;
#else
    m_cameraAnimatePosition += std::chrono::microseconds(16666);
#endif

    // Smoothly interpolate between camera positions and rotations
    m_cameraAnimationFrame++;
    float animationTime = std::chrono::duration<float>(m_cameraAnimatePosition).count();
    float t             = glm::clamp(animationTime / m_cameraPaths[m_cameraPathIndex].duration, 0.0f, 1.0f);
    if(animationTime > m_cameraPaths[m_cameraPathIndex].duration)
      m_cameraAnimating = false;

    m_cameraPaths[m_cameraPathIndex].interpolate(t).toCamera(nvh::CameraManipulator::Singleton());
  }
}

void CameraPathsElement::onUIMenu()
{
  if(ImGui::BeginMenu("View"))
  {
    ImGui::MenuItem(WindowName, "", &m_showWindow);
    ImGui::EndMenu();
  }
}
