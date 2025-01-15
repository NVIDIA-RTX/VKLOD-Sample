/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef PATHTRACE_DEVICE_HOST_H
#define PATHTRACE_DEVICE_HOST_H

#include "shaders_frame_params.h"
#include "shaders_glm.h"
#include "shaders_scene.h"

#define PATHTRACE_MAX_RECURSION_DEPTH 4

#ifdef __cplusplus
namespace shaders {
#endif  // __cplusplus

struct PathtraceConfig
{
  int32_t lodVisualization;
  int32_t sampleCountPixel;
  int32_t sampleCountAO;
  int32_t maxDepth;
  float   aoRadius;
  int32_t pathtrace;
};

struct PathtraceConstant
{
  DEVICE_ADDRESS(Instance) instancesAddress;
  DEVICE_ADDRESS(Mesh) meshesAddress;
  PathtraceConfig config;
  uint32_t        frame;
  float           errorOverDistanceThreshold;
};

#ifdef __cplusplus
}  // namespace shaders
#endif

#endif  // PATHTRACE_DEVICE_HOST_H
