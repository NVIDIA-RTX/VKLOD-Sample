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

#ifndef RASTERIZE_DEVICE_HOST_H
#define RASTERIZE_DEVICE_HOST_H

#include "shaders_frame_params.h"
#include "shaders_glm.h"
#include "shaders_scene.h"

#ifdef __cplusplus
namespace shaders {
#endif  // __cplusplus

const int BRasterFrameInfo = 0;
const int BRasterConstants = 1;
const int BRasterSkyParams = 2;
const int BRasterTextures  = 3;

struct RasterizeConfig
{
  int32_t lodVisualization;
};

struct RasterizeConstants
{
  DEVICE_ADDRESS(Instance) instancesAddress;
  DEVICE_ADDRESS(Mesh) meshesAddress;
  DEVICE_ADDRESS(DrawCluster) drawClustersAddress;
  DEVICE_ADDRESS(DrawStats) drawStatsAddress;
  RasterizeConfig config;
  uint32_t        frame;
  float           errorOverDistanceThreshold;
};

#ifdef __cplusplus
}  // namespace shaders
#else
DECL_BUFFER_REF(DrawClusterArray, DrawCluster)
#endif

#endif  // RASTERIZE_DEVICE_HOST_H
