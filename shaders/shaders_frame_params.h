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

#ifndef FRAME_PARAMS_H
#define FRAME_PARAMS_H

#ifdef __cplusplus
namespace shaders {
#endif  // __cplusplus

struct FrameParams
{
  mat4  proj;
  mat4  view;
  mat4  projInv;
  mat4  viewInv;
  mat4  viewProj;
  vec3  camPos;
  float fogHeightOffset;
  float fogHeightScale;
};

#ifdef __cplusplus
}  // namespace shaders
#endif

#endif  // FRAME_PARAMS_H
