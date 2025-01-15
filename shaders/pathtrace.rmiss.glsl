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

#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : enable

#include "dh_bindings.h"
#include "nvvkhl/shaders/dh_sky.h"
#include "pathtrace_device_host.h"
#include "pathtrace_payload.h"

layout(location = 0) rayPayloadInEXT HitPayload payload;

layout(set = 0, binding = BRtSkyParam) uniform SkyInfo_
{
  SimpleSkyParameters skyInfo;
};


void main()
{
  SimpleSkyParameters p = skyInfo;
  if(payload.depth > 0)
    p.lightRadiance = vec3(0.0);  // Sunlight is added in the closest hit shader
  vec3 sky_color = evalSimpleSky(p, gl_WorldRayDirectionEXT);
  payload.radiance += payload.transmittance * sky_color;
  payload.depth = PATHTRACE_MAX_RECURSION_DEPTH;  // Stop tracing
}
