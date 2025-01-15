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
#include "nvvkhl/shaders/constants.h"
#include "nvvkhl/shaders/random.h"
#include "pathtrace_device_host.h"
#include "pathtrace_payload.h"

// clang-format off
layout(location = 0) rayPayloadEXT HitPayload payload;

layout(set = 0, binding = BRtTlas) uniform accelerationStructureEXT topLevelAS;
layout(set = 0, binding = BRtOutImage, rgba32f) uniform image2D image;
layout(set = 0, binding = BRtFrameInfo) uniform FrameParams_ { FrameParams frameInfo; };
// clang-format on

layout(push_constant) uniform RtxPushConstant_
{
  PathtraceConstant pc;
};


void main()
{
  uint seed = xxhash32(uvec3(gl_LaunchIDEXT.xy, pc.config.pathtrace != 0 ? pc.frame : 0));

  int32_t sampleCount   = pc.config.sampleCountPixel;
  vec3    sampleAverage = vec3(0);
  for(int sampleIndex = 0; sampleIndex < sampleCount; ++sampleIndex)
  {
    const vec2 jitter     = vec2(rand(seed), rand(seed));
    const vec2 pixelCoord = vec2(gl_LaunchIDEXT.xy + jitter) / vec2(gl_LaunchSizeEXT.xy);
    const vec3 esTarget   = vec3(frameInfo.projInv * vec4(pixelCoord * 2.0 - 1.0, 0.0, 1.0));
    const vec3 wsPos      = vec3(frameInfo.viewInv * vec4(0.0, 0.0, 0.0, 1.0));
    const vec3 wsDir      = normalize(mat3(frameInfo.viewInv) * esTarget);

    payload = initPayload(wsPos, wsDir, seed);
    do
    {
      const uint rayFlags = 0;        // e.g. gl_RayFlagsCullBackFacingTrianglesEXT;
      traceRayEXT(topLevelAS,         // acceleration structure
                  rayFlags,           // rayFlags
                  0xFF,               // cullMask
                  0,                  // sbtRecordOffset
                  0,                  // sbtRecordStride
                  0,                  // missIndex
                  payload.origin,     // ray origin
                  0.0,                // ray t_min
                  payload.direction,  // ray direction
                  INFINITE,           // ray t_max
                  0                   // payload (location = 0)
      );
    } while(pc.config.pathtrace != 0 && payload.depth < pc.config.maxDepth
            && payload.depth < PATHTRACE_MAX_RECURSION_DEPTH && any(greaterThan(payload.transmittance, vec3(0.01))));

    // Avoid fireflies
    payload.radiance = min(payload.radiance, vec3(10.0));

    // Accumulate
    sampleAverage += payload.radiance;

    // Keep sampling...
    seed = payload.seed;
  }
  sampleAverage /= sampleCount;

  const float exposure = 1.0;
  sampleAverage *= exposure;

  float gamma = 2.2;
  if(pc.config.pathtrace != 0)
  {
    // Accumulate result into the pixel
    vec3 pixelColor = imageLoad(image, ivec2(gl_LaunchIDEXT.xy)).xyz;
    pixelColor      = pow(pixelColor, vec3(gamma));
    if(pc.frame == 0)
      pixelColor = sampleAverage;  // just in case NaNs exist
    else
      pixelColor = mix(pixelColor, sampleAverage, 1.0F / float(pc.frame + 1));
    pixelColor = pow(pixelColor, vec3(1.0 / gamma));
    imageStore(image, ivec2(gl_LaunchIDEXT.xy), vec4(pixelColor, 1.0F));
  }
  else
  {
    sampleAverage = pow(sampleAverage, vec3(1.0 / gamma));
    imageStore(image, ivec2(gl_LaunchIDEXT.xy), vec4(sampleAverage, 1.0F));
  }
}
