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

#ifndef SHADERS_BUFFER_REF_H
#define SHADERS_BUFFER_REF_H

#ifdef __cplusplus
#define DEVICE_ADDRESS(Type) vkobj::DeviceAddress<Type>
#else
#define DEVICE_ADDRESS(Type) uint64_t
#endif

// Utility macros for GL_EXT_buffer_reference2
#ifdef __cplusplus
namespace shaders {
#include <glm/glm.hpp>
#define DECL_BUFFER_REF_BASE(RefName, Keywords, Type, Align)                                                           \
  namespace {                                                                                                          \
  using namespace glm;                                                                                                 \
  static_assert(alignof(Type) < Align);                                                                                \
  }
#define DECL_BUFFER_REF_SINGLE_BASE(RefName, Keywords, Type, Align)                                                    \
  namespace {                                                                                                          \
  using namespace glm;                                                                                                 \
  static_assert(alignof(Type) < Align);                                                                                \
  }
#else
#define DECL_BUFFER_REF_BASE(RefName, Keywords, Type, Align)                                                           \
  layout(buffer_reference, scalar, buffer_reference_align = Align) Keywords buffer RefName                             \
  {                                                                                                                    \
    Type array[];                                                                                                      \
  };
#define DECL_BUFFER_REF_SINGLE_BASE(RefName, Keywords, Type, Align)                                                    \
  layout(buffer_reference, scalar, buffer_reference_align = Align) Keywords buffer RefName                             \
  {                                                                                                                    \
    Type d;                                                                                                            \
  };
#define sizeof(Type) (uint64_t(Type(uint64_t(0)) + 1))
#endif

#define DECL_BUFFER_REF(RefName, Type) DECL_BUFFER_REF_BASE(RefName, readonly, Type, 16)
#define DECL_MUTABLE_BUFFER_REF(RefName, Type) DECL_BUFFER_REF_BASE(RefName, , Type, 16)
DECL_BUFFER_REF(FloatArray, float);
DECL_BUFFER_REF(Vec2Array, vec2);
DECL_BUFFER_REF(Vec3Array, vec3);
DECL_BUFFER_REF(Vec4Array, vec4);
DECL_BUFFER_REF(UVec2Array, uvec2);
DECL_BUFFER_REF(UVec3Array, uvec3);
DECL_BUFFER_REF(U8Vec3Array, u8vec3);
DECL_BUFFER_REF(UVec4Array, uvec4);
DECL_BUFFER_REF(Uint8Array, uint8_t);
DECL_BUFFER_REF(UintArray, uint32_t);
DECL_BUFFER_REF(Uint32Array, uint32_t);
DECL_BUFFER_REF(Uint64Array, uint64_t);
DECL_MUTABLE_BUFFER_REF(MutUintArray, uint32_t);
DECL_MUTABLE_BUFFER_REF(MutUint32Array, uint32_t);
DECL_MUTABLE_BUFFER_REF(MutUint64Array, uint64_t);

#ifdef __cplusplus
}  // namespace shaders
#endif

#endif  // SHADERS_BUFFER_REF_H
