/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

// TODO: could replace most of this whole file with inlining
// std::span(cgltf.ptr, cgltf.count)

#include <cgltf.h>
#include <filesystem>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <memory>
#include <meshops_array_view.h>
#include <nvh/filemapping.hpp>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string_view>
#include <type_traits>

namespace fs = std::filesystem;

class MappedFile
{
public:
  MappedFile(const fs::path& path)
      : m_mapping(std::make_unique<nvh::FileReadMapping>())
  {
    m_mapping->open(path.string().c_str());
    if(!m_mapping->valid())
      throw std::runtime_error("Failed to map file '" + path.string() + "'");
  }

  const void* data() const { return m_mapping->data(); }
  size_t      size() const { return m_mapping->size(); }

private:
  // FileReadMapping is not safe to move
  std::unique_ptr<nvh::FileReadMapping> m_mapping;
};

static const char* getError(cgltf_result result, cgltf_data* data)
{
  // clang-format off
  switch(result)
  {
    case cgltf_result_file_not_found: return data ? "resource not found" : "file not found";
    case cgltf_result_io_error: return "I/O error";
    case cgltf_result_invalid_json: return "invalid JSON";
    case cgltf_result_invalid_gltf: return "invalid GLTF";
    case cgltf_result_out_of_memory: return "out of memory";
    case cgltf_result_legacy_gltf: return "legacy GLTF";
    case cgltf_result_data_too_short: return data ? "buffer too short" : "not a GLTF file";
    case cgltf_result_unknown_format: return data ? "unknown resource format" : "not a GLTF file";
    default: break;
  }
  // clang-format on
  return "unknown error";
}

using UniqueCgltfData = std::unique_ptr<cgltf_data, decltype(&cgltf_free)>;
inline UniqueCgltfData makeUniqueCgltfData(const void* inputData, size_t inputSize, const cgltf_options& options)
{
  cgltf_data*  data   = nullptr;
  cgltf_result result = cgltf_parse(&options, inputData, inputSize, &data);
  if(result != cgltf_result_success)
  {
    throw std::runtime_error(getError(result, data));
  }
  return UniqueCgltfData(data, cgltf_free);
}

// Adaptors for cgltf_accessor
// clang-format off
inline constexpr const char* cgltfTypeName(cgltf_component_type component_type, cgltf_type type)
{
  #define ROW(c) { "vec2_" c, "vec3_" c, "vec4_" c, "mat2_" c, "mat3_" c, "mat4_" c, "scalar_" c }
  constexpr const char* table[8][7] ={ ROW("i8"), ROW("ui8"), ROW("i16"), ROW("ui16"), ROW("ui32"), ROW("f32") };

  int typeIndex = 0;
  switch(type)
  {
    case cgltf_type_vec2: typeIndex = 0; break;
    case cgltf_type_vec3: typeIndex = 1; break;
    case cgltf_type_vec4: typeIndex = 2; break;
    case cgltf_type_mat2: typeIndex = 3; break;
    case cgltf_type_mat3: typeIndex = 4; break;
    case cgltf_type_mat4: typeIndex = 5; break;
    case cgltf_type_scalar: typeIndex = 6; break;
    default: return "unknown";
  }

  switch(component_type)
  {
    case cgltf_component_type_r_8: return table[0][typeIndex];
    case cgltf_component_type_r_8u: return table[1][typeIndex];
    case cgltf_component_type_r_16: return table[2][typeIndex];
    case cgltf_component_type_r_16u: return table[3][typeIndex];
    case cgltf_component_type_r_32u: return table[4][typeIndex];
    case cgltf_component_type_r_32f: return table[5][typeIndex];
    default: return "unknown";
  }
}

template <class T> struct cgltf_type_traits;
template <> struct cgltf_type_traits<int8_t>     { static constexpr cgltf_component_type component_type = cgltf_component_type_r_8;   static constexpr cgltf_type type = cgltf_type_scalar; static constexpr const char* name = cgltfTypeName(component_type, type); };
template <> struct cgltf_type_traits<uint8_t>    { static constexpr cgltf_component_type component_type = cgltf_component_type_r_8u;  static constexpr cgltf_type type = cgltf_type_scalar; static constexpr const char* name = cgltfTypeName(component_type, type); };
template <> struct cgltf_type_traits<int16_t>    { static constexpr cgltf_component_type component_type = cgltf_component_type_r_16;  static constexpr cgltf_type type = cgltf_type_scalar; static constexpr const char* name = cgltfTypeName(component_type, type); };
template <> struct cgltf_type_traits<uint16_t>   { static constexpr cgltf_component_type component_type = cgltf_component_type_r_16u; static constexpr cgltf_type type = cgltf_type_scalar; static constexpr const char* name = cgltfTypeName(component_type, type); };
template <> struct cgltf_type_traits<uint32_t>   { static constexpr cgltf_component_type component_type = cgltf_component_type_r_32u; static constexpr cgltf_type type = cgltf_type_scalar; static constexpr const char* name = cgltfTypeName(component_type, type); };
template <> struct cgltf_type_traits<float>      { static constexpr cgltf_component_type component_type = cgltf_component_type_r_32f; static constexpr cgltf_type type = cgltf_type_scalar; static constexpr const char* name = cgltfTypeName(component_type, type); };
template <> struct cgltf_type_traits<glm::vec2>  { static constexpr cgltf_component_type component_type = cgltf_component_type_r_32f; static constexpr cgltf_type type = cgltf_type_vec2;   static constexpr const char* name = cgltfTypeName(component_type, type); };
template <> struct cgltf_type_traits<glm::uvec2> { static constexpr cgltf_component_type component_type = cgltf_component_type_r_32u; static constexpr cgltf_type type = cgltf_type_vec2;   static constexpr const char* name = cgltfTypeName(component_type, type); };
template <> struct cgltf_type_traits<glm::vec3>  { static constexpr cgltf_component_type component_type = cgltf_component_type_r_32f; static constexpr cgltf_type type = cgltf_type_vec3;   static constexpr const char* name = cgltfTypeName(component_type, type); };
template <> struct cgltf_type_traits<glm::uvec3> { static constexpr cgltf_component_type component_type = cgltf_component_type_r_32u; static constexpr cgltf_type type = cgltf_type_vec3;   static constexpr const char* name = cgltfTypeName(component_type, type); };
template <> struct cgltf_type_traits<glm::vec4>  { static constexpr cgltf_component_type component_type = cgltf_component_type_r_32f; static constexpr cgltf_type type = cgltf_type_vec4;   static constexpr const char* name = cgltfTypeName(component_type, type); };
template <> struct cgltf_type_traits<glm::uvec4> { static constexpr cgltf_component_type component_type = cgltf_component_type_r_32u; static constexpr cgltf_type type = cgltf_type_vec4;   static constexpr const char* name = cgltfTypeName(component_type, type); };

template <cgltf_component_type component_type, cgltf_type type> struct cgltf_type_traits_inv;
template <> struct cgltf_type_traits_inv<cgltf_component_type_r_8,   cgltf_type_scalar> { using element_type = int8_t; };
template <> struct cgltf_type_traits_inv<cgltf_component_type_r_8u,  cgltf_type_scalar> { using element_type = uint8_t; };
template <> struct cgltf_type_traits_inv<cgltf_component_type_r_16,  cgltf_type_scalar> { using element_type = int16_t; };
template <> struct cgltf_type_traits_inv<cgltf_component_type_r_16u, cgltf_type_scalar> { using element_type = uint16_t; };
template <> struct cgltf_type_traits_inv<cgltf_component_type_r_32u, cgltf_type_scalar> { using element_type = uint32_t; };
template <> struct cgltf_type_traits_inv<cgltf_component_type_r_32f, cgltf_type_scalar> { using element_type = float; };
template <> struct cgltf_type_traits_inv<cgltf_component_type_r_32f, cgltf_type_vec2>   { using element_type = glm::vec2; };
template <> struct cgltf_type_traits_inv<cgltf_component_type_r_32u, cgltf_type_vec2>   { using element_type = glm::uvec2; };
template <> struct cgltf_type_traits_inv<cgltf_component_type_r_32f, cgltf_type_vec3>   { using element_type = glm::vec3; };
template <> struct cgltf_type_traits_inv<cgltf_component_type_r_32u, cgltf_type_vec3>   { using element_type = glm::uvec3; };
template <> struct cgltf_type_traits_inv<cgltf_component_type_r_32f, cgltf_type_vec4>   { using element_type = glm::vec4; };
template <> struct cgltf_type_traits_inv<cgltf_component_type_r_32u, cgltf_type_vec4>   { using element_type = glm::uvec4; };
// clang-format on

// From nvpro_core
inline glm::mat4 getLocalMatrix(const cgltf_node& node)
{
  glm::mat4 translation{1.0f};
  glm::mat4 scale{1.0f};
  glm::mat4 rotation{1.0f};
  glm::mat4 matrix{1.0f};

  if(node.has_translation)
    translation = glm::translate(translation, glm::vec3(node.translation[0], node.translation[1], node.translation[2]));
  if(node.has_scale)
    scale = glm::scale(scale, glm::vec3(node.scale[0], node.scale[1], node.scale[2]));
  if(node.has_rotation)
  {
    glm::quat quat{};
    quat[0]  = node.rotation[0];
    quat[1]  = node.rotation[1];
    quat[2]  = node.rotation[2];
    quat[3]  = node.rotation[3];
    rotation = glm::mat4_cast(quat);
  }
  if(node.has_matrix)
  {
    std::ranges::copy(node.matrix, glm::value_ptr(matrix));
  }
  return translation * rotation * scale * matrix;
}

template <class CgltfType, class cgltf_type>
auto cgltf_wrap(const std::span<cgltf_type>& span)
{
  return span | std::ranges::views::transform([](cgltf_type& obj) {
           if constexpr(std::is_pointer_v<cgltf_type> && !std::is_constructible_v<CgltfType, cgltf_type>)
             return CgltfType(*obj);
           else
             return CgltfType(obj);
         });
}

template <class T>
class CgltfAccessor : public cgltf_accessor, public meshops::ArrayView<const T>
{
public:
  CgltfAccessor() = default;
  explicit CgltfAccessor(const cgltf_accessor& accessor)
      : cgltf_accessor(accessor)
      , meshops::ArrayView<const T>(reinterpret_cast<const T*>(data()), count, stride())
  {
    using dT = std::decay_t<T>;
    if(accessor.component_type != cgltf_type_traits<dT>::component_type || accessor.type != cgltf_type_traits<dT>::type)
      throw std::runtime_error(std::string("Invalid accessor. Got ") + cgltfTypeName(accessor.component_type, accessor.type)
                               + ", expecting " + cgltf_type_traits<T>::name);
  }

private:
  const void* data() const
  {
    return reinterpret_cast<std::byte*>(buffer_view->buffer->data) + buffer_view->offset + offset;
  }
  size_t                stride() const { return buffer_view->stride ? buffer_view->stride : base().stride; }
  const cgltf_accessor& base() const { return *this; }
};

class CgltfPrimitive : public cgltf_primitive
{
public:
  template <class T>
  CgltfAccessor<T> indices() const
  {
    return CgltfAccessor<T>(*base().indices);
  }
  //CgltfMaterial material() const { return {base().material}; }
  std::span<const cgltf_attribute>        attributes_c() const { return {base().attributes, attributes_count}; }
  std::span<const cgltf_material_mapping> mappings_c() const { return {base().mappings, mappings_count}; }
  std::span<const cgltf_morph_target>     targets_c() const { return {base().targets, targets_count}; }
  std::span<const cgltf_extension>        extensions_c() const { return {base().extensions, extensions_count}; }
  template <class T>
  std::optional<CgltfAccessor<T>> attribute(const cgltf_attribute_type& type) const
  {
    std::optional<CgltfAccessor<T>> result;
    auto                            attrs = attributes_c();
    auto attr = std::ranges::find_if(attrs, [&type](const auto& a) { return a.type == type; });
    if(attr != attrs.end())
      result.emplace(*attr->data);
    return result;
  }

private:
  const cgltf_primitive& base() const { return *this; }
};

class CgltfMesh : public cgltf_mesh
{
public:
  std::span<const cgltf_primitive> primitives_c() const { return {base().primitives, primitives_count}; }
  auto                             primitives() const { return cgltf_wrap<CgltfPrimitive>(primitives_c()); }
  std::span<const cgltf_float>     weights() const { return {base().weights, weights_count}; }
  std::span<char const* const>     target_names_c() const { return {base().target_names, target_names_count}; }
  auto                             target_names() const { return cgltf_wrap<std::string_view>(target_names_c()); }
  std::span<const cgltf_extension> extensions_c() const { return {base().extensions, extensions_count}; }

private:
  const cgltf_mesh& base() const { return *this; }
};

class CgltfNode : public cgltf_node
{
public:
  std::optional<CgltfNode> parent() { return base().parent ? std::optional<CgltfNode>{*base().parent} : std::nullopt; }
  std::optional<CgltfMesh> mesh() { return base().mesh ? std::optional<CgltfMesh>{*base().mesh} : std::nullopt; }
  std::span<cgltf_node const* const> children_c() const { return {base().children, base().children_count}; }
  auto                               children() const { return cgltf_wrap<CgltfNode>(children_c()); }
  std::span<const cgltf_float>       weights() const { return {base().weights, weights_count}; }
  std::span<const cgltf_extension>   extensions_c() const { return {base().extensions, extensions_count}; }
  glm::mat4                          transform() const { return getLocalMatrix(base()); }

  // Provide access to the original cgltf types
  const cgltf_node* operator->() const { return &base(); }

private:
  const cgltf_node& base() const { return *this; }
};

class CgltfScene : public cgltf_scene
{
public:
  std::span<cgltf_node const* const> nodes_c() const { return {base().nodes, base().nodes_count}; }
  auto                               nodes() const { return cgltf_wrap<CgltfNode>(nodes_c()); }
  std::span<const cgltf_extension>   extensions_c() const { return {base().extensions, extensions_count}; }

private:
  const cgltf_scene& base() const { return *this; }
};

class CgltfModel
{
public:
  CgltfModel(const fs::path& path, const cgltf_options& options = {})
      : m_mappedFile(path.string())
      , m_data(makeUniqueCgltfData(m_mappedFile.data(), m_mappedFile.size(), options))
  {
    // Duplicate cgltf_load_buffers() functionality but with file mapping
    if(buffers_c().size() && buffers_c()[0].data == nullptr && buffers_c()[0].uri == nullptr && m_data->bin != nullptr)
    {
      if(m_data->bin_size < buffers_c()[0].size)
      {
        throw std::runtime_error("data too short");  // ??
      }

      m_data->buffers[0].data             = const_cast<void*>(m_data->bin);  // DANGER: const_cast
      m_data->buffers[0].data_free_method = cgltf_data_free_method_none;
    }
    for(auto& buffer : std::span{m_data->buffers, m_data->buffers_count})
    {
      if(buffer.data)
        continue;
      if(std::string_view(buffer.uri).starts_with("data:"))
      {
        throw std::runtime_error("data uri not implemented");
      }
      m_secondaryFiles.emplace_back((path.parent_path() / buffer.uri).string());
      buffer.data = reinterpret_cast<char*>(const_cast<void*>(m_secondaryFiles.back().data()));  // DANGER: const_cast
      buffer.data_free_method = cgltf_data_free_method_none;
    }
  }
  const cgltf_data& operator&() const { return *m_data; }
  const cgltf_data* operator->() const { return &*m_data; }

  std::span<const cgltf_mesh>        meshes_c() const { return {m_data->meshes, m_data->meshes_count}; }
  auto                               meshes() const { return cgltf_wrap<CgltfMesh>(meshes_c()); }
  std::span<const cgltf_accessor>    accessors_c() const { return {m_data->accessors, m_data->accessors_count}; }
  std::span<const cgltf_buffer_view> buffer_views_c() const
  {
    return {m_data->buffer_views, m_data->buffer_views_count};
  }
  std::span<const cgltf_buffer>  buffers_c() const { return {m_data->buffers, m_data->buffers_count}; }
  std::span<const cgltf_image>   images_c() const { return {m_data->images, m_data->images_count}; }
  std::span<const cgltf_texture> textures_c() const { return {m_data->textures, m_data->textures_count}; }
  std::span<const cgltf_sampler> samplers_c() const { return {m_data->samplers, m_data->samplers_count}; }
  std::span<const cgltf_skin>    skins_c() const { return {m_data->skins, m_data->skins_count}; }
  std::span<const cgltf_camera>  cameras_c() const { return {m_data->cameras, m_data->cameras_count}; }
  std::span<const cgltf_light>   lights_c() const { return {m_data->lights, m_data->lights_count}; }
  std::span<const cgltf_node>    nodes_c() const { return {m_data->nodes, m_data->nodes_count}; }
  auto                           nodes() const { return cgltf_wrap<CgltfNode>(nodes_c()); }
  std::span<const cgltf_scene>   scenes_c() const { return {m_data->scenes, m_data->scenes_count}; }
  auto                           scenes() const { return cgltf_wrap<CgltfScene>(scenes_c()); }
  std::optional<CgltfScene>      scene() const
  {
    return m_data->scene ? std::optional<CgltfScene>{*m_data->scene} : std::nullopt;
  }
  std::span<const cgltf_animation> animations_c() const { return {m_data->animations, m_data->animations_count}; }
  std::span<const cgltf_material_variant> variants_c() const { return {m_data->variants, m_data->variants_count}; }
  std::span<const cgltf_extension>        data_extensions_c() const
  {
    return {m_data->data_extensions, m_data->data_extensions_count};
  }
  std::span<char const* const> extensions_used_c() const
  {
    return {m_data->extensions_used, m_data->extensions_used_count};
  }
  auto                         extensions_used() const { return cgltf_wrap<std::string_view>(extensions_used_c()); }
  std::span<char const* const> extensions_required_c() const
  {
    return {m_data->extensions_required, m_data->extensions_required_count};
  }
  auto extensions_required() const { return cgltf_wrap<std::string_view>(extensions_required_c()); }

private:
  MappedFile              m_mappedFile;
  std::vector<MappedFile> m_secondaryFiles;
  UniqueCgltfData         m_data;
};

// clang-format off
template <class T> struct cgltf_wrapper_traits;
template <> struct cgltf_wrapper_traits<cgltf_scene>             { using wrapper_type = CgltfScene; };
template <> struct cgltf_wrapper_traits<cgltf_primitive>         { using wrapper_type = CgltfPrimitive; };
template <> struct cgltf_wrapper_traits<cgltf_mesh>              { using wrapper_type = CgltfMesh; };
template <> struct cgltf_wrapper_traits<cgltf_node>              { using wrapper_type = CgltfNode; };
template <> struct cgltf_wrapper_traits<cgltf_node const* const> { using wrapper_type = CgltfNode; };
template <> struct cgltf_wrapper_traits<char const* const>       { using wrapper_type = std::string_view; };
template <class T> using cgltf_wrapper_t = typename cgltf_wrapper_traits<T>::wrapper_type;
// clang-format on

// Define a shortcut for the type returned by the range wrappers. Use a lambda
// since std::result_of_t fails with templates
template <class T>
using cgltf_wrap_result_t =
    decltype([](const std::span<T>& span) { return cgltf_wrap<cgltf_wrapper_t<T>, T>(span); }(std::declval<std::span<T>>()));
