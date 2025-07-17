/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <chrono>
#include <cmath>
#include <filesystem>
#include <glm/gtx/hash.hpp>
#include <gltf_view.hpp>
#include <iostream>
#include <nvh/alignment.hpp>
#include <nvvk/error_vk.hpp>
#include <ranges>
#include <sample_allocation.hpp>
#include <sample_raytracing_objects.hpp>
#include <scene.hpp>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vulkan/vulkan_core.h>

// Scoped profiler for quick and coarse results
// https://stackoverflow.com/questions/31391914/timing-in-an-elegant-way-in-c
class Stopwatch
{
public:
  Stopwatch(std::string name)
      : m_name(std::move(name))
      , m_beg(std::chrono::high_resolution_clock::now())
  {
  }
  ~Stopwatch()
  {
    try
    {
      auto end = std::chrono::high_resolution_clock::now();
      auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - m_beg);
      std::cout << m_name << " : " << dur.count() << " ms\n";
    }
    catch(const std::exception& e)
    {
      // Print and ignore, to satisfy static analysis
      std::cerr << "Error in Stopwatch: " << e.what() << "\n";
    }
  }

private:
  std::string                                                 m_name;
  std::chrono::time_point<std::chrono::high_resolution_clock> m_beg;
};

// Returns the point on the AABB corner given an index in [0, 7]
// where the bits represent the x, y, and z coordinates of the corner
constexpr inline glm::vec3 corner(const AABB& aabb, int cornerBits)
{
  glm::vec3 result = aabb.min;
  if(cornerBits & 1)
    result.x = aabb.max.x;
  if(cornerBits & 2)
    result.y = aabb.max.y;
  if(cornerBits & 4)
    result.z = aabb.max.z;
  return result;
}

// Returns the AABB of the AABB after being transformed by the given matrix
// NOTE: there is a more efficient way of doing this for affine transforms
// https://stackoverflow.com/a/58630206
inline AABB transform(const glm::mat4& m, AABB aabb)
{
  AABB result;
  for(int i = 0; i < 8; ++i)
  {
    glm::vec3 c = m * glm::vec4(corner(aabb, i), 1.0f);
    if(i == 0)
      result = AABB(c, c);
    else
      result += AABB(c, c);
  }
  return result;
}

nvcluster_Context makeClusterContext()
{
  nvcluster_Context           nvclusterContext;
  nvcluster_ContextCreateInfo nvclusterContextInfo{};
  if(nvclusterCreateContext(&nvclusterContextInfo, &nvclusterContext) != nvcluster_Result::NVCLUSTER_SUCCESS)
    throw std::runtime_error("nvclusterCreateContext() failed");  // TODO: translate error code
  return nvclusterContext;
}

void destroyClusterContext(nvcluster_Context nvclusterContext)
{
  if(nvclusterDestroyContext(nvclusterContext) != nvcluster_Result::NVCLUSTER_SUCCESS)
    throw std::runtime_error("nvclusterDestroyContext() failed");  // TODO: translate error code
}

nvclusterlod_Context makeLodContext(nvcluster_Context nvclusterContext)
{
  nvclusterlod_Context           nvlodContext;
  nvclusterlod_ContextCreateInfo nvlodContextInfo{};
  nvlodContextInfo.clusterContext = nvclusterContext;
  if(nvclusterlodCreateContext(&nvlodContextInfo, &nvlodContext) != nvclusterlod_Result::NVCLUSTERLOD_SUCCESS)
    throw std::runtime_error("nvclusterlodCreateContext() failed");  // TODO: translate error code
  return nvlodContext;
}

void destroyLodContext(nvclusterlod_Context nvlodContext)
{
  if(nvclusterlodDestroyContext(nvlodContext) != nvclusterlod_Result::NVCLUSTERLOD_SUCCESS)
    throw std::runtime_error("nvclusterlodDestroyContext() failed");  // TODO: translate error code
}

// Shortcut to make a lod_mesh::LocalizedLodMesh with some default values
nvclusterlod::LocalizedLodMesh makeLodMesh(nvclusterlod_Context        context,
                                           std::span<const glm::uvec3> inputTriangleVertices,
                                           std::span<const glm::vec3>  inputVertexPositions,
                                           const SceneLodConfig&       lodConfig)
{
  nvclusterlod_MeshInput input{
      .triangleVertices = reinterpret_cast<const nvclusterlod_Vec3u*>(inputTriangleVertices.data()),
      .triangleCount    = uint32_t(inputTriangleVertices.size()),
      .vertexPositions  = reinterpret_cast<const nvcluster_Vec3f*>(inputVertexPositions.data()),
      .vertexCount      = uint32_t(inputVertexPositions.size()),
      .clusterConfig    = {},
      .groupConfig      = {},
      .decimationFactor = lodConfig.lodLevelDecimationFactor,
  };
  input.clusterConfig.minClusterSize     = (lodConfig.clusterSize * 3) / 4;
  input.clusterConfig.maxClusterSize     = lodConfig.clusterSize;
  input.clusterConfig.maxClusterVertices = 256u;  // VK_NV_cluster_acceleration_structure limit
  input.groupConfig.minClusterSize       = (lodConfig.clusterGroupSize * 3) / 4;
  input.groupConfig.maxClusterSize       = lodConfig.clusterGroupSize;

  nvclusterlod::LocalizedLodMesh localizedLodMesh;
  nvclusterlod_Result            result = nvclusterlod::generateLocalizedLodMesh(context, input, localizedLodMesh);
  if(result != nvclusterlod_Result::NVCLUSTERLOD_SUCCESS)
    throw std::runtime_error("nvclusterlod::generateLocalizedLodMesh() failed");
  return localizedLodMesh;
}

// Shortcut to make a lod_hierarchy::LodHierarchy and write it to a memory
// mapped file
LodHierarchyView makeLodHierarchy(nvclusterlod_Context context, const nvclusterlod::LodMesh& mesh, file_writer& alloc)
{
  nvclusterlod_HierarchyInput input{
      .clusterGeneratingGroups = mesh.clusterGeneratingGroups.data(),
      .clusterBoundingSpheres  = mesh.clusterBoundingSpheres.data(),
      .groupQuadricErrors      = mesh.groupQuadricErrors.data(),
      .groupClusterRanges      = mesh.groupClusterRanges.data(),
      .lodLevelGroupRanges     = mesh.lodLevelGroupRanges.data(),
      .clusterCount            = uint32_t(mesh.clusterBoundingSpheres.size()),
      .groupCount              = uint32_t(mesh.groupClusterRanges.size()),
      .lodLevelCount           = uint32_t(mesh.lodLevelGroupRanges.size()),
  };

  nvclusterlod::LodHierarchy hierarchy;
  nvclusterlod::generateLodHierarchy(context, input, hierarchy);

  // Copy the LodHierarchy contents to the memory mapping in alloc and return a
  // view (pointers to the data)
  // TODO: the lod_hierarchy::LodHierarchy allocation could be replace by
  // constructing direclty to the view
  return LodHierarchyView{
      .nodes                          = alloc.createArray(hierarchy.nodes),
      .groupCumulativeBoundingSpheres = alloc.createArray(hierarchy.groupCumulativeBoundingSpheres),
      .groupCumulativeQuadricError    = alloc.createArray(hierarchy.groupCumulativeQuadricError),
  };
}

// Enable hashing tuples of hashable values for std::unordered_map. This is used
// to hash multiple vertex attributes
namespace std {
namespace {
template <class T>
size_t hash_value(const T& v)
{
  return std::hash<T>{}(v);
}

template <class Tuple, size_t Index = std::tuple_size<Tuple>::value - 1>
struct hash_tuple
{
  static void combine(size_t& seed, Tuple const& tuple)
  {
    glm::detail::hash_combine(seed, hash_value(std::get<Index>(tuple)));
    hash_tuple<Tuple, Index - 1>::combine(seed, tuple);
  }
};

template <class Tuple>
struct hash_tuple<Tuple, 0>
{
  static void combine(size_t& seed, Tuple const& tuple)
  {
    glm::detail::hash_combine(seed, hash_value(std::get<0>(tuple)));
  }
};
}  // namespace

template <class... T>
struct hash<std::tuple<T...>>
{
  size_t operator()(const std::tuple<T...>& tuple) const noexcept
  {
    size_t seed = 0;
    hash_tuple<std::tuple<T...>>::combine(seed, tuple);
    return seed;
  };
};
}  // namespace std


using NodesList = cgltf_wrap_result_t<cgltf_node const* const>;

void multiplyNodeTransforms(const NodesList&                                nodes,
                            const glm::mat4&                                parentTransform,
                            std::vector<std::pair<cgltf_mesh*, glm::mat4>>& instances)
{
  for(const auto& node : nodes)
  {
    glm::mat4 transform = parentTransform * node.transform();
    if(node->mesh)
    {
      instances.push_back({node->mesh, transform});
    }
    multiplyNodeTransforms(node.children(), transform, instances);
  }
}

// Recursively propagate transforms of the node hierarchy and instantiate
// meshes
std::vector<std::pair<cgltf_mesh*, glm::mat4>> flatSceneInstances(NodesList sceneNodes)
{
  std::vector<std::pair<cgltf_mesh*, glm::mat4>> instances;
  multiplyNodeTransforms(sceneNodes, glm::identity<glm::mat4>(), instances);
  return instances;
}

// Common mesh representation before splitting into clusters.
struct Mesh
{
  Mesh(const CgltfPrimitive& primitive, std::unordered_map<std::u8string, size_t>& imagesIndex);
  std::vector<glm::uvec3> meshTriIndices;
  std::vector<glm::vec3>  meshPositions;
  std::vector<glm::vec3>  meshNormals;
  std::vector<glm::vec2>  meshTexCoords;
  shaders::Material       material;
};

Mesh::Mesh(const CgltfPrimitive& primitive, std::unordered_map<std::u8string, size_t>& imagesIndex)
{
#if 0
      try
      {
        meshops::ArrayView<const glm::uvec3> indices(primitive.indices<uint32_t>());
        std::ranges::copy(indices, std::back_inserter(meshTriIndices));
      }
      catch([[maybe_unused]] const std::runtime_error& e)
      {
        meshops::ArrayView<const glm::u16vec3> indices(primitive.indices<uint16_t>());
        std::ranges::copy(indices, std::back_inserter(meshTriIndices));
      }
      meshops::ArrayView<const glm::vec3> positions(primitive.attribute<glm::vec3>(cgltf_attribute_type_position).value());
      std::ranges::copy(positions, std::back_inserter(meshPositions));
#else
  // Rebuild topology with unique positions only, merging vertices that may
  // have been split due to hard edges.
  meshops::ArrayView<const glm::vec3> positions(primitive.attribute<glm::vec3>(cgltf_attribute_type_position).value());
  meshops::ArrayView<const glm::vec3> normals{};
  if(primitive.attribute<glm::vec3>(cgltf_attribute_type_normal))
  {
    normals = meshops::ArrayView<const glm::vec3>(primitive.attribute<glm::vec3>(cgltf_attribute_type_normal).value());
  }
  meshops::ArrayView<const glm::vec2> texCoords(
      primitive.attribute<glm::vec2>(cgltf_attribute_type_texcoord).value_or(CgltfAccessor<glm::vec2>{}));
  std::unordered_map<std::tuple<glm::vec3, glm::vec2>, uint32_t> uniquePositions;
  auto                                                           uniqueIndex = [&](uint32_t vertexIndex) {
    auto [it, created] =
        uniquePositions.try_emplace({positions[vertexIndex], texCoords.empty() ? glm::vec2{0.0f} : texCoords[vertexIndex]},
                                                                                              uint32_t(uniquePositions.size()));
    if(created)
    {
      meshPositions.push_back(positions[vertexIndex]);
      if(normals.size())
        meshNormals.push_back(normals[vertexIndex]);
      if(texCoords.size())
        meshTexCoords.push_back(texCoords[vertexIndex]);
    }
    return it->second;
  };
  try
  {
    meshops::ArrayView<const glm::uvec3> indices(primitive.indices<uint32_t>());
    for(auto& t : indices)
      meshTriIndices.emplace_back(uniqueIndex(t.x), uniqueIndex(t.y), uniqueIndex(t.z));
  }
  catch([[maybe_unused]] const std::runtime_error& e)
  {
    meshops::ArrayView<const glm::u16vec3> indices(primitive.indices<uint16_t>());
    for(auto& t : indices)
      meshTriIndices.emplace_back(uniqueIndex(t.x), uniqueIndex(t.y), uniqueIndex(t.z));
  }
#endif

  // Generate angle-weighted normals if missing
  if(meshNormals.empty())
  {
    meshNormals.resize(meshPositions.size(), glm::vec3(0.0f));
    for(const glm::uvec3& t : meshTriIndices)
    {
      glm::vec3 p0 = meshPositions[t.x];
      glm::vec3 p1 = meshPositions[t.y];
      glm::vec3 p2 = meshPositions[t.z];
      glm::vec3 n012 = glm::cross(p2 - p1, p0 - p1);
      glm::vec3 n120 = glm::cross(p0 - p2, p1 - p2);
      glm::vec3 n201 = glm::cross(p1 - p0, p2 - p0);
      meshNormals[t.x] += n012 * (asinf(glm::length(n012)) / (glm::length(n012) + 1e-10f));
      meshNormals[t.y] += n120 * (asinf(glm::length(n120)) / (glm::length(n120) + 1e-10f));
      meshNormals[t.z] += n201 * (asinf(glm::length(n201)) / (glm::length(n201) + 1e-10f));
    }
    for(glm::vec3& n : meshNormals)
      n = glm::normalize(n);
  }

  if(primitive.material && primitive.material->has_pbr_metallic_roughness)
  {
    const cgltf_pbr_metallic_roughness& pbr = primitive.material->pbr_metallic_roughness;
    std::ranges::copy(pbr.base_color_factor, glm::value_ptr(material.albedo));
    if(pbr.base_color_texture.texture)
    {
      const cgltf_image* baseImage = pbr.base_color_texture.texture->image;
      assert(baseImage);
      // This uses std::u8string because glTF URIs are UTF-8 encoded, and we
      // want to communicate that to the fs::path constructor.
      std::u8string uri     = std::u8string(reinterpret_cast<const char8_t*>(baseImage->uri));
      auto [index, created] = imagesIndex.try_emplace(std::move(uri), imagesIndex.size());
      assert(index->second < 256);
      material.albedoTexture = uint8_t(index->second);
    }
    else
    {
      material.albedoTexture = -1;
    }
    material.roughness = pbr.roughness_factor;
    material.metallic  = pbr.metallic_factor;
    if(material.albedo == glm::vec4(1.0f))  // filter out unnatural 100% albedo
    {
      material.albedo = glm::vec4(0.5f, 0.5f, 0.5f, 1.0f);
    }
  }
  else
  {
    // Default material
    material.albedo        = {0.41f, 0.27f, 0.14f, 1.0f};
    material.albedoTexture = -1;
    material.roughness     = 0.8f;
    material.metallic      = 0.3f;
  }
}

// Utility to extract a subset of a vector by indices
template <class T>
static std::vector<T> selection(const std::vector<T> source, std::span<const uint32_t> items)
{
  std::vector<T> result;
  result.reserve(items.size());
  for(const uint32_t& index : items)
    result.push_back(source[index]);
  return result;
}

Cluster::Cluster(const Mesh& baseMesh, std::span<const glm::uvec3> triangleIndices, std::span<const uint32_t> vertexIndices, file_writer& alloc)
    : meshTriIndices(alloc.createArray(std::views::transform(triangleIndices, [](glm::uvec3 v) { return glm::u8vec3(v); })))  // uvec3 -> u8vec3 conversion
    , meshPositions(alloc.createArray(selection(baseMesh.meshPositions, vertexIndices)))
    , meshNormals(alloc.createArray(selection(baseMesh.meshNormals, vertexIndices)))
    , meshTexCoords(alloc.createArray(baseMesh.meshTexCoords.empty() ? std::vector<glm::vec2>() :
                                                                       selection(baseMesh.meshTexCoords, vertexIndices)))
    , material(baseMesh.material)
{
  // Make sure uint8_t triangle indices is enough, i.e. indices are < 256
  assert(std::ranges::equal(meshTriIndices, triangleIndices,
                            [](glm::u8vec3 a, glm::uvec3 b) { return a.x == b.x && a.y == b.y && a.z == b.z; }));
}

ClusteredMesh::ClusteredMesh(nvclusterlod_Context context, const Mesh& mesh, const SceneLodConfig& lodConfig, file_writer& alloc)
    : ClusteredMesh(context, mesh, makeLodMesh(context, mesh.meshTriIndices, mesh.meshPositions, lodConfig), alloc)
{
}
ClusteredMesh::ClusteredMesh(nvclusterlod_Context context, const Mesh& baseMesh, nvclusterlod::LocalizedLodMesh&& lodMesh, file_writer& alloc)
    : hierarchy(makeLodHierarchy(context, lodMesh.lodMesh, alloc))
    , clusterTriangleRanges(alloc.createArray(lodMesh.lodMesh.clusterTriangleRanges))
    , clusterVertexRanges(alloc.createArray(lodMesh.clusterVertexRanges))
    , clusteredMesh(baseMesh,
                    std::span(reinterpret_cast<const glm::uvec3*>(lodMesh.lodMesh.triangleVertices.data()),
                              lodMesh.lodMesh.triangleVertices.size()),
                    lodMesh.vertexGlobalIndices,
                    alloc)
    , lodLevelGroups(alloc.createArray(lodMesh.lodMesh.lodLevelGroupRanges))
    , clusterGeneratingGroups(alloc.createArray(lodMesh.lodMesh.clusterGeneratingGroups))
    , groupClusterRanges(alloc.createArray(lodMesh.lodMesh.groupClusterRanges))
{
  // Builid bi-directional group dependency tables
  std::vector<offset_span<uint32_t>>  tmpGroupGeneratingGroups;
  std::vector<std::vector<uint32_t>>  tmpGroupGeneratedGroups;
  nvclusterlod::GroupGeneratingGroups generatingGroupsInit;

  nvclusterlod_Result success =
      nvclusterlod::generateGroupGeneratingGroups(groupClusterRanges, clusterGeneratingGroups, generatingGroupsInit);
  if(success != nvclusterlod_Result::NVCLUSTERLOD_SUCCESS)
    throw std::runtime_error("nvclusterlod::generateGroupGeneratingGroups() failed");

  for(const nvcluster_Range& range : generatingGroupsInit.ranges)
  {
    std::span<const uint32_t> generatingGroups(subspan(generatingGroupsInit.groups, range));
    tmpGroupGeneratingGroups.emplace_back(alloc.createArray(generatingGroups));
  }
  groupGeneratingGroups = alloc.createArray(tmpGroupGeneratingGroups);
  tmpGroupGeneratedGroups.resize(tmpGroupGeneratingGroups.size());
  for(uint32_t groupIndex = 0; groupIndex < uint32_t(tmpGroupGeneratingGroups.size()); ++groupIndex)
  {
    for(uint32_t generatingGroup : tmpGroupGeneratingGroups[groupIndex])
      tmpGroupGeneratedGroups[generatingGroup].push_back(groupIndex);
  }
  groupGeneratedGroups = alloc.createArray<offset_span<uint32_t>>(tmpGroupGeneratedGroups.size());
  for(uint32_t groupIndex = 0; groupIndex < uint32_t(tmpGroupGeneratingGroups.size()); ++groupIndex)
    groupGeneratedGroups[groupIndex] = alloc.createArray(tmpGroupGeneratedGroups[groupIndex]);
}

// Utility linear allocator that doesn't own any memory and returns offsets
// instead of absolute addresses. It expects the base address is maximally
// aligned. std::pmr::monotonic_buffer_resource could also be used.
class OffsetAllocator
{
public:
  OffsetAllocator(VkDeviceSize minAlign)
      : m_align(minAlign)
  {
  }
  template <class T>
  vkobj::DeviceAddress<T> allocateOffset(VkDeviceSize elements)
  {
    VkDeviceSize bytes  = elements * sizeof(T);
    m_align             = std::max(m_align, alignof(T));
    VkDeviceSize offset = (m_next + m_align - 1llu) & ~(m_align - 1llu);
    m_next              = offset + bytes;
    return vkobj::DeviceAddress<T>(offset);
  }
  VkDeviceSize allocatedSize() { return m_next; }
  VkDeviceSize maxAlign() { return m_align; }

private:
  VkDeviceSize m_align = 0;
  VkDeviceSize m_next  = 0;
};

ClusterGroupGeometryVk::ClusterGroupGeometryVk(ResourceAllocator*   allocator,
                                               VkBuffer             memoryBuffer,
                                               PoolAllocator&       memoryPool,
                                               const ClusteredMesh& mesh,
                                               uint32_t             groupIndex,
                                               VkCommandBuffer      transferCmd)
{
  nvcluster_Range clusterRange = mesh.groupClusterRanges[groupIndex];

  // Create a staging buffer to pack geometry data. Align to 16 bytes to support
  // the default buffer_reference_align from
  // https://github.com/KhronosGroup/GLSL/blob/main/extensions/ext/GLSL_EXT_buffer_reference.txt
  OffsetAllocator subAlloc(16);

  // Pack cluster group geometry, just computing offsets for now
  auto clusterOffsetsHost     = std::vector<shaders::ClusterGeometry>();
  auto clustersOffset         = subAlloc.allocateOffset<shaders::ClusterGeometry>(clusterRange.count);
  auto generatingGroupsOffset = subAlloc.allocateOffset<uint32_t>(clusterRange.count);
  auto clasAddressesOffset    = subAlloc.allocateOffset<VkDeviceAddress>(clusterRange.count);
  clusterOffsetsHost.reserve(clusterRange.count);
  for(uint32_t clusterIndex : indices(clusterRange))
  {
    nvcluster_Range triangleRange = mesh.clusterTriangleRanges[clusterIndex];
    nvcluster_Range vertexRange   = mesh.clusterVertexRanges[clusterIndex];
    clusterOffsetsHost.push_back(shaders::ClusterGeometry{
        .triangleCount           = triangleRange.count,
        .vertexCount             = vertexRange.count,
        .triangleVerticesAddress = subAlloc.allocateOffset<glm::u8vec3>(triangleRange.count),
        .vertexPositionsAddress  = subAlloc.allocateOffset<glm::vec3>(vertexRange.count),

        .vertexNormalsAddress   = subAlloc.allocateOffset<glm::vec3>(vertexRange.count),
        .vertexTexcoordsAddress = mesh.clusteredMesh.meshTexCoords.empty() ?
                                      vkobj::DeviceAddress<glm::vec2>(0) :
                                      subAlloc.allocateOffset<glm::vec2>(vertexRange.count),
    });
  }

  // Allocate space in the pool
  m_alloc = PoolMemory(memoryPool, subAlloc.allocatedSize(), subAlloc.maxAlign());

  // Create a staging buffer, map it and schedule a copy to the memory pool
  void* stagingPtr = allocator->getStaging()->cmdToBufferT<std::byte>(transferCmd, memoryBuffer, memoryPool.offsetOf(m_alloc),
                                                                      subAlloc.allocatedSize());

  // Copy the file-mapped source data, mesh.clusteredMesh, into the staging
  // buffer. We also need to compute device pointers for
  // shaders::ClusterGeometry with translateOffset() and copy those into the
  // staging memory.
  auto copyAndTranslate = [this, &stagingPtr](auto offset, auto array) {
    // Copy an array into the staging memory at offset 'offset'
    std::ranges::copy(array, vkobj::translateOffset(offset, stagingPtr));

    // Return a device pointer to the beginning of the copied range
    return vkobj::translateOffset(offset, m_alloc);
  };
  auto clusterAddressesStaging      = std::span(vkobj::translateOffset(clustersOffset, stagingPtr), clusterRange.count);
  m_clusterGeometryAddressesAddress = vkobj::translateOffset(clustersOffset, m_alloc);
  m_clusterGeneratingGroupsAddress =
      copyAndTranslate(generatingGroupsOffset, subspan(mesh.clusterGeneratingGroups, clusterRange));
  m_clasAddressesAddress = vkobj::translateOffset(clasAddressesOffset, m_alloc);
  for(size_t i = 0; i < clusterOffsetsHost.size(); ++i)
  {
    nvcluster_Range          triangleRange = mesh.clusterTriangleRanges[clusterRange.offset + i];
    nvcluster_Range          vertexRange   = mesh.clusterVertexRanges[clusterRange.offset + i];
    shaders::ClusterGeometry cluster       = clusterOffsetsHost[i];
    assert(cluster.triangleCount <= 256);
    assert(cluster.vertexCount <= 256);
    clusterAddressesStaging[i] = shaders::ClusterGeometry{
        .triangleCount = cluster.triangleCount,
        .vertexCount   = cluster.vertexCount,
        .triangleVerticesAddress =
            copyAndTranslate(cluster.triangleVerticesAddress, subspan(mesh.clusteredMesh.meshTriIndices, triangleRange)),
        .vertexPositionsAddress =
            copyAndTranslate(cluster.vertexPositionsAddress, subspan(mesh.clusteredMesh.meshPositions, vertexRange)),
        .vertexNormalsAddress = copyAndTranslate(cluster.vertexNormalsAddress, subspan(mesh.clusteredMesh.meshNormals, vertexRange)),
        .vertexTexcoordsAddress =
            mesh.clusteredMesh.meshTexCoords.empty() ?
                vkobj::DeviceAddress<glm::vec2>(0) :
                copyAndTranslate(cluster.vertexTexcoordsAddress, subspan(mesh.clusteredMesh.meshTexCoords, vertexRange)),
    };
  }
}

SceneImage::SceneImage(const Image& image, file_writer& alloc)
    : format(image.format)
    , extent(image.extent)
    , data(alloc.createArray(image.data))
{
}

SceneFile::SceneFile(const fs::path& gltfPath, const fs::path& cachePath, const SceneLodConfig& lodConfig, bool invalidateCache)
    : path(gltfPath)
{
  if(!invalidateCache && std::filesystem::is_regular_file(cachePath))
  {
    memoryMap.emplace(cachePath);
    uint32_t version = reinterpret_cast<const Scene*>(memoryMap->data())->version;
    if(version == SCENE_RENDERCACHE_VERSION)
    {
      LOGI("Using cache: %s\n", cachePath.string().c_str());
    }
    else
    {
      LOGI("Recreating rendercache %s (version %u, expected %u)\n", cachePath.string().c_str(), version, SCENE_RENDERCACHE_VERSION);
      memoryMap.reset();
    }
  }

  // Create the render cache if it doesn't exist
  if(!memoryMap)
  {
    Stopwatch stopwatch("Processing " + path.string());
    fs::path tmpPath = cachePath;
    tmpPath.replace_extension(".creating");  // temporary output in case we fail/crash
    file_writer  writer(tmpPath, 100llu << 30 /* 100GB max. virtual address space */);
    Scene*       newData = writer.create<Scene>();
    SceneCounts& counts  = newData->counts;

    // TODO: allocate and fill arrays directly in the memory mapping to avoid
    // the extra allocation and copying memory around
    std::vector<uint32_t>                     meshGroupOffsets = {0};
    std::vector<Instance>                     instances;
    std::vector<ClusteredMesh>                meshes;
    CgltfModel                                gltfModel(path);
    std::unordered_map<cgltf_mesh*, uint32_t> meshIndices;
    std::unordered_map<std::u8string, size_t> imagesIndex;
    std::vector<std::vector<uint32_t>>        meshPrimitives;

    // These don't currently do anything, but are here in case the library does
    // use them at some point
    std::unique_ptr<nvcluster_Context_t, void (*)(nvcluster_Context_t*)> nvclusterContext(makeClusterContext(), destroyClusterContext);
    std::unique_ptr<nvclusterlod_Context_t, void (*)(nvclusterlod_Context_t*)> nvlodContext(
        makeLodContext(nvclusterContext.get()), destroyLodContext);
    for(auto [cgltfMesh, transform] : flatSceneInstances(gltfModel.scenes()[0].nodes()))
    {
      auto [it, created] = meshIndices.try_emplace(cgltfMesh, uint32_t(meshIndices.size()));

      // If this is the first reference to this mesh
      if(created)
      {
        // Create the LOD hierarchy
        meshPrimitives.emplace_back();
        for(const CgltfPrimitive& primitive : CgltfMesh{*cgltfMesh}.primitives())
        {
          LOGI("Creating LOD for mesh %zu\n", meshes.size());

          // Convert gltf to consistent mesh, e.g. uint32_t triangle indices
          meshPrimitives.back().push_back(uint32_t(meshes.size()));
          meshes.emplace_back(nvlodContext.get(), Mesh(primitive, imagesIndex), lodConfig, writer);
          meshGroupOffsets.push_back((meshGroupOffsets.empty() ? 0 : meshGroupOffsets.back())
                                     + uint32_t(meshes.back().hierarchy.groupCumulativeBoundingSpheres.size()));
          for(auto& groupClusterRange : meshes.back().groupClusterRanges)
            counts.maxClustersPerGroup = std::max(counts.maxClustersPerGroup, groupClusterRange.count);
          counts.maxClustersPerMesh = std::max(counts.maxClustersPerMesh, uint32_t(meshes.back().clusterTriangleRanges.size()));
          counts.maxLODLevel = std::max(counts.maxLODLevel, uint32_t(meshes.back().lodLevelGroups.size()));
          counts.totalGroups += uint32_t(meshes.back().groupClusterRanges.size());
          counts.totalClusters += uint32_t(meshes.back().clusterTriangleRanges.size());
          for(nvcluster_Range triangleRange : meshes.back().clusterTriangleRanges)
          {
            counts.maxClusterTriangleCount = std::max(counts.maxClusterTriangleCount, triangleRange.count);
            counts.totalTriangles += triangleRange.count;
          }
          for(nvcluster_Range vertexRange : meshes.back().clusterVertexRanges)
          {
            counts.maxClusterVertexCount = std::max(counts.maxClusterVertexCount, vertexRange.count);
            counts.totalVertices += vertexRange.count;
          }

          meshes.back().aabb = AABB::empty();
          for(auto& position : meshes.back().clusteredMesh.meshPositions)
            meshes.back().aabb += AABB(position, position);
        }
      }

      // Add an instance
      float uniformScale = std::max(glm::length(transform[0]), std::max(glm::length(transform[1]), glm::length(transform[2])));
      for(uint32_t meshPrimitiveMesh : meshPrimitives[it->second])
        instances.push_back({transform, meshPrimitiveMesh, uniformScale});
      counts.maxTotalInstanceClusters += uint32_t(meshes[it->second].clusterTriangleRanges.size());
      counts.maxTotalInstanceNodes += uint32_t(meshes[it->second].hierarchy.nodes.size());
    }

    if(meshes.empty() || instances.empty())
      throw std::runtime_error("scene is empty");

    // Per-instance processing
    newData->worldAABB = AABB::empty();
    std::vector<uint32_t> meshInstanceCounts(meshes.size(), 0);
    for(const Instance& instance : instances)
    {
      // Record per-mesh instances for conservative memory allocation
      meshInstanceCounts[instance.meshIndex]++;

      // Compute the world AABB
      newData->worldAABB += transform(instance.transform, meshes[instance.meshIndex].aabb);
    }

    // Write everything to the render cache file
    counts.totalMeshes          = uint32_t(meshes.size());
    counts.totalInstances       = uint32_t(instances.size());
    newData->meshes             = writer.createArray(meshes);
    newData->instances          = writer.createArray(instances);
    newData->meshGroupOffsets   = writer.createArray(meshGroupOffsets);
    newData->meshInstanceCounts = writer.createArray(meshInstanceCounts);
    newData->images             = writer.createArray<SceneImage>(imagesIndex.size());
    for(auto& [uri, index] : imagesIndex)
    {
      // TODO: handle srgb override from gltf
      std::optional<Image> image = createImage(path.parent_path() / fs::path(uri), false /* srgb */);
      if(image)
        newData->images[index] = SceneImage(*image, writer);
    }

    newData->version = SCENE_RENDERCACHE_VERSION;

    // Destroy writer in the current scope to save more {...} indentation
    file_writer(std::move(writer));

    // Finished writing. Rename to the final filename
    fs::remove(cachePath);  // Because windows errors out if the target exists
    fs::rename(tmpPath, cachePath);
  }

  // Reload the render cache if it was just created
  if(!memoryMap)
    memoryMap.emplace(cachePath);
  data = reinterpret_cast<const Scene*>(memoryMap->data());

  if(data->version != SCENE_RENDERCACHE_VERSION)
    throw std::runtime_error("unknown rendercache version " + std::to_string(data->version) + ", expected "
                             + std::to_string(SCENE_RENDERCACHE_VERSION));

  for(auto& mesh : data->meshes)
    if(mesh.hierarchy.nodes.size() > (1u << 26) /* NodeRange::childOffset */ || mesh.hierarchy.nodes.size() > TRAVERSAL_MAX_NODES)
      throw std::runtime_error("node count overflow in mesh " + std::to_string(&mesh - &data->meshes.front()));

  if(data->counts.totalGroups > (1u << 31))  // streaming::GroupRequest::globalGroup
    throw std::runtime_error("total group count overflow");

  if(data->counts.maxClustersPerGroup == 0)
    throw std::runtime_error("max clusters per mesh is zero");

  uint32_t totalLod0Triangles = 0;
  for(auto& mesh : data->meshes)
    for(uint32_t group : indices(mesh.lodLevelGroups[0]))
      for(uint32_t cluster : indices(mesh.groupClusterRanges[group]))
        totalLod0Triangles += mesh.clusterTriangleRanges[cluster].count;
  printf("Total LOD0 triangles: %u\n", totalLod0Triangles);
}

constexpr VkBufferUsageFlags s_defaultUsage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
constexpr VkBufferUsageFlags s_defaultUsageBuild = s_defaultUsage | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
constexpr VkMemoryPropertyFlags s_defaultProps = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

ClusteredMeshVk::ClusteredMeshVk(ResourceAllocator* allocator, const ClusteredMesh& clusteredMesh, VkCommandBuffer transferCmd)
    : nodes(allocator, clusteredMesh.hierarchy.nodes, s_defaultUsage, s_defaultProps, transferCmd)
    , groups(allocator, clusteredMesh.hierarchy.groupCumulativeQuadricError.size(), s_defaultUsage, s_defaultProps)
    , groupQuadricErrors(allocator, clusteredMesh.hierarchy.groupCumulativeQuadricError, s_defaultUsage, s_defaultProps, transferCmd)
    , groupBoundingSphers(allocator, clusteredMesh.hierarchy.groupCumulativeBoundingSpheres, s_defaultUsage, s_defaultProps, transferCmd)
{
  vkCmdFillBuffer(transferCmd, groups, 0, nvh::align_up(groups.size_bytes(), 4), 0u);

  // Build per-cluster LOD levels from the lod level cluster ranges. Wasteful
  // for memory, but convenient for visualizing the LOD level in a shader.
  std::vector<uint8_t> groupLodLevelsInit(groupQuadricErrors.size());
  for(uint8_t lodLevel = 0; lodLevel < uint8_t(clusteredMesh.lodLevelGroups.size()); ++lodLevel)
  {
    nvcluster_Range lodGroupsRange = clusteredMesh.lodLevelGroups[lodLevel];
    for(uint8_t& groupLodLevel : subspan(groupLodLevelsInit, lodGroupsRange))
      groupLodLevel = lodLevel;
  }
  groupLodLevels = vkobj::Buffer(allocator, groupLodLevelsInit,
                                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, transferCmd);
}

SceneVK::SceneVK(ResourceAllocator* allocator, const Scene& scene, VkCommandPool initPool, VkQueue initQueue)
    : counts(scene.counts)
    , instances(allocator,
                scene.instances,
                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                vkobj::ImmediateCommandBuffer(allocator->getDevice(), initPool, initQueue))
    , allGroupNeededFlags(allocator,
                          scene.counts.totalGroups,
                          VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
{
  // A command buffer to transfer scene data to the GPU. This is periodically
  // submitted when staging memory is full
  vkobj::ImmediateCommandBuffer transferCmd(allocator->getDevice(), initPool, initQueue);
  auto                          flushStagingWhenNeeded = [&]() {
    VkDeviceSize allocated, used;
    allocator->getStaging()->getUtilization(allocated, used);
    // When staging usage is above 100MB,
    if(used > 1llu << 20)
    {
      // Submit the command buffer, wait for execution to complete amd create a
      // new command buffer to continue loading.
      transferCmd = vkobj::ImmediateCommandBuffer(allocator->getDevice(), initPool, initQueue);

      // Mark all staging buffers for reuse
      allocator->finalizeStaging();
    }
  };

  clusteredMeshes.reserve(scene.meshes.size());
  for(const ClusteredMesh& clusteredMesh : scene.meshes)
  {
    // Separate uploads to avoid creating too many staging buffers
    clusteredMeshes.emplace_back(allocator, clusteredMesh, transferCmd);
    flushStagingWhenNeeded();
  }

  std::unordered_map<int8_t, int8_t> imageReindex{{-1, -1}};
  int8_t                             nextImage      = 0;
  int8_t                             nextValidImage = 0;
  for(const SceneImage& image : scene.images)
  {
    if(!image.data.empty())
    {
      // Separate uploads to avoid creating too many staging buffers
      ImageBase imageBase{.format = image.format, .extent = image.extent, .data = image.data};
      textures.push_back(createTextureVk(*allocator, imageBase, transferCmd, false));
      textureDescriptors.push_back(textures.back().descriptor());
      imageReindex[nextImage] = nextValidImage++;
      flushStagingWhenNeeded();
    }
    ++nextImage;
  }

  std::vector<shaders::Mesh> meshPointersBuilder;
  meshPointersBuilder.reserve(clusteredMeshes.size());
  for(size_t i = 0; i < clusteredMeshes.size(); ++i)
  {
    const ClusteredMesh&   clusteredMesh   = scene.meshes[i];
    const ClusteredMeshVk& clusteredMeshVk = clusteredMeshes[i];

    // HACK: vulkan cannot trivially bind null image descriptors so it's
    // easier to filter out textures that failed to load. The above filtering
    // breaks image indices so they need to be translated.
    // TODO: load and validate images before storing the original index
    shaders::Material material = clusteredMesh.clusteredMesh.material;
    assert(imageReindex.count(material.albedoTexture));
    material.albedoTexture = imageReindex[material.albedoTexture];

    vkobj::DeviceAddress<uint8_t> meshGroupNeededFlagsAddress(VkDeviceAddress(allGroupNeededFlags.address())
                                                              + sizeof(uint8_t) * scene.meshGroupOffsets[i]);
    meshPointersBuilder.push_back(shaders::Mesh{
        .nodesAddress                = deviceReinterpretCast<shaders::Node>(clusteredMeshVk.nodes.address()),
        .groupsAddress               = clusteredMeshVk.groups.address(),
        .groupQuadricErrorsAddress   = clusteredMeshVk.groupQuadricErrors.address(),
        .groupBoundingSpheresAddress = deviceReinterpretCast<glm::vec4>(clusteredMeshVk.groupBoundingSphers.address()),
        .groupNeededFlagsAddress     = meshGroupNeededFlagsAddress,
        .groupLodLevelsAddress       = clusteredMeshVk.groupLodLevels.address(),
        .material                    = material,
        .groupCount                  = uint32_t(clusteredMeshVk.groupQuadricErrors.size()),
        .residentClusterCount        = 0,
    });
  }
  {
    meshPointers = vkobj::Buffer(allocator, meshPointersBuilder,
                                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, transferCmd);

    // A single barrier at the end of uploading the scene to make sure it's
    // visible to anything reading it
    memoryBarrier(transferCmd, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                  VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
  }
}

// TODO: remove Scene& parameter
void SceneVK::cmdResetStreaming(ResourceAllocator* allocator, const Scene& scene, VkCommandBuffer cmd)
{
  totalResidentClusters         = 0;
  totalResidentInstanceClusters = 0;

  for(size_t meshIndex = 0; meshIndex < clusteredMeshes.size(); ++meshIndex)
  {
    const ClusteredMeshVk& mesh = clusteredMeshes[meshIndex];
    vkCmdFillBuffer(cmd, mesh.groups, 0, nvh::align_up(mesh.groups.size_bytes(), 4), 0u);
    vkCmdFillBuffer(cmd, meshPointers, sizeof(shaders::Mesh) * meshIndex + offsetof(shaders::Mesh, residentClusterCount),
                    sizeof(shaders::Mesh::residentClusterCount), 0u);
  }

  // Mark the last group in each mesh as needed and a root node
  std::span<uint8_t> groupNeededFlagsHost = cmdToArray(*allocator->getStaging(), cmd, allGroupNeededFlags);
  std::ranges::fill(groupNeededFlagsHost, uint8_t(0));
  for(size_t meshIndex = 0; meshIndex < scene.meshes.size(); ++meshIndex)
  {
    uint32_t meshGroupCount = uint32_t(scene.meshes[meshIndex].groupClusterRanges.size());
    groupNeededFlagsHost[scene.meshGroupOffsets[meshIndex] + meshGroupCount - 1] = STREAMING_GROUP_IS_NEEDED | STREAMING_GROUP_IS_ROOT;
  }
  memoryBarrier(cmd, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
}
