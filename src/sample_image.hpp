/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2014-2022 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <filesystem>
#include <nvvk/images_vk.hpp>
#include <nvvk/resourceallocator_vk.hpp>
#include <optional>
#include <stb_image.h>
#include <vulkan/vulkan_core.h>

// Unique pointers to automatically free stb_image data
using UniqueStbiImage8  = std::unique_ptr<stbi_uc, decltype(&stbi_image_free)>;
using UniqueStbiImage16 = std::unique_ptr<stbi_us, decltype(&stbi_image_free)>;

// Non-owning image data descriptor and pointer
struct ImageBase
{
  VkFormat                   format;
  VkExtent2D                 extent;
  std::span<const std::byte> data;
};

// Image descriptor and data
struct Image : ImageBase
{
  UniqueStbiImage8  image8  = {nullptr, &stbi_image_free};
  UniqueStbiImage16 image16 = {nullptr, &stbi_image_free};
};

// RAII Vulkan image object
class ImageVk
{
public:
  // nvvk::ResourceAllocator::destroy(nvvk::Texture) also destroys the image, so
  // m_image must be able to be moved out of this wrapper.
  friend class TextureVk;

  ImageVk(nvvk::ResourceAllocator& allocator, const VkImageCreateInfo& createInfo, VkMemoryPropertyFlags memoryPropertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
      : m_allocator(&allocator)
      , m_image(m_allocator->createImage(createInfo, memoryPropertyFlags))
  {
  }
  ImageVk(VkCommandBuffer            cmd,
          nvvk::ResourceAllocator&   allocator,
          std::span<const std::byte> initialData,
          const VkImageCreateInfo&   createInfo,
          VkImageLayout              initialLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
      : m_allocator(&allocator)
      , m_image(m_allocator->createImage(cmd, initialData.size(), initialData.data(), createInfo, initialLayout))
  {
  }
  ~ImageVk()
  {
    if(m_allocator)
      m_allocator->destroy(m_image);
  }
  ImageVk(const ImageVk&)            = delete;
  ImageVk& operator=(const ImageVk&) = delete;
  ImageVk(ImageVk&& other) noexcept
      : m_allocator(other.m_allocator)
      , m_image(other.m_image)
  {
    other.m_allocator = nullptr;
  }
  ImageVk& operator=(ImageVk&& other) noexcept
  {
    if(m_allocator)
      m_allocator->destroy(m_image);
    m_allocator       = other.m_allocator;
    m_image           = std::move(other.m_image);
    other.m_allocator = nullptr;
    return *this;
  }
  operator VkImage() const { return m_image.image; }

private:
  nvvk::ResourceAllocator* m_allocator = nullptr;
  nvvk::Image              m_image;
};

// Loads an decompresses an image from disk into system memory using stb_image
inline std::optional<Image> createImage(const std::filesystem::path& path, bool srgb)
{
  // Read the header once to check how many channels it has. We can't trivially use RGB/VK_FORMAT_R8G8B8_UNORM and
  // need to set req_comp=4 in such cases.
  int w = 0, h = 0, comp = 0;
  if(!stbi_info(path.string().c_str(), &w, &h, &comp))
  {
    LOGE("Failed to read %s\n", path.string().c_str());
    return std::nullopt;
  }

  // Read the header again to check if it has 16 bit data, e.g. for a heightmap.
  bool   is_16Bit        = stbi_is_16_bit(path.string().c_str());
  int    req_comp        = comp == 1 ? 1 : 4;
  size_t bytes_per_pixel = (is_16Bit ? sizeof(stbi_us) : sizeof(stbi_uc)) * req_comp;

  Image result;
  result.extent = {uint32_t(w), uint32_t(h)};
  if(is_16Bit)
  {
    result.image16 = UniqueStbiImage16(stbi_load_16(path.string().c_str(), &w, &h, &comp, req_comp), &stbi_image_free);
    result.data = std::span{reinterpret_cast<const std::byte*>(result.image16.get()), size_t(bytes_per_pixel * w * h)};
    result.format = req_comp == 1 ? VK_FORMAT_R16_UNORM : VK_FORMAT_R16G16B16A16_UNORM;
  }
  else
  {
    result.image8 = UniqueStbiImage8(stbi_load(path.string().c_str(), &w, &h, &comp, req_comp), &stbi_image_free);
    result.data   = std::span{reinterpret_cast<const std::byte*>(result.image8.get()), size_t(bytes_per_pixel * w * h)};
    result.format = req_comp == 1 ? VK_FORMAT_R8_UNORM : (srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM);
  }
  return result;
}

// Creates a vulkan image initialized from the description and data in system memory
inline ImageVk createImageVk(nvvk::ResourceAllocator& allocator, const ImageBase& image, VkCommandBuffer cmd, bool generateMipmaps)
{
  VkImageCreateInfo imageCreateInfo =
      nvvk::makeImage2DCreateInfo(image.extent, image.format, VK_IMAGE_USAGE_SAMPLED_BIT, generateMipmaps);
  if(!generateMipmaps)
    imageCreateInfo.mipLevels = 1;
  ImageVk result(cmd, allocator, image.data, imageCreateInfo, VK_IMAGE_LAYOUT_GENERAL /*VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL*/);
  //nvvk::cmdBarrierImageLayout(cmd, result, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  if(generateMipmaps)
    nvvk::cmdGenerateMipmaps(cmd, result, image.format, image.extent, imageCreateInfo.mipLevels);
  return result;
}

// A VkImage, VkImageView and a VkSampler, allocated by nvvk::ResourceAllocator
// and freed on destruction. Takes ownership and wraps the given ImageVk.
class TextureVk
{
public:
  // Note: consumes the image due to nvvk::ResourceAllocator::destroy() design
  TextureVk(nvvk::ResourceAllocator& allocator, ImageVk&& image, const VkImageViewCreateInfo& imageViewCreateInfo, const VkSamplerCreateInfo& samplerCreateInfo)
      : m_allocator(&allocator)
      , m_texture(m_allocator->createTexture(image.m_image, imageViewCreateInfo, samplerCreateInfo))
  {
    // TextureVk takes ownership of the image as m_allocator->destroy(m_texture)
    // will destroy it.
    image.m_allocator = nullptr;
  }
  ~TextureVk()
  {
    if(m_allocator)
      m_allocator->destroy(m_texture);
  }
  TextureVk(const TextureVk&)            = delete;
  TextureVk& operator=(const TextureVk&) = delete;
  TextureVk(TextureVk&& other) noexcept
      : m_allocator(other.m_allocator)
      , m_texture(other.m_texture)
  {
    other.m_allocator = nullptr;
  }
  TextureVk& operator=(TextureVk&& other) noexcept
  {
    if(m_allocator)
      m_allocator->destroy(m_texture);
    other.m_allocator = nullptr;
    m_allocator       = other.m_allocator;
    m_texture         = std::move(other.m_texture);
    return *this;
  }
  const VkDescriptorImageInfo& descriptor() const { return m_texture.descriptor; }

private:
  nvvk::ResourceAllocator* m_allocator = nullptr;
  nvvk::Texture            m_texture;
};

// Utility call to combine createImageVk() and TextureVk(), using some default
// sampler parameters
inline TextureVk createTextureVk(nvvk::ResourceAllocator& allocator, const ImageBase& image, VkCommandBuffer cmd, bool generateMipmaps)
{
  ImageVk imageVk = createImageVk(allocator, image, cmd, generateMipmaps);
  // DANGER: VkImageCreateInfo needs to be the same as in createImageVk
  VkImageCreateInfo imageCreateInfo =
      nvvk::makeImage2DCreateInfo(image.extent, image.format, VK_IMAGE_USAGE_SAMPLED_BIT, generateMipmaps);
  VkImageViewCreateInfo imageViewCreateInfo = nvvk::makeImageViewCreateInfo(imageVk, imageCreateInfo);
  VkSamplerCreateInfo   samplerCreateInfo{
        .sType                   = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .pNext                   = nullptr,
        .flags                   = 0,
        .magFilter               = VK_FILTER_LINEAR,
        .minFilter               = VK_FILTER_LINEAR,
        .mipmapMode              = VK_SAMPLER_MIPMAP_MODE_LINEAR,
        .addressModeU            = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeV            = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeW            = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .mipLodBias              = 0.0f,
        .anisotropyEnable        = VK_FALSE,
        .maxAnisotropy           = 0.0f,
        .compareEnable           = VK_FALSE,
        .compareOp               = VK_COMPARE_OP_NEVER,
        .minLod                  = 0.0f,
        .maxLod                  = FLT_MAX,
        .borderColor             = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
        .unnormalizedCoordinates = VK_FALSE,
  };
  return TextureVk(allocator, std::move(imageVk), imageViewCreateInfo, samplerCreateInfo);
}
