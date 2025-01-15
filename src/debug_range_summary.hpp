/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#pragma once

// This file contains rangeSummary() and rangeSummaryVk(), which simply print
// the start and end of a C++ range or GPU buffer. Really useful for debugging,
// provided you can submit and finish command buffers between stages and then
// use this to inspect the data.

#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>
#include <nvvk/stagingmemorymanager_vk.hpp>
#include <ostream>
#include <sample_vulkan_objects.hpp>
#include <span>
#include <vulkan/vulkan_core.h>

namespace numerical_chars {
inline std::ostream& operator<<(std::ostream& os, char c)
{
  return std::is_signed<char>::value ? os << static_cast<int>(c) : os << static_cast<unsigned int>(c);
}

inline std::ostream& operator<<(std::ostream& os, signed char c)
{
  return os << static_cast<int>(c);
}

inline std::ostream& operator<<(std::ostream& os, unsigned char c)
{
  return os << static_cast<unsigned int>(c);
}
}  // namespace numerical_chars


// clang-format off
namespace glm {
inline std::ostream & operator<<(std::ostream &os, const glm::vec2& v)    { return os << "{" << v.x << ", " << v.y << "}"; }
inline std::ostream & operator<<(std::ostream &os, const glm::vec3& v)    { return os << "{" << v.x << ", " << v.y << ", " << v.z << "}"; }
inline std::ostream & operator<<(std::ostream &os, const glm::vec4& v)    { return os << "{" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << "}"; }
inline std::ostream & operator<<(std::ostream &os, const glm::uvec2& v)   { return os << "{" << v.x << ", " << v.y << "}"; }
inline std::ostream & operator<<(std::ostream &os, const glm::uvec3& v)   { return os << "{" << v.x << ", " << v.y << ", " << v.z << "}"; }
inline std::ostream & operator<<(std::ostream &os, const glm::uvec4& v)   { return os << "{" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << "}"; }
inline std::ostream & operator<<(std::ostream &os, const glm::u16vec2& v) { return os << "{" << v.x << ", " << v.y << "}"; }
inline std::ostream & operator<<(std::ostream &os, const glm::u16vec3& v) { return os << "{" << v.x << ", " << v.y << ", " << v.z << "}"; }
inline std::ostream & operator<<(std::ostream &os, const glm::u16vec4& v) { return os << "{" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << "}"; }
inline std::ostream & operator<<(std::ostream &os, const glm::u8vec2& v)  { using numerical_chars::operator<<; return os << "{" << v.x << ", " << v.y << "}"; }
inline std::ostream & operator<<(std::ostream &os, const glm::u8vec3& v)  { using numerical_chars::operator<<; return os << "{" << v.x << ", " << v.y << ", " << v.z << "}"; }
inline std::ostream & operator<<(std::ostream &os, const glm::u8vec4& v)  { using numerical_chars::operator<<; return os << "{" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << "}"; }
}
// clang-format on

template <glm::length_t M, glm::length_t N, typename T, glm::qualifier Q>
inline std::ostream& operator<<(std::ostream& os, const glm::mat<M, N, T, Q>& v)
{
  return os << glm::to_string(v);
}

// A class to wrap a std::ostream and insert a prefix at the start of every newline
class PrefixedLines : public std::streambuf
{
public:
  PrefixedLines(std::streambuf* output, std::string_view prefix)
      : m_output(output)
      , m_prefix(prefix)
  {
  }

  virtual int sync() override { return m_output->pubsync(); }

  // slow individual virtual calls per character per ostream in the chain, but
  // short code :)
  virtual int overflow(int c) override
  {
    assert(c == traits_type::to_char_type(c));
    if(c == traits_type::eof())
    {
      sync();
      return traits_type::eof();
    }
    if(std::exchange(m_newline, c == '\n'))
      m_output->sputn(m_prefix.data(), m_prefix.size());
    return m_output->sputc(traits_type::to_char_type(c));
  }

private:
  std::streambuf*  m_output;
  std::string_view m_prefix;
  bool             m_newline = false;
};

// Print the start and end of a std range
template <class T>
std::ostream& rangeSummary(std::ostream& os, const T& range, size_t maxItems = 6, bool multiline = false)
{
  using numerical_chars::operator<<;
  constexpr bool isString  = std::is_same_v<std::decay_t<decltype(*std::begin(range))>, char>;
  const char*    separator = multiline ? ",\n  " : (isString ? "" : ", ");

  // Indent any newlines written during the range
  PrefixedLines indent(os.rdbuf(), "  ");
  std::ostream  ios(multiline ? &indent : os.rdbuf());

  // Specialize based on whether the size of the range can be computed
  if constexpr(std::is_base_of_v<std::random_access_iterator_tag, typename std::iterator_traits<decltype(std::begin(range))>::iterator_category>)
  {
    os << "{";
    size_t size = std::distance(std::begin(range), std::end(range));
    if(multiline && size > 1)
    {
      ios << "\n";
    }
    if(size <= maxItems)
    {
      auto it = std::begin(range);
      if(it != std::end(range))
      {
        ios << *it;
        ++it;
      }
      for(; it != std::end(range); ++it)
      {
        ios << separator << *it;
      }
    }
    else
    {
      auto it = std::begin(range);
      for(size_t i = 0; i < maxItems / 2; ++i)
      {
        ios << *it++ << separator;
      }
      it += std::distance(it, std::end(range)) - maxItems / 2;
      ios << "...";
      while(it != std::end(range))
      {
        ios << separator << *it++;
      }
    }
    os << "}[" << size << "]";
  }
  else
  {
    os << "{";
    // Unknown size
    auto it = std::begin(range);
    if(multiline && it != std::end(range))
    {
      ios << "\n  ";
    }
    if(it != std::end(range))
    {
      ios << *it;
      ++it;
    }
    for(size_t i = 1; i < maxItems && it != std::end(range); ++i, ++it)
    {
      ios << separator << *it;
    }
    if(it != std::end(range))
    {
      ios << "...";
    }
    os << "}";
  }
  return os;
}

// Debugging utility class to allow dumping objects in GPU memory recursively
// with rangeSummaryVk.
// Usage:
// {
//   BufferDownloader downloader(context.device, queueFamilyIndex, context.allocator->getStaging());
//   rangeSummaryVk<glm::uvec3>(std::cerr << "Triangles: ", triangleIndicesVkBuffer, triangleCount) << "\n";
// }
// Requires VK_NV_copy_memory_indirect to be enabled
class BufferDownloader
{
public:
  BufferDownloader(VkDevice device, uint32_t queueFamilyIndex, nvvk::StagingMemoryManager* stagingMemoryManager)
  {
    assert(s_device == VK_NULL_HANDLE);
    s_device               = device;
    s_queueFamilyIndex     = queueFamilyIndex;
    s_stagingMemoryManager = stagingMemoryManager;
  }
  ~BufferDownloader()
  {
    s_device               = VK_NULL_HANDLE;
    s_queueFamilyIndex     = 0;
    s_stagingMemoryManager = nullptr;
  }
  template <class T>
  static const void* download(T bufferOrAddress, size_t bytes)
  {
    assert(s_device != VK_NULL_HANDLE);  // Make sure BufferDownloader() exists in the current scope
    VkCommandPoolCreateInfo poolCreateInfo = {
        .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext            = nullptr,
        .flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
        .queueFamilyIndex = s_queueFamilyIndex,
    };
    VkCommandPool pool;
    NVVK_CHECK(vkCreateCommandPool(s_device, &poolCreateInfo, nullptr, &pool));
    VkQueue queue;
    vkGetDeviceQueue(s_device, s_queueFamilyIndex, 0, &queue);
    VkCommandBufferAllocateInfo cmdAllocInfo = {
        .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext              = nullptr,
        .commandPool        = pool,
        .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBuffer cmd;
    NVVK_CHECK(vkAllocateCommandBuffers(s_device, &cmdAllocInfo, &cmd));
    VkCommandBufferBeginInfo beginInfo = {
        .sType            = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext            = nullptr,
        .flags            = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    };
    NVVK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));
    VkMemoryBarrier memBarrier = {.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                                  .pNext         = nullptr,
                                  .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
                                  .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &memBarrier, 0,
                         nullptr, 0, nullptr);
    const void* hostCopy;
    if constexpr(std::is_same_v<T, VkBuffer>)
      hostCopy = s_stagingMemoryManager->cmdFromBuffer(cmd, bufferOrAddress, 0, bytes);
    else
      hostCopy = s_stagingMemoryManager->cmdFromAddressNV(cmd, static_cast<VkDeviceAddress>(bufferOrAddress), bytes);
    NVVK_CHECK(vkEndCommandBuffer(cmd));
    VkSubmitInfo submit = {
        .sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext                = nullptr,
        .waitSemaphoreCount   = 0,
        .pWaitSemaphores      = nullptr,
        .pWaitDstStageMask    = nullptr,
        .commandBufferCount   = 1,
        .pCommandBuffers      = &cmd,
        .signalSemaphoreCount = 0,
        .pSignalSemaphores    = nullptr,
    };
    NVVK_CHECK(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
    NVVK_CHECK(vkQueueWaitIdle(queue));
    vkDestroyCommandPool(s_device, pool, nullptr);
    return hostCopy;
  }

  template <class T>
  static std::span<const T> download(const vkobj::Buffer<T>& array)
  {
    return std::span(reinterpret_cast<const T*>(download(static_cast<VkBuffer>(array), array.size() * sizeof(T))), array.size());
  }

  template <class T>
  static std::span<const T> download(vkobj::DeviceAddress<T> address, size_t elementCount)
  {
    return std::span(reinterpret_cast<const T*>(download(static_cast<VkDeviceAddress>(address), elementCount * sizeof(T))), elementCount);
  }

private:
  static inline thread_local VkDevice                    s_device               = VK_NULL_HANDLE;
  static inline thread_local uint32_t                    s_queueFamilyIndex     = 0;
  static inline thread_local nvvk::StagingMemoryManager* s_stagingMemoryManager = nullptr;
};

template <class T>
std::ostream& rangeSummaryVk(std::ostream& os, VkBuffer buffer, size_t elementCount, size_t maxItems = 6, bool multiline = false)
{
  std::span hostArray(reinterpret_cast<const T*>(BufferDownloader::download(buffer, elementCount * sizeof(T))), elementCount);
  return rangeSummary(os, hostArray, maxItems, multiline);
}

template <class T>
std::ostream& rangeSummaryVk(std::ostream& os, const vkobj::Buffer<T>& array, size_t maxItems = 6, bool multiline = false)
{
  std::span<const T> hostArray = BufferDownloader::download(array);
  return rangeSummary(os << static_cast<VkDeviceAddress>(array.address()), hostArray, maxItems, multiline);
}

template <class T>
std::ostream& rangeSummaryVk(std::ostream& os, vkobj::DeviceAddress<T> address, size_t elementCount, size_t maxItems = 6, bool multiline = false)
{
  std::span<const T> hostArray = BufferDownloader::download(address, elementCount);
  return rangeSummary(os << static_cast<VkDeviceAddress>(address), hostArray, maxItems, multiline);
}

template <class T>
std::ostream& rangeSummaryVk(std::ostream& os, VkDeviceAddress address, size_t elementCount, size_t maxItems = 6, bool multiline = false)
{
  return rangeSummaryVk(os << address, vkobj::DeviceAddress<T>(address), elementCount, maxItems, multiline);
}
