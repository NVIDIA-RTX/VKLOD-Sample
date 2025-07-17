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

#pragma once

#include <exception>
#include <future>
#include <limits>
#include <memory>
#include <memory_resource>
#include <mutex>
#include <nvvk/memallocator_vk.hpp>
#include <nvvk/resourceallocator_vk.hpp>
#include <ratio>
#include <sample_allocation.hpp>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vulkan/vulkan_core.h>

namespace vkobj {

// Move-only wrapper with a destroy function
template <class T, auto DestroyFunc>
class VulkanHandle
{
public:
  // Support empty construction for non-raii users. Note that the address of the
  // object can never be taken.
  VulkanHandle() = default;

  // Allow a raw handle to be passed in for automatic deletion
  VulkanHandle(VkDevice device, T&& handle)  // require std::move() so there's a clear change of ownership
      : m_device(device)
      , m_handle(handle)
  {
  }
  ~VulkanHandle() { destroy(); }
  VulkanHandle(const VulkanHandle& other) = delete;
  VulkanHandle(VulkanHandle&& other) noexcept
      : m_device(other.m_device)
      , m_handle(other.m_handle)
  {
    other.m_handle = VK_NULL_HANDLE;
  }
  VulkanHandle& operator=(const VulkanHandle& other) = delete;
  VulkanHandle& operator=(VulkanHandle&& other) noexcept
  {
    destroy();
    m_device       = other.m_device;
    m_handle       = other.m_handle;
    other.m_handle = VK_NULL_HANDLE;
    return *this;
  }
  operator T() const { return m_handle; }
  explicit operator bool() const { return m_handle != VK_NULL_HANDLE; }

private:
  void destroy()
  {
    if(m_handle != VK_NULL_HANDLE)
      DestroyFunc(m_device, m_handle, nullptr);
  }
  VkDevice m_device = VK_NULL_HANDLE;
  T        m_handle = VK_NULL_HANDLE;
};

// Adds a constructor supporting many vkCreate*() calls that take a
// Vk*CreateInfo struct
template <class T, class CreateInfo, auto CreateFunc, auto DestroyFunc>
class VulkanObject : public VulkanHandle<T, DestroyFunc>
{
public:
  using VulkanHandle<T, DestroyFunc>::VulkanHandle;
  VulkanObject(VkDevice device, const CreateInfo& createInfo)
      : VulkanHandle<T, DestroyFunc>(device, create(device, createInfo))
  {
  }

private:
  static T create(VkDevice device, const CreateInfo& createInfo)
  {
    T result = VK_NULL_HANDLE;
    NVVK_CHECK(CreateFunc(device, &createInfo, nullptr, &result));
    return result;
  }
};

using Semaphore   = VulkanObject<VkSemaphore, VkSemaphoreCreateInfo, vkCreateSemaphore, vkDestroySemaphore>;
using CommandPool = VulkanObject<VkCommandPool, VkCommandPoolCreateInfo, vkCreateCommandPool, vkDestroyCommandPool>;
using PipelineLayout = VulkanObject<VkPipelineLayout, VkPipelineLayoutCreateInfo, vkCreatePipelineLayout, vkDestroyPipelineLayout>;
using Pipeline     = VulkanHandle<VkPipeline, vkDestroyPipeline>;
using ShaderModule = VulkanObject<VkShaderModule, VkShaderModuleCreateInfo, vkCreateShaderModule, vkDestroyShaderModule>;

inline Semaphore makeTimelineSemaphore(VkDevice device, uint64_t initialValue)
{
  VkSemaphoreTypeCreateInfo timelineSemaphoreCreateInfo{.sType         = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                                                        .pNext         = nullptr,
                                                        .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
                                                        .initialValue  = initialValue};
  VkSemaphoreCreateInfo     semaphoreCreateInfo{
          .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &timelineSemaphoreCreateInfo, .flags = 0};
  return vkobj::Semaphore{device, semaphoreCreateInfo};
}

inline bool waitTimelineSemaphore(VkDevice device, VkSemaphore semaphore, uint64_t value, uint64_t timeout)
{
  VkSemaphoreWaitInfo waitInfo{
      .sType          = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
      .pNext          = nullptr,
      .flags          = 0,
      .semaphoreCount = 1,
      .pSemaphores    = &semaphore,
      .pValues        = &value,
  };
  VkResult r = vkWaitSemaphores(device, &waitInfo, timeout);
  if(r == VK_TIMEOUT)
    return false;
  NVVK_CHECK(r);
  return true;
}

class TimelineSubmitCancel : std::exception
{
};

// Timeline semaphore and value pair
// Using std::shared_future since the value is only known at the time of
// submission and multiple threads may be preparing a submission for a queue at
// once.
struct SemaphoreValue
{
  VkSemaphore                  semaphore = VK_NULL_HANDLE;
  std::shared_future<uint64_t> value;

  bool wait(VkDevice device, uint64_t timeout = std::numeric_limits<uint64_t>::max()) const
  {
    assert(semaphore != VK_NULL_HANDLE);
    assert(value.valid());
    if(timeout != std::numeric_limits<uint64_t>::max())
    {
      auto start = std::chrono::high_resolution_clock::now();
      if(value.wait_for(std::chrono::nanoseconds(timeout)) == std::future_status::timeout)
        return false;
      auto end = std::chrono::high_resolution_clock::now();
      timeout -= uint64_t((end > start ? std::chrono::duration_cast<std::chrono::nanoseconds>(end - start) :
                                         std::chrono::nanoseconds::zero())
                              .count());
    }
    try
    {
      return waitTimelineSemaphore(device, semaphore, value.get(), timeout);
    }
    catch(TimelineSubmitCancel&)
    {
      return false;
    }
  }

  // NOTE: may throw TimelineSubmitCancel
  [[nodiscard]] VkSemaphoreSubmitInfo submitInfo(VkPipelineStageFlags2 stageMask, uint32_t deviceIndex = 0)
  {
    assert(semaphore != VK_NULL_HANDLE);
    return {.sType       = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .pNext       = nullptr,
            .semaphore   = semaphore,
            .value       = value.get(),
            .stageMask   = stageMask,
            .deviceIndex = deviceIndex};
  }
};

// A VkQueue with a timeline semaphore that tracks submissions. Not thread safe.
class TimelineQueue
{
public:
  TimelineQueue(const TimelineQueue& other)            = delete;
  TimelineQueue& operator=(const TimelineQueue& other) = delete;
  TimelineQueue(VkDevice device, uint32_t familyIndex_, VkQueue queue_)
      : queue(queue_)
      , familyIndex(familyIndex_)
      , timelineNext(0)
      , timelineSemaphore(makeTimelineSemaphore(device, timelineNext++))
  {
    newPromise();
  }
  ~TimelineQueue()
  {
    // If anything is waiting, there will never be a value.
    promiseValue.set_exception(std::make_exception_ptr(TimelineSubmitCancel()));
  }

  [[nodiscard]] SemaphoreValue nextSubmitValue() const { return {timelineSemaphore, sharedFutureValue}; }

  // It is assumed the caller will guarantee a submit to the same queue before
  // any future calls to nextSubmitValue(). This is not thread safe. Ideally the
  // vkQueueSubmit2() call could be made by this object, wrapped with a mutex.
  [[nodiscard]] VkSemaphoreSubmitInfo submitInfoAndAdvance(VkPipelineStageFlags2 stageMask, uint32_t deviceIndex = 0)
  {
    VkSemaphoreSubmitInfo result = {.sType       = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                                    .pNext       = nullptr,
                                    .semaphore   = timelineSemaphore,
                                    .value       = timelineNext++,
                                    .stageMask   = stageMask,
                                    .deviceIndex = deviceIndex};
    promiseValue.set_value(result.value);
    newPromise();
    return result;
  }

  VkQueue  queue       = nullptr;
  uint32_t familyIndex = 0;

private:
  void newPromise()
  {
    promiseValue      = std::promise<uint64_t>();
    sharedFutureValue = std::shared_future<uint64_t>(promiseValue.get_future());
  }

  uint64_t                     timelineNext = 0;
  vkobj::Semaphore             timelineSemaphore;
  std::promise<uint64_t>       promiseValue;
  std::shared_future<uint64_t> sharedFutureValue;
};

// A memory owning nvvk::Buffer
class ByteBuffer
{
public:
  friend class ByteBufferMapping;

  // DANGER: allow creating an invalid buffer for delayed initialization
  ByteBuffer() = default;

  static constexpr VkBufferUsageFlags debugUsageFlags =
#if !defined(NDEBUG)
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT;  // used to download and dump data
#else
      0;
#endif

  ByteBuffer(const ByteBuffer& other) = delete;
  ByteBuffer(ByteBuffer&& other)
      : m_allocator(std::move(other.m_allocator))
      , m_buffer(std::move(other.m_buffer))
  {
    other.m_allocator = nullptr;
  }
  ByteBuffer(ResourceAllocator* allocator, VkDeviceSize size, VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags propertyFlags)
      : m_allocator(allocator)
      // align to 4 bytes to guarantee assumption even though the allocator would do this anyway
      , m_buffer(allocator->createBuffer(nvh::align_up(size, 4), usageFlags | debugUsageFlags, propertyFlags))
  {
  }
  template <std::ranges::contiguous_range Range>
  ByteBuffer(ResourceAllocator* allocator, Range&& range, VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags propertyFlags, VkCommandBuffer cmd)
      : m_allocator(allocator)
      // align to 4 bytes to guarantee assumption even though the allocator would do this anyway
      , m_buffer(allocator->createBuffer(cmd, nvh::align_up(sizeof(*range.data()) * range.size(), 4), range.data(), usageFlags | debugUsageFlags, propertyFlags))
  {
  }
  ~ByteBuffer() { destroy(); }
  ByteBuffer& operator=(const ByteBuffer& other) = delete;
  ByteBuffer& operator=(ByteBuffer&& other) noexcept
  {
    destroy();
    m_allocator       = other.m_allocator;
    m_buffer          = other.m_buffer;
    other.m_allocator = nullptr;
    return *this;
  };
  const VkDeviceAddress& address() const { return m_buffer.address; }
  operator const VkBuffer&() const { return m_buffer.buffer; }

private:
  void destroy()
  {
    if(m_allocator)
      m_allocator->destroy(m_buffer);
  }

  ResourceAllocator* m_allocator = nullptr;
  nvvk::Buffer       m_buffer;
};

// Trivial wrapper to give a compile error when mixing up types after a
// copy/paste error. Allows the code to be more explicit about what is a host
// and device pointer with a little type safety.
template <class T>
struct DeviceAddress
{
  using value_type         = T;
  explicit DeviceAddress() = default;
  explicit DeviceAddress(VkDeviceAddress raw)
      : address(raw)
  {
  }
  VkDeviceAddress             address = 0xffffffffffffffffull;
  explicit                    operator VkDeviceAddress() const { return address; }
  explicit                    operator bool() const { return address != 0; }
  friend inline std::ostream& operator<<(std::ostream& os, const DeviceAddress& a) { return os << a.address; }
  int64_t                     operator-(const DeviceAddress& other)
  {
    // similar behaviour to pointer arithmetic where subtraction gives the
    // distance in elements
    int64_t size = address - other.address;
    assert(size % sizeof(T) == 0);
    return size / sizeof(T);
  }
};

template <class T, class U>
  requires(sizeof(T) == sizeof(U))
DeviceAddress<T> deviceReinterpretCast(DeviceAddress<U> address)
{
  return DeviceAddress<T>{VkDeviceAddress(address)};
}

template <class T>
T* translateOffset(DeviceAddress<T> offsetAddress, void* hostBase)
{
  if(!hostBase)
    return nullptr;
  return reinterpret_cast<T*>(reinterpret_cast<std::byte*>(hostBase) + offsetAddress.address);
}

template <class T>
DeviceAddress<T> translateOffset(DeviceAddress<T> offsetAddress, VkDeviceAddress deviceBase)
{
  if(!deviceBase)
    return DeviceAddress<T>{0};
  return DeviceAddress<T>{offsetAddress.address + deviceBase};
}

// A VkBuffer with an element type and count
template <class T>
  requires std::is_trivially_copyable_v<T>
class Buffer
{
public:
  template <class>
  friend class BufferMapping;

  using value_type = T;
  Buffer()         = default;  // Allow empty initialization
  Buffer(ResourceAllocator* allocator, size_t elementCount, VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags propertyFlags)
      : m_size(elementCount)
      , m_buffer(elementCount ? ByteBuffer(allocator, sizeof(T) * elementCount, usageFlags, propertyFlags) : ByteBuffer())
  {
  }
  template <std::ranges::contiguous_range Range>
    requires std::is_same_v<std::ranges::range_value_t<Range>, T>
  Buffer(ResourceAllocator* allocator, Range&& range, VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags propertyFlags, VkCommandBuffer cmd)
      : m_size(std::ranges::size(range))
      , m_buffer(m_size ? ByteBuffer(allocator, std::forward<Range>(range), usageFlags, propertyFlags, cmd) : ByteBuffer())
  {
  }

  size_t           size() const { return m_size; }
  size_t           size_bytes() const { return m_size * sizeof(value_type); }
  DeviceAddress<T> address(size_t offsetIndex = 0) const
  {
    assert(offsetIndex < m_size);
    return DeviceAddress<T>(m_buffer.address() + offsetIndex * sizeof(value_type));
  }

  operator const VkBuffer&() const { return m_buffer; }

  ByteBuffer moveByteBuffer()
  {
    m_size = 0;
    return std::move(m_buffer);
  }

private:
  size_t     m_size = 0;
  ByteBuffer m_buffer;
};

// Scoped mapping of a buffer to make sure not to miss a call to unmap()
class ByteBufferMapping
{
public:
  ByteBufferMapping()                               = default;
  ByteBufferMapping(const ByteBufferMapping& other) = delete;
  ByteBufferMapping(ByteBufferMapping&& other) noexcept
      : m_memAllocator(other.m_memAllocator)
      , m_memHandle(other.m_memHandle)
      , m_mapping(other.m_mapping)
  {
    other.m_mapping = nullptr;
  }
  ~ByteBufferMapping() { destroy(); }
  ByteBufferMapping& operator=(const ByteBufferMapping& other) = delete;
  ByteBufferMapping& operator=(ByteBufferMapping&& other) noexcept
  {
    destroy();
    m_memAllocator  = other.m_memAllocator;
    m_memHandle     = other.m_memHandle;
    m_mapping       = other.m_mapping;
    other.m_mapping = nullptr;
    return *this;
  }
  explicit ByteBufferMapping(const ByteBuffer& buffer)
      : m_memAllocator(buffer.m_allocator->getMemoryAllocator())
      , m_memHandle(buffer.m_buffer.memHandle)
      , m_mapping(m_memAllocator->map(m_memHandle))
  {
  }
  operator void*() const { return m_mapping; }
  explicit operator bool() const { return m_mapping != nullptr; }

private:
  void destroy()
  {
    if(m_mapping)
      m_memAllocator->unmap(m_memHandle);
  }
  nvvk::MemAllocator* m_memAllocator = nullptr;
  nvvk::MemHandle     m_memHandle    = nullptr;
  void*               m_mapping      = nullptr;
};

template <class T>
class BufferMapping
{
public:
  BufferMapping() = default;
  BufferMapping(Buffer<T>& buffer)
      : m_mapping(buffer.m_buffer)
      , m_size(buffer.size())
  {
  }
  std::span<T> span() const { return std::span(reinterpret_cast<T*>(static_cast<void*>(m_mapping)), m_size); }
  explicit     operator bool() const { return m_mapping; }

private:
  ByteBufferMapping m_mapping;
  size_t            m_size = 0;
};

// deduction guide for deducing the element type from a range
template <std::ranges::contiguous_range Range>
Buffer(ResourceAllocator*, Range&&, VkBufferUsageFlags, VkMemoryPropertyFlags, VkCommandBuffer)
    -> Buffer<std::ranges::range_value_t<Range>>;

// Command buffer wrapper that calls vkFreeCommandBuffers() on destruction (not
// for use with vkResetCommandPool)
class CommandBuffer
{
public:
  // DANGER: Allow delayed initialization
  CommandBuffer() noexcept = default;
  CommandBuffer(VkDevice device, const VkCommandBufferAllocateInfo& allocateInfo)
      : m_device(device)
      , m_pool(allocateInfo.commandPool)
  {
    NVVK_CHECK(vkAllocateCommandBuffers(m_device, &allocateInfo, &m_cmd));
  }
  ~CommandBuffer() { destroy(); }
  operator VkCommandBuffer() const { return m_cmd; }
  explicit operator bool() const { return m_cmd != VK_NULL_HANDLE; }
  CommandBuffer(const CommandBuffer& other) = delete;
  CommandBuffer(CommandBuffer&& other) noexcept
      : m_device(other.m_device)
      , m_pool(other.m_pool)
      , m_cmd(other.m_cmd)
  {
    other.m_cmd = VK_NULL_HANDLE;
  }
  CommandBuffer& operator=(const CommandBuffer& other) = delete;
  CommandBuffer& operator=(CommandBuffer&& other) noexcept
  {
    destroy();
    m_device    = other.m_device;
    m_pool      = other.m_pool;
    m_cmd       = other.m_cmd;
    other.m_cmd = VK_NULL_HANDLE;
    return *this;
  }

private:
  void destroy()
  {
    if(m_device != VK_NULL_HANDLE)
      vkFreeCommandBuffers(m_device, m_pool, 1, &m_cmd);
  }
  VkDevice        m_device = VK_NULL_HANDLE;
  VkCommandPool   m_pool   = VK_NULL_HANDLE;
  VkCommandBuffer m_cmd    = VK_NULL_HANDLE;
};

// Single, recording, primary command buffer with no inheritance info.
class BuildingCommandBuffer
{
public:
  // DANGER: Allow delayed initialization
  BuildingCommandBuffer() noexcept                                         = default;
  BuildingCommandBuffer(BuildingCommandBuffer&& other) noexcept            = default;
  BuildingCommandBuffer& operator=(BuildingCommandBuffer&& other) noexcept = default;
  BuildingCommandBuffer(VkDevice device, VkCommandPool pool, VkCommandBufferUsageFlags flags = 0)
      : m_cmd(device,
              VkCommandBufferAllocateInfo{
                  .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                  .pNext              = nullptr,
                  .commandPool        = pool,
                  .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                  .commandBufferCount = 1,
              })
  {
    VkCommandBufferBeginInfo beginInfo = {
        .sType            = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext            = nullptr,
        .flags            = flags,
        .pInheritanceInfo = nullptr,
    };
    NVVK_CHECK(vkBeginCommandBuffer(m_cmd, &beginInfo));
  }
  operator VkCommandBuffer() const { return m_cmd; }
  explicit operator bool() const { return m_cmd != VK_NULL_HANDLE; }

  CommandBuffer&& endRecording()
  {
    NVVK_CHECK(vkEndCommandBuffer(m_cmd));
    return std::move(m_cmd);
  }

  // Verify endRecording() was called, taken ownership. It's probably a bug if a
  // command buffer was destroyed during recording
  ~BuildingCommandBuffer() { assert(!m_cmd); }

private:
  CommandBuffer m_cmd;
};

class ReadyCommandBuffer
{
public:
  ReadyCommandBuffer() = default;
  explicit ReadyCommandBuffer(BuildingCommandBuffer&& cmd)
      : m_cmd(cmd.endRecording())
  {
  }
  void submit(VkQueue queue) const
  {
    VkCommandBuffer cmd    = m_cmd;
    VkSubmitInfo    submit = {
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
  }
  void submitAfter(VkQueue queue, std::span<VkSemaphoreSubmitInfo> waits) const { submit(queue, waits, {}); }
  void submitAndSignal(VkQueue queue, std::span<VkSemaphoreSubmitInfo> signals) const { submit(queue, {}, signals); }
  void submit(VkQueue queue, std::span<VkSemaphoreSubmitInfo> waits, std::span<VkSemaphoreSubmitInfo> signals) const
  {
    VkCommandBufferSubmitInfo cmdInfo{
        .sType         = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
        .pNext         = nullptr,
        .commandBuffer = m_cmd,
        .deviceMask    = 0,  // all
    };
    VkSubmitInfo2 submit = {
        .sType                    = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .pNext                    = nullptr,
        .flags                    = 0,
        .waitSemaphoreInfoCount   = uint32_t(waits.size()),
        .pWaitSemaphoreInfos      = waits.data(),
        .commandBufferInfoCount   = 1,
        .pCommandBufferInfos      = &cmdInfo,
        .signalSemaphoreInfoCount = uint32_t(signals.size()),
        .pSignalSemaphoreInfos    = signals.data(),
    };
    NVVK_CHECK(vkQueueSubmit2(queue, 1, &submit, VK_NULL_HANDLE));
  }
  void submit(TimelineQueue& queue, VkPipelineStageFlags2 stageMask)
  {
    VkCommandBufferSubmitInfo cmdInfo{
        .sType         = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
        .pNext         = nullptr,
        .commandBuffer = m_cmd,
        .deviceMask    = 0,  // all
    };
    VkSemaphoreSubmitInfo signalSubmitInfo = queue.submitInfoAndAdvance(stageMask);
    VkSubmitInfo2         submit           = {
                          .sType                    = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
                          .pNext                    = nullptr,
                          .flags                    = 0,
                          .waitSemaphoreInfoCount   = 0u,
                          .pWaitSemaphoreInfos      = nullptr,
                          .commandBufferInfoCount   = 1,
                          .pCommandBufferInfos      = &cmdInfo,
                          .signalSemaphoreInfoCount = 1u,
                          .pSignalSemaphoreInfos    = &signalSubmitInfo,
    };
    NVVK_CHECK(vkQueueSubmit2(queue.queue, 1, &submit, VK_NULL_HANDLE));
  }
  explicit        operator bool() const { return static_cast<VkCommandBuffer>(m_cmd) != VK_NULL_HANDLE; }
  VkCommandBuffer get() const { return m_cmd; }

private:
  CommandBuffer m_cmd;
};

// A command buffer that submits itself to the queue when it goes out of scope
template <class Queue>
class ImmediateCommandBuffer;

// TODO: remove?
template <>
class ImmediateCommandBuffer<VkQueue>
{
public:
  ImmediateCommandBuffer(VkDevice device, VkCommandPool pool, VkQueue queue)
      : m_cmd(device, pool, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)
      , m_queue(queue)
  {
  }
  ImmediateCommandBuffer(const ImmediateCommandBuffer& other)            = delete;
  ImmediateCommandBuffer(ImmediateCommandBuffer&& other) noexcept        = default;
  ImmediateCommandBuffer& operator=(const ImmediateCommandBuffer& other) = delete;
  ImmediateCommandBuffer& operator=(ImmediateCommandBuffer&& other) noexcept
  {
    destroy();
    m_cmd   = std::move(other.m_cmd);
    m_queue = other.m_queue;
    return *this;
  }
  ~ImmediateCommandBuffer() { destroy(); }
  operator VkCommandBuffer() const { return m_cmd; }

private:
  void destroy()
  {
    if(m_cmd)
    {
      ReadyCommandBuffer recorded(std::move(m_cmd));
      recorded.submit(m_queue);
      NVVK_CHECK(vkQueueWaitIdle(m_queue));
    }
  }
  BuildingCommandBuffer m_cmd;
  VkQueue               m_queue;
};

// deduction guide
ImmediateCommandBuffer(VkDevice, VkCommandPool, VkQueue) -> ImmediateCommandBuffer<VkQueue>;

// TODO: consolidate
template <>
class ImmediateCommandBuffer<TimelineQueue>
{
public:
  ImmediateCommandBuffer(VkDevice device, VkCommandPool pool, TimelineQueue& queue, VkPipelineStageFlags2 stageMask)
      : m_cmd(device, pool, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)
      , m_queue(&queue)
      , m_stageMask(stageMask)
  {
  }
  ImmediateCommandBuffer(const ImmediateCommandBuffer& other)            = delete;
  ImmediateCommandBuffer(ImmediateCommandBuffer&& other) noexcept        = default;
  ImmediateCommandBuffer& operator=(const ImmediateCommandBuffer& other) = delete;
  ImmediateCommandBuffer& operator=(ImmediateCommandBuffer&& other) noexcept
  {
    destroy();
    m_cmd       = std::move(other.m_cmd);
    m_queue     = other.m_queue;
    m_stageMask = other.m_stageMask;
    return *this;
  }
  ~ImmediateCommandBuffer() { destroy(); }
  operator VkCommandBuffer() const { return m_cmd; }

private:
  void destroy()
  {
    if(m_cmd)
    {
      ReadyCommandBuffer recorded(std::move(m_cmd));
      recorded.submit(*m_queue, m_stageMask);
      NVVK_CHECK(vkQueueWaitIdle(m_queue->queue));
    }
  }
  BuildingCommandBuffer m_cmd;
  TimelineQueue*        m_queue = nullptr;
  VkPipelineStageFlags2 m_stageMask;
};

// deduction guide
ImmediateCommandBuffer(VkDevice, VkCommandPool, TimelineQueue&, VkPipelineStageFlags2) -> ImmediateCommandBuffer<TimelineQueue>;


// Vulkan objects common to a thread doing vulkan work. This includes a
// non-thread safe StagingMemoryManager in the allocator.
struct Context
{
  Context(VkDevice device_, VkPhysicalDevice physicalDevice_, nvvk::MemAllocator* memAllocator, uint32_t queueFamilyIndex_, VkQueue queue_)
      : device(device_)
      , physicalDevice(physicalDevice_)
      , commandPool(device,
                    VkCommandPoolCreateInfo{
                        .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                        .pNext            = nullptr,
                        .flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
                        .queueFamilyIndex = queueFamilyIndex_,
                    })
      , queue(queue_)
      , queueFamilyIndex(queueFamilyIndex_)
      , allocatorObj(std::make_unique<ResourceAllocator>(device,
                                                         physicalDevice,
                                                         memAllocator,
                                                         NVVK_DEFAULT_STAGING_BLOCKSIZE,
                                                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT /* for cmdFromAddressNV */))
      , allocator(allocatorObj.get())  // to save calling .get() everywhere
  {
  }
  VkDevice                           device;
  VkPhysicalDevice                   physicalDevice;
  vkobj::CommandPool                 commandPool;  // reusable single-shot command buffers
  VkQueue                            queue;        // owned by this thread (note, there are better designs)
  uint32_t                           queueFamilyIndex;
  std::unique_ptr<ResourceAllocator> allocatorObj;  // polymorphic
  ResourceAllocator*                 allocator;     // shortcut
};

// Returns a mapped staging buffer, valid until the context's allocator's
// staging memory is finalized
template <class T>
std::span<const T> downloadNow(ResourceAllocator* allocator, VkCommandPool pool, VkQueue queue, Buffer<T>& array)
{
  ImmediateCommandBuffer cmd(allocator->getDevice(), pool, queue);
  return cmdFromArray(*allocator->getStaging(), cmd, array);
}

// Uploads source to the destination array
template <class T, std::ranges::range Range>
void uploadNow(ResourceAllocator* allocator, VkCommandPool pool, VkQueue queue, const Range& source, Buffer<T>& destination)
{
  std::span<T> mapped;
  {
    ImmediateCommandBuffer cmd(allocator->getDevice(), pool, queue);
    mapped = cmdToArray(*allocator->getStaging(), cmd, destination);
  }
  std::ranges::copy(source, mapped);
}

template <class T>
std::span<const T> cmdFromArray(nvvk::StagingMemoryManager& staging, VkCommandBuffer cmd, const Buffer<T>& array)
{
  return std::span(staging.cmdFromBufferT<const T>(cmd, array, 0, array.size_bytes()), array.size());
}

template <class T>
std::span<T> cmdToArray(nvvk::StagingMemoryManager& staging, VkCommandBuffer cmd, const Buffer<T>& array)
{
  return std::span(staging.cmdToBufferT<T>(cmd, array, 0, array.size_bytes()), array.size());
}

template <std::ranges::contiguous_range Range, class T>
  requires std::is_same_v<std::ranges::range_value_t<Range>, T>
void cmdToArray(nvvk::StagingMemoryManager& staging, VkCommandBuffer cmd, Range&& input, const Buffer<T>& array)
{
  if(std::ranges::size(input) > array.size())
    throw std::runtime_error("cmdToArray input out of bounds");
  staging.cmdToBuffer(cmd, array, 0, std::ranges::size(input) * sizeof(T), std::ranges::data(input));
}

}  // namespace vkobj

inline void memoryBarrier(VkCommandBuffer cmd, VkAccessFlags srcAccess, VkAccessFlags dstAccess, VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage)
{
  VkMemoryBarrier memoryBarrier = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .pNext = nullptr, .srcAccessMask = srcAccess, .dstAccessMask = dstAccess};
  vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);
}
