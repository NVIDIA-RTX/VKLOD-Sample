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

#pragma once

#include "vk_mem_alloc.h"
#include <assert.h>
#include <bit>
#include <exception>
#include <memory>
#include <mutex>
#include <nvvk/memallocator_vk.hpp>
#include <set>
#include <vulkan/vulkan_core.h>

// A remote allocator that suballocates a given range of memory. Uses a sorted
// binary tree free list.
// TODO: merge freed allocations in the free list
class PoolAllocator
{
  struct Range
  {
    VkDeviceAddress address;
    VkDeviceSize    size;
    bool            operator<(const Range& other) const
    {
      return size == other.size ? address < other.address : size < other.size;
    }
  };

public:
  PoolAllocator(VkDeviceAddress address, VkDeviceSize size)
      : m_base(address)
      , m_bytesTotal(size)
      , m_initialBlockSize(size)
      , m_freeList({{m_base, m_bytesTotal}})
  {
  }

  VkDeviceAddress allocate(VkDeviceSize userSize, VkDeviceSize align)
  {
    assert(userSize >= align);  // alignment should always be at least the size
    std::lock_guard lk(m_mutex);

    VkDeviceSize allocSize = adjustSize(userSize);

    // Binary search to find free allocation
    auto it = m_freeList.lower_bound({0, allocSize});
    for(; it != m_freeList.end(); ++it)
    {
      VkDeviceSize freeSize = it->size;
      assert(allocSize <= it->size);
      void* resultVP = reinterpret_cast<void*>(it->address);
      static_assert(sizeof(void*) == sizeof(VkDeviceAddress));  // TODO: x32 ... ?
      size_t remaining = it->size;
      if(std::align(align, allocSize, resultVP, remaining))
      {
        VkDeviceAddress result = reinterpret_cast<VkDeviceAddress>(resultVP);
        // Reinsert any remaining space in the free block
        m_freeList.erase(it);
        assert(remaining >= allocSize);
        if(remaining > allocSize)
          m_freeList.insert({result + allocSize, remaining - allocSize});

        // Track memory usage, excluding alignment and fragmentation overhead
        m_bytesAllocated += userSize;
        m_fragmentationInternal += allocSize - userSize;
        if(freeSize == m_initialBlockSize)
          m_initialBlockSize = remaining - allocSize;  // track one block size to remove from m_fragmentationOuter
        else
          m_fragmentationExternal -= freeSize;  // reusing freed blocks reduces the outer fragmentation
        return result;
      }
      // else, does not fit due to alignment
    }
    throw std::bad_alloc();
  }

  void deallocate(VkDeviceAddress address, VkDeviceSize userSize) noexcept
  {
    std::lock_guard lk(m_mutex);
    m_bytesAllocated -= userSize;
    VkDeviceSize allocSize = adjustSize(userSize);  // match the adjustment done in allocate()
    try
    {
      [[maybe_unused]] auto [it, inserted] = m_freeList.insert({address, allocSize});
      assert(inserted);  // should never collide
    }
    catch(const std::bad_alloc&)
    {
      std::terminate();  // fatal. mostly for coverity. could also just leak
    }
    //fprintf(stderr, "Free list size: %zu\n", m_freeList.size());
    assert(m_fragmentationInternal >= allocSize - userSize);
    m_fragmentationInternal -= allocSize - userSize;
    m_fragmentationExternal += allocSize;
  }

  VkDeviceSize offsetOf(VkDeviceAddress address) const { return address - m_base; }

  VkDeviceSize bytesAllocated() const
  {
    std::lock_guard lk(m_mutex);
    return m_bytesAllocated;
  }

  VkDeviceSize size() const { return m_bytesTotal; }

  VkDeviceSize internalFragmentation() const { return m_fragmentationInternal; }

  VkDeviceSize externalFragmentation() const { return m_fragmentationExternal; }

  VkDeviceSize fragmentation() const { return internalFragmentation() + externalFragmentation(); }

private:
  static VkDeviceSize adjustSize(VkDeviceSize size)
  {
    // Grow size to next power of two to reduce fragmentation
    return std::bit_ceil(size);
  }

  VkDeviceAddress    m_base                  = 0;
  VkDeviceSize       m_bytesAllocated        = 0;
  VkDeviceSize       m_bytesTotal            = 0;
  VkDeviceSize       m_fragmentationInternal = 0;
  VkDeviceSize       m_fragmentationExternal = 0;
  VkDeviceSize       m_initialBlockSize      = 0;  // used to filter the initial block out of m_fragmentationOuter
  std::set<Range>    m_freeList;
  mutable std::mutex m_mutex;
};

// Move-only destructing memory allocation from PoolAllocator
class PoolMemory
{
public:
  PoolMemory()                        = default;
  PoolMemory(const PoolMemory& other) = delete;
  PoolMemory(PoolMemory&& other) noexcept
      : m_allocator(other.m_allocator)
      , m_address(other.m_address)
      , m_size(other.m_size)
  {
    other.m_allocator = nullptr;
  }
  PoolMemory& operator=(const PoolMemory& other) = delete;
  PoolMemory& operator=(PoolMemory&& other) noexcept
  {
    destroy();
    m_allocator = nullptr;
    std::swap(m_allocator, other.m_allocator);
    m_address = other.m_address;
    m_size    = other.m_size;
    return *this;
  }
  PoolMemory(PoolAllocator& allocator, VkDeviceSize size, VkDeviceSize align)
      : m_allocator(&allocator)
      , m_address(allocator.allocate(std::max(size, align), align))
      , m_size(size)
  {
  }
  ~PoolMemory() { destroy(); }
  operator VkDeviceAddress() const { return m_address; }

private:
  void destroy()
  {
    if(m_allocator)
      m_allocator->deallocate(m_address, m_size);
  }

  PoolAllocator*  m_allocator = nullptr;
  VkDeviceAddress m_address   = 0xffffffffffffffffull;
  VkDeviceSize    m_size      = 0;
};

class ScopedVmaAllocator
{
public:
  ScopedVmaAllocator(const VmaAllocatorCreateInfo& createInfo) { vmaCreateAllocator(&createInfo, &m_handle); }
  ~ScopedVmaAllocator() { free(); }
  ScopedVmaAllocator(const ScopedVmaAllocator& other) = delete;
  ScopedVmaAllocator(ScopedVmaAllocator&& other) noexcept
      : m_handle(other.m_handle)
  {
    other.m_handle = nullptr;
  }
  ScopedVmaAllocator& operator=(const ScopedVmaAllocator& other) = delete;
  ScopedVmaAllocator& operator=(ScopedVmaAllocator&& other) noexcept
  {
    free();
    m_handle = nullptr;
    std::swap(m_handle, other.m_handle);
    return *this;
  }
  operator VmaAllocator() const { return m_handle; }

private:
  void free()
  {
    if(m_handle)
      vmaDestroyAllocator(m_handle);
  }
  VmaAllocator m_handle = nullptr;
};

// A nvvk::VMAMemoryAllocator that owns its VmaAllocator. This works around
// limitations of nvvk::AllocVma, which does not expose the nvvk::MemAllocator
// interface needed to chain allocators.
class VMAMemAllocator : public nvvk::VMAMemoryAllocator
{
public:
  VMAMemAllocator(const VmaAllocatorCreateInfo& createInfo)
      : VMAMemAllocator(createInfo.device, createInfo.physicalDevice, ScopedVmaAllocator(createInfo))
  {
  }
  VMAMemAllocator(VkDevice device, VkPhysicalDevice physicalDevice, ScopedVmaAllocator&& vmaAllocator)
      : nvvk::VMAMemoryAllocator(device, physicalDevice, vmaAllocator)
      , m_vmaAllocator(std::move(vmaAllocator))
  {
  }

private:
  ScopedVmaAllocator m_vmaAllocator;
};

// An nvvk::ResourceAllocator that calls releaseStaging() in the destructor
class ResourceAllocator : public nvvk::ResourceAllocator
{
public:
  using nvvk::ResourceAllocator::ResourceAllocator;
  virtual ~ResourceAllocator()
  {
    if(getStaging())
      releaseStaging();
  }
};

// Synchonized wrapper around nvvk::MemAllocator
class SynchronizedMemAllocator : public nvvk::MemAllocator
{
public:
  SynchronizedMemAllocator(std::unique_ptr<nvvk::MemAllocator> memAlloc)
      : m_alloc(std::move(memAlloc))
  {
  }

  virtual nvvk::MemHandle allocMemory(const nvvk::MemAllocateInfo& allocInfo, VkResult* pResult = nullptr) override
  {
    std::lock_guard lock(m_mutex);
    return m_alloc->allocMemory(allocInfo, pResult);
  }

  virtual void freeMemory(nvvk::MemHandle memHandle) override
  {
    std::lock_guard lock(m_mutex);
    m_alloc->freeMemory(memHandle);
  }

  virtual MemInfo getMemoryInfo(nvvk::MemHandle memHandle) const override
  {
    std::lock_guard lock(m_mutex);
    return m_alloc->getMemoryInfo(memHandle);
  }

  virtual void* map(nvvk::MemHandle memHandle, VkDeviceSize offset = 0, VkDeviceSize size = VK_WHOLE_SIZE, VkResult* pResult = nullptr) override
  {
    std::lock_guard lock(m_mutex);
    return m_alloc->map(memHandle, offset, size, pResult);
  }

  virtual void unmap(nvvk::MemHandle memHandle) override
  {
    std::lock_guard lock(m_mutex);
    m_alloc->unmap(memHandle);
  }

  virtual VkDevice         getDevice() const override { return m_alloc->getDevice(); }
  virtual VkPhysicalDevice getPhysicalDevice() const override { return m_alloc->getPhysicalDevice(); }

private:
  std::unique_ptr<nvvk::MemAllocator> m_alloc;
  mutable std::mutex                  m_mutex;
};
