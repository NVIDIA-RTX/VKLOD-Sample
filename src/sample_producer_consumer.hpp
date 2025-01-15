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

#include <array>
#include <condition_variable>
#include <functional>
#include <initializer_list>
#include <mutex>

// Synchronized one producer/ one consumer queue of reusable items implemented
// as a circular/ring buffer. Each side must wait or check for items to be
// available before accessing them.
//
// Example blocking producer usage:
//  if ( pc.waitProduce() ){
//    pc.producerObject() = something produced.
//    pc.produceDone()
//  } else {
//    // producer consumer object was canceled,
//    // e.g. during object destruction.
//  }
//
// Example non blocking producer usage:
//  if ( pc.tryProduce() ){
//    pc.producerObject() = something produced.
//    pc.produceDone()
//  } else {
//    if ( pc.canceled() ){
//      // producer consumer object was canceled,
//      // e.g. during object destruction.
//    } else {
//      // queue is full, waiting for consumer
//      assert( pc.full() );
//    }
//  }
template <class T, size_t N>
class ProducerConsumer
{
public:
  // Allow default construction, provided T is default constructible
  ProducerConsumer() = default;

  // Forward objects to std::array for when T is not default constructible
  // TODO: this is far from generic/complete
  template <class... Args>
    requires std::is_move_constructible_v<T> && (sizeof...(Args) == N)
  ProducerConsumer(Args&&... args)
      : m_items{std::forward<Args>(args)...}
  {
  }

  ~ProducerConsumer()
  {
    // Unblock any threads waiting on this object
    cancel();
  }

  // Safer API with callbacks, guaranteeing order of internal calls
  bool tryProduce(const std::function<void(T&)>& produceCB) { return produce(false, produceCB); }
  bool waitProduce(const std::function<void(T&)>& produceCB) { return produce(true, produceCB); }
  bool tryConsume(const std::function<void(T&)>& consumeCB) { return consume(false, consumeCB); }
  bool waitConsume(const std::function<void(T&)>& consumeCB) { return consume(true, consumeCB); }
  bool maybeTryConsume(const std::function<bool(T&)>& consumeCB) { return consume(false, consumeCB); }
  bool maybeWaitConsume(const std::function<bool(T&)>& consumeCB) { return consume(true, consumeCB); }

  bool produce(bool block, const std::function<void(T&)>& produceCB)
  {
    // Wait for a product to be recycled by the consumer
    if(block)
    {
      if(!waitProduce())
        return false;
    }
    else if(!tryProduce())
      return false;

    // Compute the product without holding any locks
    //fprintf(stderr, "PRODUCE %p %zu %p (%zu)\n", this, m_produceCursor, &m_items[m_produceCursor], size());
    produceCB(producerObject());

    // Emit the product for consuming
    produceDone();
    return true;
  }

  bool consume(bool block, const std::function<void(T&)>& consumeCB)
  {
    // Wait for a product to be produced
    if(block)
    {
      if(!waitConsume())
        return false;
    }
    else if(!tryConsume())
      return false;

    // Consume the product without holding any locks
    //fprintf(stderr, "CONSUME %p %zu %p (%zu)\n", this, m_consumeCursor, &m_items[m_consumeCursor], size());
    consumeCB(consumerObject());

    // Return the product for reuse
    consumeDone();
    return true;
  }

  // Alternative consume that allows the callback to cancel by returning false.
  // This is useful when the object contains an is-ready fence that the user
  // wants to poll before consuming.
  bool maybeConsume(bool block, const std::function<bool(T&)>& consumeCB)
  {
    // Wait for a product to be produced
    if(block)
    {
      if(!waitConsume())
        return false;
    }
    else if(!tryConsume())
      return false;

    // Consume the product without holding any locks
    //fprintf(stderr, "CONSUME %p %zu %p (%zu)\n", this, m_consumeCursor, &m_items[m_consumeCursor], size());
    bool result = consumeCB(consumerObject());

    // Return the product for reuse
    if(result)
      consumeDone();

    return result;
  }

  // Unblocks any threads waiting on this object, e.g. in wait*() calls.
  void cancel()
  {
    std::lock_guard lock(m_mutex);
    m_cancel = true;
    m_produceCV.notify_one();
    m_returnCV.notify_one();
  }

  // Differentiate between failed produce due to a full queue or cancel() call
  bool canceled() const
  {
    std::lock_guard lock(m_mutex);
    return m_cancel;
  }

  [[nodiscard]] size_t size() const
  {
    std::lock_guard lock(m_mutex);
    return m_size;
  }

  [[nodiscard]] bool empty() const { return size() == 0; };
  [[nodiscard]] bool full() const { return size() == N; };

  // Facilitates auditing a pipeline of producer/consumers. Otherwise item
  // counts can hide inbetween calls to consume and produce.
  [[nodiscard]] size_t promisedSize() const
  {
    std::lock_guard lock(m_mutex);
    return m_size + (m_producePromised ? size_t(1) : size_t(0));
  }
  [[nodiscard]] bool promisedEmpty() const { return promisedSize() == 0; };

  // Called by producer to wait for consumer to empty the queue
  void drain()
  {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_returnCV.wait(lock, [this] { return emptyLocked_(); });
  }

  // Unsafe access to all items
  // TODO: remove and provide circular iterators
  const std::array<T, N>& storage() const { return m_items; }

private:
  // Alternative API. Producer is considered to have promised to call
  // produceDone() if tryToProduce() or waitProduce() returns true in order to
  // provide promisedSize()
  bool tryProduce()
  {
    std::lock_guard lock(m_mutex);
    bool            result = !m_cancel && !fullLocked_();
    m_producePromised      = result;
    return result;
  }
  bool waitProduce()
  {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_returnCV.wait(lock, [this] { return m_cancel || !fullLocked_(); });
    bool result       = !m_cancel;
    m_producePromised = result;
    return result;
  }
  T&   producerObject() { return m_items[m_produceCursor]; }
  void produceDone()
  {
    std::lock_guard lock(m_mutex);
    ++m_size;
    if(++m_produceCursor == N)
      m_produceCursor = 0;
    m_cancel          = false;
    m_producePromised = false;
    m_produceCV.notify_one();
  }
  bool tryConsume() const
  {
    std::lock_guard lock(m_mutex);
    return !m_cancel && !emptyLocked_();
  }
  bool waitConsume() const
  {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_produceCV.wait(lock, [this] { return m_cancel || !emptyLocked_(); });
    return !m_cancel;
  }
  T&   consumerObject() { return m_items[m_consumeCursor]; }
  void consumeDone()
  {
    std::lock_guard lock(m_mutex);
    --m_size;
    if(++m_consumeCursor == N)
      m_consumeCursor = 0;
    m_returnCV.notify_one();
  }

private:
  [[nodiscard]] bool fullLocked_() const { return m_size == N; }
  [[nodiscard]] bool emptyLocked_() const { return m_size == 0; }

  // A circular buffer of just begin/end pointers cannot differentiate between
  // empty and full. Either an empty element must be maintained or an additional
  // flag/size is needed. This implementation chooses the latter.
  std::array<T, N>                m_items;
  size_t                          m_produceCursor   = 0;
  size_t                          m_consumeCursor   = 0;
  size_t                          m_size            = 0;
  bool                            m_producePromised = false;
  bool                            m_cancel          = false;
  mutable std::mutex              m_mutex;
  mutable std::condition_variable m_produceCV;  // notified by consumer returning an item for reuse
  mutable std::condition_variable m_returnCV;   // notified by producer pushing an item to the consumer
};

#if 0
template <size_t N>
class ProducerConsumerTester
{
public:
  ProducerConsumerTester()
  {
    ProducerConsumer<int, N> pc;
    std::jthread             t([&] {
      bool running = true;
      int  expect  = 1000;
      while(running)
      {
        pc.consume(rand() % 2 == 0, [&](int v) {
          //fprintf(stderr, "%i\n", v);
          assert(v == expect--);
          running = v != 0;
        });
        if(rand() % 2 == 0)
          usleep(100);
      }
    });
    for(int i = 1000; i >= 0; --i)
    {
      pc.waitProduce([&](int& v) { v = i; });
      if(rand() % 2 == 0)
        usleep(100);
    }
  }
};
static inline ProducerConsumerTester<1> g_producerConsumerTest1;
static inline ProducerConsumerTester<2> g_producerConsumerTest2;
static inline ProducerConsumerTester<3> g_producerConsumerTest3;
#endif
