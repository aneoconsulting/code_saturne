#pragma once

#include "base/cs_base_accel.h"
#include "base/cs_dispatch.h"
#include "base/cs_mem.h"
#include "fvm/fvm_group.h"

#ifdef __CUDACC__

#include <cuda.h>
#include <cuda_runtime.h>

#include <memory>
#include <tuple>
#include <utility>
#include <vector>

struct cs_event_t;
struct cs_task;

struct cs_event_t {
  cudaEvent_t cuda_event;
  cs_event_t(cudaEvent_t event) : cuda_event(event) {}
};

//! A cs_task_t object represents a task that can be syncronized to and with.
//! It holds a cs_device_context with a unique CUDA stream and CUDA events can
//! be recorded from the task to synchronize other tasks with it.
//!
//! cs_task_t objects are meant to be spawned from a cs_queue_t object.

class cs_task {
  cs_device_context context_;

public:
  //! Creates a new task with a given context and initializes a new stream.
  cs_task(cs_device_context context) : context_(std::move(context))
  {
    cudaStream_t new_stream;
    cudaStreamCreate(&new_stream);
    context_.set_cuda_stream(new_stream);
  }

  //! Creates a new task with a cs_device_context initialized with a new stream.
  cs_task()
  {
    cudaStream_t new_stream;
    cudaStreamCreate(&new_stream);
    context_ = cs_device_context(new_stream);
  }

  void
  wait_for(cs_event_t const &event)
  {
    cudaStreamWaitEvent(context_.cuda_stream(), event.cuda_event);
  }

  //! Waits for all the events in sync_events.
  //! Elements of sync_events must be convertible to cs_event_t.
  void
  wait_for_range(std::initializer_list<cs_event_t> const &sync_events)
  {
    for (auto const &event : sync_events) {
      wait_for(event);
    }
  }

  //! Records an event from the task.
  cs_event_t
  record_event()
  {
    cudaEvent_t event;
    cudaEventCreate(&event);
    cudaEventRecord(event, context_.cuda_stream());
    return { event };
  }

  //! Calls record_event() to implicitely convert a task to an event.
  operator cs_event_t() { return record_event(); }

  cs_device_context &
  get_context()
  {
    return context_;
  }
};

template <class... Args>
class cs_single_task_wrapper : public cs_task {
  std::unique_ptr<void *> data_ptr_;

public:
  using function_t   = void (*)(Args...);
  using data_tuple_t = std::tuple<function_t, std::tuple<Args...>>;

  inline static constexpr auto wrapper_function = [](void *data_ptr) -> void {
    auto &[fun, args_tuple] = *(data_tuple_t *)(data_ptr);
  };

  cs_single_task_wrapper(cs_device_context context)
    : cs_task(std::move(context))
  {
  }

  bool
  set_task(function_t function, Args... args)
  {
    if (data_ptr_ != nullptr) {
      return false;
    }

    data_ptr_ =
      std::make_unique<data_tuple_t>(function,
                                     std::make_tuple(std::move(args)...));

    cudaLaunchHostFunc(get_context().cuda_stream(),
                       wrapper_function,
                       data_ptr_);

    return true;
  }
};

class cs_queue {
public:
  //! Context used to initialize tasks.
  cs_device_context initializer_context;

  // Loop over n elements
  // Must be redefined by the child class
  template <class F, class... Args>
  cs_task
  parallel_for(cs_lnum_t n, F &&f, Args &&...args)
  {
    cs_task new_task(initializer_context);
    new_task.get_context().parallel_for(n,
                                        std::forward<F>(f),
                                        std::forward<Args>(args)...);
    return new_task;
  }

  template <class F, class... Args>
  cs_task
  parallel_for(cs_lnum_t                                n,
               std::initializer_list<cs_event_t> const &sync_events,
               F                                      &&f,
               Args &&...args)
  {
    cs_task new_task(initializer_context);

    new_task.wait_for_range(sync_events);

    new_task.get_context().parallel_for(n,
                                        std::forward<F>(f),
                                        std::forward<Args>(args)...);
    return new_task;
  }

  template <class M, class F, class... Args>
  cs_task
  parallel_for_i_faces(const M *m, F &&f, Args &&...args)
  {
    cs_task new_task(initializer_context);
    new_task.get_context().parallel_for_i_faces(m,
                                                std::forward<F>(f),
                                                std::forward<Args>(args)...);
    return new_task;
  }

  template <class M, class F, class... Args>
  cs_task
  parallel_for_i_faces(const M                                 *m,
                       std::initializer_list<cs_event_t> const &sync_events,
                       F                                      &&f,
                       Args &&...args)
  {
    cs_task new_task(initializer_context);

    new_task.wait_for_range(sync_events);

    new_task.get_context().parallel_for_i_faces(m,
                                                std::forward<F>(f),
                                                std::forward<Args>(args)...);
    return new_task;
  }

  template <class M, class F, class... Args>
  cs_task
  parallel_for_b_faces(const M *m, F &&f, Args &&...args)
  {
    cs_task new_task(initializer_context);
    new_task.get_context().parallel_for_b_faces(m,
                                                std::forward<F>(f),
                                                std::forward<Args>(args)...);
    return new_task;
  }

  template <class M, class F, class... Args>
  cs_task
  parallel_for_b_faces(const M                                 *m,
                       std::initializer_list<cs_event_t> const &sync_events,
                       F                                      &&f,
                       Args &&...args)
  {
    cs_task new_task(initializer_context);

    new_task.wait_for_range(sync_events);

    new_task.get_context().parallel_for_b_faces(m,
                                                std::forward<F>(f),
                                                std::forward<Args>(args)...);
    return new_task;
  }

  template <class T, class F, class... Args>
  cs_task
  parallel_for_reduce_sum(cs_lnum_t n, T &sum, F &&f, Args &&...args)
  {
    cs_task new_task(initializer_context);
    parallel_for_reduce_sum(n,
                            sum,
                            std::forward<F>(f),
                            std::forward<Args>(args)...);
    return new_task;
  }

  template <class T, class F, class... Args>
  cs_task
  parallel_for_reduce_sum(cs_lnum_t                                n,
                          std::initializer_list<cs_event_t> const &sync_events,
                          T                                       &sum,
                          F                                      &&f,
                          Args &&...args)
  {
    cs_task new_task(initializer_context);

    new_task.wait_for_range(sync_events);

    parallel_for_reduce_sum(n,
                            sum,
                            std::forward<F>(f),
                            std::forward<Args>(args)...);
    return new_task;
  }

  template <class T, class R, class F, class... Args>
  cs_task
  parallel_for_reduce(cs_lnum_t n, T &r, R &reducer, F &&f, Args &&...args)
  {
    cs_task new_task(initializer_context);
    parallel_for_reduce(n,
                        r,
                        reducer,
                        std::forward<F>(f),
                        std::forward<Args>(args)...);
    return new_task;
  }

  template <class... Args>
  cs_single_task_wrapper<Args...>
  single_task(
    std::initializer_list<cs_event_t> const             &sync_events,
    typename cs_single_task_wrapper<Args...>::function_t host_function,
    Args &&...args)
  {
    cs_single_task_wrapper single_task(initializer_context);
    single_task.set_task(host_function, std::forward<Args>(args)...);
    return single_task;
  }
};

inline void
foo()
{
  cs_alloc_mode_t alloc_mode = CS_ALLOC_DEVICE;

  cs_queue Q;

  constexpr std::size_t N = 16 * 1024 * 1024;

  int *a, *b;

  CS_MALLOC_HD(a, N, int, CS_ALLOC_HOST_DEVICE);
  CS_MALLOC_HD(b, N, int, CS_ALLOC_DEVICE);

  // task A
  auto e1 = Q.parallel_for(N, [=](std::size_t id) { a[id] = 1; });

  // task B
  auto e2 = Q.parallel_for(N, [=](std::size_t id) { b[id] = 2; });

  // task C
  auto e3 =
    Q.parallel_for(N, { e1, e2 }, [=](std::size_t id) { a[id] += b[id]; });

  // task D

  auto single_task_args = std::make_tuple(a);

  using args_type = decltype(single_task_args);

  cs_single_task_wrapper<int *> single_task(Q.initializer_context);

  // Q.single_task(
  //   { e3 },
  //   [](int *data) -> void {
  //     using std::get;
  //     for (int i = 1; i < N; i++) {
  //       data[0] += data[i];
  //     }
  //
  //     data[0] /= 3;
  //   },
  //   a);

  CS_FREE_HD(a);
  CS_FREE_HD(b);
}

#endif // __CUDACC__
