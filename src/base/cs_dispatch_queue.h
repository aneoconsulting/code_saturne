#pragma once

#include "base/cs_dispatch.h"

#include <type_traits>

//! Forces synchronous execution of tasks, even on GPU.
#ifndef CS_DISPATCH_QUEUE_FORCE_SYNC
#define CS_DISPATCH_QUEUE_FORCE_SYNC 0
#endif

#if defined(__CUDACC__)
#include <cuda.h>
#include <cuda_runtime.h>
#else
#endif

#include <chrono>
#include <initializer_list>
#include <tuple>
#include <utility>

struct cs_event;
struct cs_task;

//! Represents an event to synchronize with. Often the end of a cs_device_task.
struct cs_event {
  using underlying_type =
#if defined(__CUDACC__)
    cudaEvent_t;
#else
    std::chrono::steady_clock::time_point;
#endif

  underlying_type event_impl;

  // Cosntruction/Destruction

  cs_event()
  {
#if defined(__CUDACC__)
    cudaEventCreate(&event_impl);
#endif
  }

  cs_event(cs_event const &other) = delete;
  cs_event &
  operator=(cs_event const &) = delete;

  cs_event(cs_event &&other)
#if defined(__CUDACC__)
  {
    event_impl       = other.event_impl;
    other.event_impl = nullptr;
  }
#else
    = default;
#endif

  cs_event &
  operator=(cs_event &&other)
#if defined(__CUDACC__)
  {
    if (event_impl != nullptr) {
      cudaEventDestroy(event_impl);
    }

    event_impl       = other.event_impl;
    other.event_impl = nullptr;

    return *this;
  }
#else
    = default;
#endif

  //! Returns the underlying implementation
  underlying_type &
  operator~()
  {
    return event_impl;
  }

  ~cs_event()
#if defined(__CUDACC__)
  {
    if (event_impl != nullptr) {
      cudaEventDestroy(event_impl);
    }
  }
#else
    = default;
#endif

  // Actions

  void
  wait()
  {
#if defined(__CUDACC__)
    cudaEventSynchronize(event_impl);
#endif
  }
};

//! cs_event_ref is a reference wrapper to a cs_event.
class cs_event_ref {
  cs_event *event_ptr;

public:
  using underlying_type = typename cs_event::underlying_type;

  cs_event_ref(cs_event &event) : event_ptr(&event) {}

  cs_event_ref() = delete;

  cs_event_ref(cs_event_ref &&other)      = default;
  cs_event_ref(cs_event_ref const &other) = default;
  cs_event_ref &
  operator=(cs_event_ref &&) & = default;
  cs_event_ref &
  operator=(cs_event_ref const &) & = default;

  //! Arrow operator to access members of the pointed event.
  cs_event *
  operator->()
  {
    return event_ptr;
  }

  //! Dereference operator to access the pointed event.
  cs_event &
  operator*()
  {
    return *event_ptr;
  }

  //! Dereference operator to access the underlying implementation
  //! of the pointed event.
  underlying_type &
  operator~()
  {
    return ~(*event_ptr);
  }
};

//! A cs_task_t object represents a task that can be syncronized to and with.
//! It holds a cs_dispatch_context with a unique CUDA stream and CUDA events can
//! be recorded from the task to synchronize other tasks with it.
//!
//! cs_task_t objects are meant to be spawned from a cs_queue_t object.
class cs_task {
  cs_dispatch_context context_;

  //! Event created at the creation of the task
  cs_event start_event;

  //! Last synchronization event
  cs_event end_event;

public:
  cs_task(cs_task const &) = delete;
  cs_task &
  operator=(cs_task const &) = delete;

  cs_task(cs_task &&) = default;
  cs_task &
  operator=(cs_task &&) = default;

  //! Creates a new task with a given context and initializes a new stream.
  cs_task(cs_dispatch_context context = {}) : context_(std::move(context))
  {
#if defined(__CUDACC__)
    cudaStream_t new_stream;
    cudaStreamCreate(&new_stream);
    context_.set_cuda_stream(new_stream);
    cudaEventRecord(~start_event, context_.cuda_stream());
#else
    ~start_event = std::chrono::steady_clock::now();
#endif
  }

  //! Adds an event to wait for
  void
  add_dependency(cs_event_ref event)
  {
#if defined(__CUDACC__)
    cudaStreamWaitEvent(context_.cuda_stream(), ~event);
#endif
  }

  //! Waits for all the events in sync_events.
  //! Elements of sync_events must be convertible to cs_event_t.
  void
  add_dependency(std::initializer_list<cs_event_ref> const &sync_events)
  {
#if defined(__CUDACC__)
    for (auto const &event : sync_events) {
      add_dependency(event);
    }
#endif
  }

  //! Waits for task termination.
  void
  wait()
  {
    end_event.wait();
  }

  //! Records an event from the task and returns a cs_event_ref to it.
  cs_event_ref
  record_end_event()
  {
#if defined(__CUDACC__)
    cudaEventRecord(~end_event, context_.cuda_stream());
#else
    ~end_event = std::chrono::steady_clock::now();
#endif
    return { end_event };
  }

  //! Calls record_event() to implicitely convert a task to an event.
  operator cs_event_ref() { return end_event; }

  //! Returns a reference to the context.
  cs_dispatch_context &
  get_context()
  {
    return context_;
  }

  //! Returns a reference to the start event.
  cs_event_ref
  get_start_event()
  {
    return start_event;
  }

  //! Returns a reference to the end event.
  cs_event_ref
  get_end_event()
  {
    return start_event;
  }

  //! Waits for task termination and destroys the associated CUDA stream.
  ~cs_task()
  {
    context_.wait();
#if defined(__CUDACC__)
    cudaStreamDestroy(context_.cuda_stream());
#endif
  }
};

//! cs_host_task extends cs_device_task to add support for host function tasks.
template <class FunctionType, class... Args>
class cs_host_task : public cs_task {
public:
  cs_host_task(cs_host_task const &) = delete;
  cs_host_task &
  operator=(cs_host_task const &) = delete;

  cs_host_task(cs_host_task &&) = default;
  cs_host_task &
  operator=(cs_host_task &&) = default;

  //! Tuple type for argument storage.
  using args_tuple_t = std::tuple<Args...>;

  //! Tuple type for function (possibly a lambda with captures)
  //! and argument storage.
  using data_tuple_t =
#if defined(__CUDACC__)
    std::tuple<FunctionType, args_tuple_t>;
#else
    std::tuple<FunctionType>;
#endif

private:
  //! Tuple that contains the function (possibly a lambda with captures)
  //! and the arguments used to invoke it.
  data_tuple_t data_tuple_;

public:
  //! Initializes a host task with given function and context.
  //! The function must be launched using the launch method.
  cs_host_task(FunctionType &&function, cs_dispatch_context context)
    : cs_task(std::move(context)),

#if defined(__CUDACC__)
      data_tuple_(std::move(function), args_tuple_t{})
#else
      data_tuple_(std::move(function))
#endif
  {
  }

  //! Launches the host function asynchronously using the given parameters
  //! with cudaLaunchHostFunc.
#if defined(__CUDACC__)
  cudaError_t
#else
  void
#endif
  launch(Args... args)
  {
#if defined(__CUDACC__)
    if (this->get_context().use_gpu()) {
      // Setting the arguments
      std::get<1>(data_tuple_) = args_tuple_t{ std::move(args)... };

      // Async launch on the task's own stream
      return cudaLaunchHostFunc(
        get_context().cuda_stream(),
        // Wrapper lambda: unwraps the parameter passed as a void* pointer
        // to invoke the host function
        [](void *data_tuple_ptr) -> void {
          auto &[f, args_tuple] = *(data_tuple_t *)(data_tuple_ptr);
          std::apply(f, args_tuple);
        },
        &data_tuple_);
    }
    else {
      this->record_end_event();
      this->wait();
      std::get<0>(data_tuple_)(args...);
      return cudaSuccess;
    }
#else
    std::get<0>(data_tuple_)(args...);
#endif
  }

  //! Waits for task termination.
  ~cs_host_task()
  {
    // We must wait host task termination to avoid data_tuple_
    // to be unstacked before the task is executed
    wait();
  }
};

//! Uses the execution model from base/cs_dispatch.h to create SYCL-like tasks
//! that can be synchronized together.
class cs_dispatch_queue {
public:
  //! Context used to initialize tasks.
  cs_dispatch_context initializer_context;

  template <class F, class... Args>
  cs_task
  parallel_for(cs_lnum_t n, F &&f, Args &&...args)
  {
    cs_task new_task(initializer_context);
    new_task.get_context().parallel_for(n,
                                        std::forward<F>(f),
                                        std::forward<Args>(args)...);
    new_task.record_end_event();
    return new_task;
  }

  template <class F, class... Args>
  cs_task
  parallel_for(cs_lnum_t                                  n,
               std::initializer_list<cs_event_ref> const &sync_events,
               F                                        &&f,
               Args &&...args)
  {
    cs_task new_task(initializer_context);
    new_task.add_dependency(sync_events);
    new_task.get_context().parallel_for(n,
                                        std::forward<F>(f),
                                        std::forward<Args>(args)...);
    new_task.record_end_event();
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
    new_task.record_end_event();
    return new_task;
  }

  template <class M, class F, class... Args>
  cs_task
  parallel_for_i_faces(const M                                   *m,
                       std::initializer_list<cs_event_ref> const &sync_events,
                       F                                        &&f,
                       Args &&...args)
  {
    cs_task new_task(initializer_context);
    new_task.add_dependency(sync_events);
    new_task.get_context().parallel_for_i_faces(m,
                                                std::forward<F>(f),
                                                std::forward<Args>(args)...);
    new_task.record_end_event();
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
    new_task.record_end_event();
    return new_task;
  }

  template <class M, class F, class... Args>
  cs_task
  parallel_for_b_faces(const M                                   *m,
                       std::initializer_list<cs_event_ref> const &sync_events,
                       F                                        &&f,
                       Args &&...args)
  {
    cs_task new_task(initializer_context);
    new_task.add_dependency(sync_events);
    new_task.get_context().parallel_for_b_faces(m,
                                                std::forward<F>(f),
                                                std::forward<Args>(args)...);
    new_task.record_end_event();
    return new_task;
  }

  template <class T, class F, class... Args>
  cs_task
  parallel_for_reduce_sum(cs_lnum_t n, T &sum, F &&f, Args &&...args)
  {
    cs_task new_task(initializer_context);
    new_task.get_context().parallel_for_reduce_sum(n,
                                                   sum,
                                                   std::forward<F>(f),
                                                   std::forward<Args>(args)...);
    new_task.record_end_event();
    return new_task;
  }

  template <class T, class F, class... Args>
  cs_task
  parallel_for_reduce_sum(
    cs_lnum_t                                  n,
    std::initializer_list<cs_event_ref> const &sync_events,
    T                                         &sum,
    F                                        &&f,
    Args &&...args)
  {
    cs_task new_task(initializer_context);
    new_task.add_dependency(sync_events);
    new_task.get_context().parallel_for_reduce_sum(n,
                                                   sum,
                                                   std::forward<F>(f),
                                                   std::forward<Args>(args)...);
    new_task.record_end_event();
    return new_task;
  }

  template <class T, class R, class F, class... Args>
  cs_task
  parallel_for_reduce(cs_lnum_t n, T &r, R &reducer, F &&f, Args &&...args)
  {
    cs_task new_task(initializer_context);
    new_task.get_context().parallel_for_reduce(n,
                                               r,
                                               reducer,
                                               std::forward<F>(f),
                                               std::forward<Args>(args)...);
    new_task.record_end_event();
    return new_task;
  }

  template <class T, class R, class F, class... Args>
  cs_task
  parallel_for_reduce(cs_lnum_t                                  n,
                      std::initializer_list<cs_event_ref> const &sync_events,
                      T                                         &r,
                      R                                         &reducer,
                      F                                        &&f,
                      Args &&...args)
  {
    cs_task new_task(initializer_context);
    new_task.add_dependency(sync_events);
    new_task.get_context().parallel_for_reduce(n,
                                               r,
                                               reducer,
                                               std::forward<F>(f),
                                               std::forward<Args>(args)...);
    new_task.record_end_event();
    return new_task;
  }

  //! Initiates a single thread task that runs on the host.
  //! This variant accepts sync_events to synchronize with other tasks.
  template <class FunctionType, class... Args>
  cs_host_task<FunctionType, std::remove_reference_t<Args>...>
  single_task(std::initializer_list<cs_event_ref> const &sync_events,
              FunctionType                             &&host_function,
              Args &&...args)
  {
    cs_host_task<FunctionType, std::remove_reference_t<Args>...> new_task(
      std::move(host_function),
      initializer_context);
    new_task.add_dependency(sync_events);
    new_task.launch(std::forward<Args>(args)...);
    new_task.record_end_event();
    return new_task;
  }

  //! Initiates a single thread task that runs on the host.
  template <class FunctionType, class... Args>
  cs_host_task<FunctionType, std::remove_reference_t<Args>...>
  single_task(FunctionType &&host_function, Args &&...args)
  {
    cs_host_task<FunctionType, std::remove_reference_t<Args>...> new_task(
      std::move(host_function),
      initializer_context);
    new_task.launch(std::forward<Args>(args)...);
    new_task.record_end_event();
    return new_task;
  }
};

//! Duration type for elapsed time between two events
using cs_event_duration =
#if defined(__CUDACC__)
  // cudaEventElapsedTime_v2 gives a time in milliseconds
  // with a resolution of around 0.5 microseconds
  std::chrono::microseconds;
#else
  std::chrono::steady_clock::duration;
#endif

//! Returns elapsed time between two events.
inline cs_event_duration
cs_elapsed_time(cs_event_ref start, cs_event_ref end)
{
  start->wait();
  end->wait();

#if defined(__CUDACC__)
  // cudaEventElapsedTime_v2 gives a time in milliseconds
  // with a resolution of around 0.5 microseconds
  float result_ms;
  cudaEventElapsedTime_v2(&result_ms, ~start, ~end);
  return cs_event_duration{ long(result_ms * 1000.f) };
#else
  return ~end - ~start;
#endif
}

//! Returns elapsed time between the start and the end of a task.
inline cs_event_duration
cs_elapsed_time(cs_task &task)
{
  return cs_elapsed_time(task.get_start_event(), task.get_end_event());
}
