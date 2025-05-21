#pragma once

#include "base/cs_base_accel.h"
#include "base/cs_dispatch.h"
#include "base/cs_mem.h"

#include <type_traits>

//! Forces synchronous execution of tasks
#ifndef CS_DISPATCH_QUEUE_FORCE_SYNC
#define CS_DISPATCH_QUEUE_FORCE_SYNC 0
#endif

#if defined(__CUDACC__)
#include <cuda.h>
#include <cuda_runtime.h>
#else
#include <chrono>
#endif

#include <initializer_list>
#include <source_location>
#include <tuple>
#include <utility>

//! Represents an event to synchronize with. Often the end of a cs_device_task.
struct cs_event {
#if defined(__CUDACC__)
  cudaEvent_t cuda_event;
#else
  std::chrono::time_point timer_event;
#endif
};

//! A cs_task_t object represents a task that can be syncronized to and with.
//! It holds a cs_dispatch_context with a unique CUDA stream and CUDA events can
//! be recorded from the task to synchronize other tasks with it.
//!
//! cs_task_t objects are meant to be spawned from a cs_queue_t object.
class cs_task {
#if defined(__CUDACC__)
  cs_dispatch_context context_;
#endif

  //! Stores the source location of the task's creation for debugging
  std::source_location creation_location;

  //! Event created at the creation of the task
  cs_event creation_event;

public:
  //! Creates a new task with a given context and initializes a new stream.
  cs_task(cs_dispatch_context  context  = {},
          std::source_location location = std::source_location::current())
    : context_(std::move(context)), creation_location(location)
  {
#if defined(__CUDACC__) && CS_DISPATCH_QUEUE_FORCE_SYNC == 0
    cudaStream_t new_stream;
    cudaStreamCreate(&new_stream);
    context_.set_cuda_stream(new_stream);
#endif

    creation_event = record_event();
  }

  //! Adds an event to wait for
  void
  add_dependency(cs_event const &event)
  {
#if defined(__CUDACC__)
    cudaStreamWaitEvent(context_.cuda_stream(), event.cuda_event);
#endif
  }

  //! Waits for all the events in sync_events.
  //! Elements of sync_events must be convertible to cs_event_t.
  void
  add_dependency(std::initializer_list<cs_event> const &sync_events)
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
    context_.wait();
  }

  //! Records an event from the task.
  cs_event
  record_event()
  {
#if defined(__CUDA__)
    cudaEvent_t event;
    cudaEventCreate(&event);
    cudaEventRecord(event, context_.cuda_stream());
    return { event };
#else
    return { std::chrono::steady_clock::now() };
#endif
  }

  //! Calls record_event() to implicitely convert a task to an event.
  operator cs_event() { return record_event(); }

  cs_dispatch_context &
  get_context()
  {
    return context_;
  }

  cs_event
  get_creation_event() const
  {
    return creation_event;
  }

  std::source_location
  get_creation_location() const
  {
    return creation_location;
  }

#if defined(__CUDACC__) && CS_DISPATCH_QUEUE_FORCE_SYNC == 0
  ~cs_task() { cudaStreamDestroy(context_.cuda_stream()); }
#endif
};

//! cs_host_task extends cs_device_task to add support for host function tasks.
template <class FunctionType, class... Args>
class cs_host_task : public cs_task {
public:
  //! Tuple type for argument storage.
  using args_tuple_t =
#if defined(__CUDACC__) && CS_DISPATCH_QUEUE_FORCE_SYNC == 0
    std::tuple<Args...>;
#else
    void;
#endif

  //! Tuple type for function (possibly a lambda with captures)
  //! and argument storage.
  using data_tuple_t = std::tuple<FunctionType, args_tuple_t>;

private:
  //! Tuple that contains the function (possibly a lambda with captures)
  //! and the arguments used to invoke it.
  data_tuple_t data_tuple_;

public:
  //! Initializes a host task with given function and context.
  //! The function must be launched using the launch method.
  cs_host_task(FunctionType &&function, cs_dispatch_context context)
    : cs_task(std::move(context)),
      data_tuple_(std::move(function), args_tuple_t{})
  {
  }

  //! Launches the host function asynchronously using the given parameters
  //! with cudaLaunchHostFunc.
  cudaError_t
  launch(Args... args)
  {
#if defined(__CUDACC__) && CS_DISPATCH_QUEUE_FORCE_SYNC == 0
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
#else
    std::get<0>(data_tuple_)(args...);
#endif
  }

  ~cs_host_task()
  {
    // We must wait host task termination to avoid data_tuple_ to be destroyed
    // before the task is executed
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
    return new_task;
  }

  template <class F, class... Args>
  cs_task
  parallel_for(cs_lnum_t                              n,
               std::initializer_list<cs_event> const &sync_events,
               F                                    &&f,
               Args &&...args)
  {
    cs_task new_task(initializer_context);

    new_task.add_dependency(sync_events);

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
  parallel_for_i_faces(const M                               *m,
                       std::initializer_list<cs_event> const &sync_events,
                       F                                    &&f,
                       Args &&...args)
  {
    cs_task new_task(initializer_context);

    new_task.add_dependency(sync_events);

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
  parallel_for_b_faces(const M                               *m,
                       std::initializer_list<cs_event> const &sync_events,
                       F                                    &&f,
                       Args &&...args)
  {
    cs_task new_task(initializer_context);

    new_task.add_dependency(sync_events);

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
  parallel_for_reduce_sum(cs_lnum_t                              n,
                          std::initializer_list<cs_event> const &sync_events,
                          T                                     &sum,
                          F                                    &&f,
                          Args &&...args)
  {
    cs_task new_task(initializer_context);

    new_task.add_dependency(sync_events);

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

  template <class T, class R, class F, class... Args>
  cs_task
  parallel_for_reduce(cs_lnum_t                              n,
                      std::initializer_list<cs_event> const &sync_events,
                      T                                     &r,
                      R                                     &reducer,
                      F                                    &&f,
                      Args &&...args)
  {
    cs_task new_task(initializer_context);
    new_task.add_dependency(sync_events);
    parallel_for_reduce(n,
                        r,
                        reducer,
                        std::forward<F>(f),
                        std::forward<Args>(args)...);
    return new_task;
  }

  template <class FunctionType, class... Args>
  cs_host_task<FunctionType, std::remove_reference_t<Args>...>
  single_task(std::initializer_list<cs_event> const &sync_events,
              FunctionType                         &&host_function,
              Args &&...args)
  {
    cs_host_task<FunctionType, std::remove_reference_t<Args>...> single_task(
      std::move(host_function),
      initializer_context);

    single_task.add_dependency(sync_events);
    single_task.launch(std::forward<Args>(args)...);

    return single_task;
  }

  template <class FunctionType, class... Args>
  cs_host_task<FunctionType, std::remove_reference_t<Args>...>
  single_task(FunctionType &&host_function, Args &&...args)
  {
    cs_host_task<FunctionType, std::remove_reference_t<Args>...> single_task(
      std::move(host_function),
      initializer_context);
    single_task.launch(std::forward<Args>(args)...);
    return single_task;
  }
};

inline void
foo()
{
  // Inspired by:
  // https://enccs.github.io/sycl-workshop/task-graphs-synchronization/#how-to-specify-dependencies

  cs_dispatch_queue     Q;
  constexpr std::size_t N = 16 * 1024 * 1024;
  int                  *a, *b;

  CS_MALLOC_HD(a, N, int, CS_ALLOC_HOST_DEVICE);
  CS_MALLOC_HD(b, N, int, CS_ALLOC_DEVICE);

  {
    // Task A
    auto task_a = Q.parallel_for(N, [=](std::size_t id) { a[id] = 1; });

    // Task B
    auto task_b = Q.parallel_for(N, [=](std::size_t id) { b[id] = 2; });

    // Task C
    auto task_c = Q.parallel_for(N, { task_a, task_b }, [=](std::size_t id) {
      a[id] += b[id];
    });

    // Task D
    int  value  = 3;
    auto task_d = Q.single_task(
      { task_c },
      [=](int *data) -> void {
        using std::get;
        for (int i = 1; i < N; i++) {
          data[0] += data[i];
        }

        data[0] /= value;
      },
      a);

    task_d.wait();
  }

  CS_FREE_HD(a);
  CS_FREE_HD(b);
}
