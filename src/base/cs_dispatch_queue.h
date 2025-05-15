#pragma once

#include "base/cs_base_accel.h"
#include "base/cs_dispatch.h"
#include "base/cs_mem.h"
#include <type_traits>

#ifdef __CUDACC__

#include <cuda.h>
#include <cuda_runtime.h>

#include <initializer_list>
#include <tuple>
#include <utility>

//! Represents an event to synchronize with. Often the end of a cs_device_task.
struct cs_event_t {
  cudaEvent_t cuda_event;
  cs_event_t(cudaEvent_t event) : cuda_event(event) {}
};

//! A cs_task_t object represents a task that can be syncronized to and with.
//! It holds a cs_device_context with a unique CUDA stream and CUDA events can
//! be recorded from the task to synchronize other tasks with it.
//!
//! cs_task_t objects are meant to be spawned from a cs_queue_t object.
class cs_device_task {
  cs_device_context context_;

public:
  //! Creates a new task with a given context and initializes a new stream.
  cs_device_task(cs_device_context context) : context_(std::move(context))
  {
    cudaStream_t new_stream;
    cudaStreamCreate(&new_stream);
    context_.set_cuda_stream(new_stream);
  }

  //! Creates a new task with a cs_device_context initialized with a new stream.
  cs_device_task()
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

//! cs_host_task extends cs_device_task to add support for host function tasks.
template <class FunctionType, class... Args>
class cs_host_task : public cs_device_task {
public:
  //! Tuple type for argument storage.
  using args_tuple_t = std::tuple<Args...>;

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
  cs_host_task(FunctionType &&function, cs_device_context context)
    : cs_device_task(std::move(context)),
      data_tuple_(std::move(function), args_tuple_t{})
  {
  }

  //! Launches the host function asynchronously using the given parameters
  //! with cudaLaunchHostFunc.
  cudaError_t
  launch(Args... args)
  {
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
};

//! SYCL-like execution "queue".
class cs_queue {
public:
  //! Context used to initialize tasks.
  cs_device_context initializer_context;

  template <class F, class... Args>
  cs_device_task
  parallel_for(cs_lnum_t n, F &&f, Args &&...args)
  {
    cs_device_task new_task(initializer_context);
    new_task.get_context().parallel_for(n,
                                        std::forward<F>(f),
                                        std::forward<Args>(args)...);
    return new_task;
  }

  template <class F, class... Args>
  cs_device_task
  parallel_for(cs_lnum_t                                n,
               std::initializer_list<cs_event_t> const &sync_events,
               F                                      &&f,
               Args &&...args)
  {
    cs_device_task new_task(initializer_context);

    new_task.wait_for_range(sync_events);

    new_task.get_context().parallel_for(n,
                                        std::forward<F>(f),
                                        std::forward<Args>(args)...);
    return new_task;
  }

  template <class M, class F, class... Args>
  cs_device_task
  parallel_for_i_faces(const M *m, F &&f, Args &&...args)
  {
    cs_device_task new_task(initializer_context);
    new_task.get_context().parallel_for_i_faces(m,
                                                std::forward<F>(f),
                                                std::forward<Args>(args)...);
    return new_task;
  }

  template <class M, class F, class... Args>
  cs_device_task
  parallel_for_i_faces(const M                                 *m,
                       std::initializer_list<cs_event_t> const &sync_events,
                       F                                      &&f,
                       Args &&...args)
  {
    cs_device_task new_task(initializer_context);

    new_task.wait_for_range(sync_events);

    new_task.get_context().parallel_for_i_faces(m,
                                                std::forward<F>(f),
                                                std::forward<Args>(args)...);
    return new_task;
  }

  template <class M, class F, class... Args>
  cs_device_task
  parallel_for_b_faces(const M *m, F &&f, Args &&...args)
  {
    cs_device_task new_task(initializer_context);
    new_task.get_context().parallel_for_b_faces(m,
                                                std::forward<F>(f),
                                                std::forward<Args>(args)...);
    return new_task;
  }

  template <class M, class F, class... Args>
  cs_device_task
  parallel_for_b_faces(const M                                 *m,
                       std::initializer_list<cs_event_t> const &sync_events,
                       F                                      &&f,
                       Args &&...args)
  {
    cs_device_task new_task(initializer_context);

    new_task.wait_for_range(sync_events);

    new_task.get_context().parallel_for_b_faces(m,
                                                std::forward<F>(f),
                                                std::forward<Args>(args)...);
    return new_task;
  }

  template <class T, class F, class... Args>
  cs_device_task
  parallel_for_reduce_sum(cs_lnum_t n, T &sum, F &&f, Args &&...args)
  {
    cs_device_task new_task(initializer_context);
    parallel_for_reduce_sum(n,
                            sum,
                            std::forward<F>(f),
                            std::forward<Args>(args)...);
    return new_task;
  }

  template <class T, class F, class... Args>
  cs_device_task
  parallel_for_reduce_sum(cs_lnum_t                                n,
                          std::initializer_list<cs_event_t> const &sync_events,
                          T                                       &sum,
                          F                                      &&f,
                          Args &&...args)
  {
    cs_device_task new_task(initializer_context);

    new_task.wait_for_range(sync_events);

    parallel_for_reduce_sum(n,
                            sum,
                            std::forward<F>(f),
                            std::forward<Args>(args)...);
    return new_task;
  }

  template <class T, class R, class F, class... Args>
  cs_device_task
  parallel_for_reduce(cs_lnum_t n, T &r, R &reducer, F &&f, Args &&...args)
  {
    cs_device_task new_task(initializer_context);
    parallel_for_reduce(n,
                        r,
                        reducer,
                        std::forward<F>(f),
                        std::forward<Args>(args)...);
    return new_task;
  }

  template <class FunctionType, class... Args>
  cs_host_task<FunctionType, std::remove_reference_t<Args>...>
  single_task(std::initializer_list<cs_event_t> const &sync_events,
              FunctionType                           &&host_function,
              Args &&...args)
  {
    cs_host_task<FunctionType, std::remove_reference_t<Args>...> single_task(
      std::move(host_function),
      initializer_context);

    single_task.wait_for_range(sync_events);
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
  int value = 3;
  Q.single_task(
    { e3 },
    [](int *data) -> void {
      using std::get;
      for (int i = 1; i < N; i++) {
        data[0] += data[i];
      }

      data[0] /= 3;
    },
    a);

  CS_FREE_HD(a);
  CS_FREE_HD(b);
}

#endif // __CUDACC__
