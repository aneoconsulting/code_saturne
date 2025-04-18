#pragma once

// Numeric values for different profiling libraries.

/// No profiling enabled.
#define CS_PROFILING_NONE 0

/// Enable NVTX profiling.
#define CS_PROFILING_NVTX 1

// No profiling library is used by default.
#ifndef CS_PROFILING
#define CS_PROFILING CS_PROFILING_NONE
#endif

#define CS_STRINGIFY_DETAIL(x) #x
#define CS_STRINGIFY(x) CS_STRINGIFY_DETAIL(x)

#define CS_COMBINE_DETAIL(x, y) x##y
#define CS_COMBINE(x, y) CS_COMBINE_DETAIL(x, y)

#if CS_PROFILING == CS_PROFILING_NONE

/// Annotates a whole function.
#define CS_PROFILE_FUNC_RANGE()

/// Annotates a range delimited by the lifetime of a variable.
#define CS_PROFILE_RANGE(range_name)

/// Adds a mark in a profile that corresponds to the current function and line.
#define CS_PROFILE_MARK_LINE()

#elif CS_PROFILING == CS_PROFILING_NVTX

#include <nvtx3/nvtx3.hpp>
#include <string>

/// Annotates a whole function.
#define CS_PROFILE_FUNC_RANGE NVTX3_FUNC_RANGE

/// Annotates a range delimited by the lifetime of a variable.
#define CS_PROFILE_RANGE(range_name)                                           \
  nvtx3::scoped_range CS_COMBINE(__cs_range_, __LINE__){ range_name };

/// Adds a mark in a profile that corresponds to the current function and line.
#define CS_PROFILE_MARK_LINE()                                                 \
  nvtx3::mark(                                                                 \
    (std::string(__FUNCTION__) + ":" + CS_STRINGIFY(__LINE__)).c_str())

#endif
