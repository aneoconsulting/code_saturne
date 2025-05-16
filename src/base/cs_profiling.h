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

#define CS_COMBINE_DETAIL(x, y) x##y
#define CS_COMBINE(x, y) CS_COMBINE_DETAIL(x, y)

#define CS_STRINGIFY_DETAIL(x) #x
#define CS_STRINGIFY(x) CS_STRINGIFY_DETAIL(x)

#if CS_PROFILING == CS_PROFILING_NONE

/// Annotates a whole function.
#define CS_PROFILE_FUNC_RANGE()

/// Annotates a range delimited by the lifetime of a variable.
#define CS_PROFILE_RANGE(range_name)

/// Adds a mark in a profile that corresponds to the current file and line.
#define CS_PROFILE_MARK_LINE()

#elif CS_PROFILING == CS_PROFILING_NVTX

#include <nvtx3/nvtx3.hpp>

/// Annotates a whole function.
#define CS_PROFILE_FUNC_RANGE() NVTX3_FUNC_RANGE()

/// Annotates a range delimited by the lifetime of a variable.
#define CS_PROFILE_RANGE(range_name)                                           \
  nvtx3::scoped_range CS_COMBINE(__cs__profile_range_, __LINE__){              \
    range_name " (" __FILE__ ":" CS_STRINGIFY(__LINE__) ")"                    \
  };

/// Adds a mark in a profile that corresponds to the current file and line.
#define CS_PROFILE_MARK_LINE() nvtx3::mark(__FILE__ ":" CS_STRINGIFY(__LINE__))

#include <sstream>

template <typename T>
std::string
paddr_to_string(const T *ptr)
{
  std::ostringstream oss;
  oss << static_cast<const void *>(ptr);
  return oss.str();
}

#endif
