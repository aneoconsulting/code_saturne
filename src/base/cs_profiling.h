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

#if CS_PROFILING == CS_PROFILING_NONE

#define CS_PROFILE_FUNC_RANGE()
#define CS_PROFILE_RANGE(range_name)
#define CS_PROFILE_MARK_LINE()

#elif CS_PROFILING == CS_PROFILING_NVTX

#include <nvtx3/nvtx3.hpp>
#include <string>

#define CS_PROFILE_FUNC_RANGE NVTX3_FUNC_RANGE

#define STRINGIFY_DETAIL(x) #x
#define STRINGIFY(x) STRINGIFY_DETAIL(x)

#define CS_PROFILE_MARK_LINE()                                                 \
  nvtx3::mark((std::string(__FUNCTION__) + ":" + STRINGIFY(__LINE__)).c_str())

#endif
