cmake_minimum_required(VERSION 3.22.1 FATAL_ERROR)
project(LegateWrapper)

set(LegateWrapperVersion 0.0.1)
message(STATUS "Project version: v${LegateWrapperVersion}")
set(LIBRARY_NAME legate_jl_wrapper)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

if (NOT DEFINED LEGATE_PREFIX)
  message(FATAL_ERROR "You must set -DLEGATE_PREFIX to your Legate install root")
endif()

# Paths to Legate things
find_library(LEGATE_LIBRARY
  NAMES legate
  PATHS "${LEGATE_PREFIX}/lib"
  NO_DEFAULT_PATH
)
find_path(LEGATE_INCLUDE_DIR
  NAMES legate.h                        # pick one of the top-level headers here
  PATHS "${LEGATE_PREFIX}/include/legate"
  NO_DEFAULT_PATH
)

if (NOT LEGATE_LIBRARY)
  message(FATAL_ERROR "Could not find liblegate.so under ${LEGATE_PREFIX}/lib")
endif()
if (NOT LEGATE_INCLUDE_DIR)
  message(FATAL_ERROR "Could not find Legate headers under ${LEGATE_PREFIX}/include")
endif()

set(LEGATE_DEPS_INCLUDE_DIR "${LEGATE_PREFIX}/include/legate/deps")
if (NOT IS_DIRECTORY "${LEGATE_DEPS_INCLUDE_DIR}")
  message(FATAL_ERROR
    "Could not find Legate deps headers under ${LEGATE_DEPS_INCLUDE_DIR}")
endif()

add_library(legate::legate SHARED IMPORTED GLOBAL)
set_target_properties(legate::legate PROPERTIES
  IMPORTED_LOCATION             "${LEGATE_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${LEGATE_INCLUDE_DIR};${LEGATE_DEPS_INCLUDE_DIR}"
)

# CxxWrap Stuff
find_package(JlCxx)
get_target_property(JlCxx_location JlCxx::cxxwrap_julia LOCATION)
get_filename_component(JlCxx_location ${JlCxx_location} DIRECTORY)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib;${JlCxx_location}")
message(STATUS "Found JlCxx at ${JlCxx_location}")

set(SOURCES
    src/types.cpp
    src/wrapper.cpp
)

add_library(${LIBRARY_NAME} SHARED ${SOURCES})
set_target_properties(${LIBRARY_NAME} PROPERTIES VERSION ${LegateWrapperVersion})

target_link_libraries(${LIBRARY_NAME} PRIVATE legate::legate JlCxx::cxxwrap_julia JlCxx::cxxwrap_julia_stl)
target_include_directories(${LIBRARY_NAME} PRIVATE include)

install(TARGETS ${LIBRARY_NAME} DESTINATION lib)
