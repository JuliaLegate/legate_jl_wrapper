cmake_minimum_required(VERSION 3.22.1 FATAL_ERROR)
project(LegateWrapper)

set(LegateWrapperVersion 0.0.1)
message(STATUS "Project version: v${LegateWrapperVersion}")
set(LIBRARY_NAME legate_jl_wrapper)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

find_package(legate REQUIRED)

# CxxWrap Stuff
find_package(JlCxx)
get_target_property(JlCxx_location JlCxx::cxxwrap_julia LOCATION)
get_filename_component(JlCxx_location ${JlCxx_location} DIRECTORY)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib;${JlCxx_location}")
message(STATUS "Found JlCxx at ${JlCxx_location}")

set(SOURCES
    src/types.cpp
    src/warpper.cpp
)

set_property(TARGET ${LIBRARY_NAME} PROPERTY VERSION ${LegateWrapperVersion})

add_library(${LIBRARY_NAME} SHARED ${SOURCES})

target_link_libraries(${LIBRARY_NAME} PRIVATE legate::legate JlCxx::cxxwrap_julia JlCxx::cxxwrap_julia_stl)
target_include_directories(${LIBRARY_NAME} PRIVATE include)

install(TARGETS ${LIBRARY_NAME} DESTINATION lib)
