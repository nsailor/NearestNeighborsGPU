cmake_minimum_required(VERSION 3.0.0)
project(NearestNeighbor VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")

include(CTest)
enable_testing()

find_package(OpenCL REQUIRED)

add_executable(NearestNeighbor main.cpp)
target_link_libraries(NearestNeighbor ${OpenCL_LIBRARIES})
target_include_directories(NearestNeighbor PUBLIC OpenCL::OpenCL)
target_compile_definitions(NearestNeighbor PUBLIC CL_SILENCE_DEPRECATION=1)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
