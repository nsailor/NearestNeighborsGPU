cmake_minimum_required(VERSION 3.0.0)
project(NearestNeighbor VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 11)

include(CTest)
enable_testing()

find_package(OpenCL REQUIRED)

add_executable(NearestNeighbor main.cpp)
target_link_libraries(NearestNeighbor OpenCL::OpenCL)
target_compile_definitions(NearestNeighbor PUBLIC CL_SILENCE_DEPRECATION=1)
target_compile_options(NearestNeighbor PRIVATE "-Wall -Wextra -Werror")

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
