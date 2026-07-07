cmake_minimum_required(VERSION 3.15)

project(auto_bcast_bench LANGUAGES CXX)

find_package(MPI REQUIRED)

add_executable(auto_bcast_bench auto_bcast_bench.cpp)
target_compile_features(auto_bcast_bench PRIVATE cxx_std_11)
target_link_libraries(auto_bcast_bench PRIVATE MPI::MPI_CXX)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|Intel")
    target_compile_options(auto_bcast_bench PRIVATE -Wall -Wextra -Wpedantic)
endif()
