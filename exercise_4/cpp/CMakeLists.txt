cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(ex4)

set(CMAKE_PREFIX_PATH "/Users/falcolentzsch/Develope/CPP_Libs/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


add_executable(ex4 ../fvmbv_exercise4.cpp)

target_link_libraries(ex4 "${TORCH_LIBRARIES}")
set_property(TARGET ex4 PROPERTY CXX_STANDARD 14)
