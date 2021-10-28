cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(gauss_smoothing_sparse)

set(CMAKE_PREFIX_PATH "/Users/falcolentzsch/Develope/CPP_Libs/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


add_executable(gauss_smoothing_sparse gauss_smoothing_sparse.cpp)
target_link_libraries(gauss_smoothing_sparse "${TORCH_LIBRARIES}")
set_property(TARGET gauss_smoothing_sparse PROPERTY CXX_STANDARD 14)


