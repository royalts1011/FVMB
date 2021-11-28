cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(exercise2)

set(CMAKE_PREFIX_PATH "/Users/falcolentzsch/Develope/CPP_Libs/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(exercise2_task12 fvmbv_exercise2_task12.cpp)
add_executable(exercise2_task3 fvmbv_exercise2_task3.cpp)

target_link_libraries(exercise2_task12 "${TORCH_LIBRARIES}")
set_property(TARGET exercise2_task12 PROPERTY CXX_STANDARD 14)
target_link_libraries(exercise2_task3 "${TORCH_LIBRARIES}")
set_property(TARGET exercise2_task3 PROPERTY CXX_STANDARD 14)