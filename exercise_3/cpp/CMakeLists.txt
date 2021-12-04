cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(exercise3)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(ex3 fvmbv_exercise3.cpp)

target_link_libraries(ex3 "${TORCH_LIBRARIES}")
set_property(TARGET ex3 PROPERTY CXX_STANDARD 14)
