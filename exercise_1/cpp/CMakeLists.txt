cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(fvmbv_exercise1)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(fvmbv_exercise1 main.cpp)
target_link_libraries(fvmbv_exercise1 "${TORCH_LIBRARIES}")
set_property(TARGET fvmbv_exercise1 PROPERTY CXX_STANDARD 14)
