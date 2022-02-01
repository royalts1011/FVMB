cmake_minimum_required(VERSION 3.20)
project(Files_Exercise_6_20220121)

set(CMAKE_CXX_STANDARD 14)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(Files_Exercise_6_20220121
        fvmbv_exercise6.cpp Net.cpp)
target_link_libraries(Files_Exercise_6_20220121 "${TORCH_LIBRARIES}")
set_property(TARGET Files_Exercise_6_20220121 PROPERTY CXX_STANDARD 14)
