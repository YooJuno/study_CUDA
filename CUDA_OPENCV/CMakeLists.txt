cmake_minimum_required(VERSION 3.18)
project(cuda_opencv LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

add_executable(main
    src/main.cpp
    src/kernel.cu
    src/DS_timer.cpp
)

target_include_directories(main
    PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(main
    PRIVATE
        ${OpenCV_LIBS}
        Threads::Threads
        cuda
        cudart
)

set_target_properties(main PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
