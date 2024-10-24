cmake_minimum_required(VERSION 3.24)
project(MS4 LANGUAGES CXX CUDA)

# Set the C++ and CUDA standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 11)

# Set the GPU architecture
set(CMAKE_CUDA_ARCHITECTURES 75)

# Find the CUDA package
find_package(CUDA REQUIRED)

# Find OpenMP package for the OpenMP version
find_package(OpenMP REQUIRED)

# Include common headers in all targets
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Define the source files used in multiple targets
set(SRC_FILES
        src/MeanShift.cpp
        src/Point.cpp
        src/Utils.cpp
        )

# Add executable for the CUDA version
add_executable(main_parallel_cuda src/main_parallel_cuda.cu ${SRC_FILES})

# Ensure CUDA definitions are set and libraries are linked
target_compile_definitions(main_parallel_cuda PRIVATE USE_CUDA)
target_link_libraries(main_parallel_cuda PRIVATE ${CUDA_LIBRARIES} cudart_static)

# Fix for default library conflict (LIBCMT) with Visual Studio
target_link_options(main_parallel_cuda PRIVATE "/NODEFAULTLIB:LIBCMT")

# Add executable for the OpenMP version (only if OpenMP is found)
if(OpenMP_CXX_FOUND)
    add_executable(main_parallel_openMP src/main_parallel_openMP.cpp ${SRC_FILES})
    target_link_libraries(main_parallel_openMP OpenMP::OpenMP_CXX)
endif()

# Add executable for the sequential version
add_executable(main_sequential src/main_sequential.cpp ${SRC_FILES})

# Add executable for the reports version
add_executable(main_reports src/main_reports.cpp ${SRC_FILES})
