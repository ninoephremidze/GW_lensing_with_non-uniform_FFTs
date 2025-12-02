cmake_minimum_required(VERSION 3.16)
project(cfastfht C)

set(CMAKE_C_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(cfastfht SHARED src/cfastfht.c)

target_include_directories(cfastfht
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include)

find_library(FINUFFT_LIB
    NAMES finufft
    HINTS
        $ENV{FINUFFT_LIB_DIR}
        $ENV{FINUFFT_DIR}
        /home/ubuntu/.local/lib/python3.12/site-packages/finufft
        /usr/local/lib
        /usr/lib)

if(NOT FINUFFT_LIB)
    message(FATAL_ERROR "libfinufft.so not found. Set FINUFFT_LIB_DIR to its directory.")
endif()

get_filename_component(FINUFFT_LIB_DIR ${FINUFFT_LIB} DIRECTORY)

target_link_libraries(cfastfht PRIVATE ${FINUFFT_LIB} m)
set_target_properties(cfastfht PROPERTIES
    BUILD_RPATH ${FINUFFT_LIB_DIR})
