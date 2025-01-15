# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.22)

project(VKLOD-Sample)
set(SAMPLE_EXECUTABLE VKLOD-Sample)

set(SOURCE_FILES
  src/acceleration_structures.cpp
  src/acceleration_structures.hpp
  src/gltf_view.hpp
  src/lod_streaming_jobs.cpp
  src/lod_streaming_jobs.hpp
  src/lod_streaming_scene.cpp
  src/lod_streaming_scene.hpp
  src/lod_traverser.cpp
  src/lod_traverser.hpp
  src/main.cpp
  src/meshops_array_view.h
  src/debug_range_summary.hpp
  src/renderer_common.cpp
  src/renderer_common.hpp
  src/renderer_raytrace.cpp
  src/renderer_raytrace.hpp
  src/renderer_rasterize.cpp
  src/renderer_rasterize.hpp
  src/sample_allocation.hpp
  src/sample_camera_paths.hpp
  src/sample_camera_paths.cpp
  src/sample_app_element.hpp
  src/sample_glsl_compiler.hpp
  src/sample_image.hpp
  src/sample_producer_consumer.hpp
  src/sample_raytracing_objects.hpp
  src/sample_vulkan_objects.hpp
  src/scene.cpp
  src/scene.hpp
  src/nvhiz_vk.cpp
  src/nvhiz_vk.hpp
  )

file(GLOB GLSL_FILES shaders/*.*)
source_group("Shaders" FILES ${GLSL_FILES})

# Look for nvpro_core 1) as a sub-folder 2) at some other locations
if(NOT BASE_DIRECTORY)
  find_path(BASE_DIRECTORY
    NAMES nvpro_core/cmake/setup.cmake
    PATHS ${CMAKE_CURRENT_SOURCE_DIR}/external ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/../..
    REQUIRED
    DOC "Directory containing nvpro_core"
  )
endif()

# Include nvpro_core makefiles
if(EXISTS ${BASE_DIRECTORY}/nvpro_core/cmake/setup.cmake)
  include(${BASE_DIRECTORY}/nvpro_core/cmake/setup.cmake)
  include(${BASE_DIRECTORY}/nvpro_core/cmake/utilities.cmake)
else()
  message(FATAL_ERROR "could not find base directory, please set BASE_DIRECTORY to folder containing nvpro_core")
endif()

# Download the bunny as the default mesh to load
download_files(FILENAMES bunny_v2.zip EXTRACT)

# Set a default CMAKE_INSTALL_PREFIX for nvpro samples
_add_project_definitions(${SAMPLE_EXECUTABLE})

file(RELATIVE_PATH NVPRO_CORE_DIR ${CMAKE_CURRENT_SOURCE_DIR} ${BASE_DIRECTORY}/nvpro_core/)

# Turn on extra nvpro_core features (the same lib can be referenced with
# multiple configurations). Some _add_package_*() nvpro_core specializations
# alter the implementation, in which case samples are requested to compile extra
# source files rather than compile extra static libraries or an all-in-one
# nvpro_core lib. This is done with COMMON_SOURCE_FILES, PACKAGE_SOURCE_FILES,
# PLATFORM_LIBRARIES, LIBRARIES_DEBUG and LIBRARIES_OPTIMIZED.
_add_package_VulkanSDK()
_add_package_ShaderC()
_add_package_ImGUI()
_add_package_NVML()
_add_nvpro_core_lib()

add_subdirectory(nv_cluster_lod_builder)

add_library(vma_static src/vma_impl.cpp)
target_link_libraries(vma_static PUBLIC vma)  # nvpro_core exports header-only 'vma'
set_property(TARGET vma_static PROPERTY FOLDER "ThirdParty")

add_library(cgltf_static src/cgltf_impl.cpp)
target_link_libraries(cgltf_static PUBLIC cgltf)  # nvpro_core exports header-only 'cgltf'
set_property(TARGET cgltf_static PROPERTY FOLDER "ThirdParty")

add_library(stb_image_static src/stb_image_impl.cpp)
target_link_libraries(stb_image_static PUBLIC stb)  # nvpro_core exports header-only 'stb'
set_property(TARGET stb_image_static PROPERTY FOLDER "ThirdParty")

# nvpro_core has some configuration dependent source files it expects
# applications to compile. These are built separately as warnings in this
# project are too strict.
add_library(nvpro_core_extra
  ${COMMON_SOURCE_FILES}
  ${PACKAGE_SOURCE_FILES})
target_link_libraries(nvpro_core_extra PUBLIC nvpro_core)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(${SAMPLE_EXECUTABLE}
  ${SOURCE_FILES}
  ${GLSL_FILES})

if(MSVC)
    target_compile_options(${SAMPLE_EXECUTABLE} PRIVATE
        /W4
        /WX

        # Enable some specific extras
        /w45038 # warning C5038: data member 'member1' will be initialized after data member 'member2'
        /w45219 # warning C5219: implicit conversion from 'int' to 'float', possible loss of data
        /w44265 # warning C4265: 'NVPWindow': class has virtual functions, but its non-trivial destructor is not virtual; instances of this class may not be destructed correctly
        /w44355 # warning C4355: 'this': used in base member initializer list
        /w45246 # warning C5246: '_Elems': the initialization of a subobject should be wrapped in braces

        # Ignore nvpro_core header warnings from /W4
        /experimental:external /external:W3
        /external:I${BASE_DIRECTORY}/nvpro_core/
        /external:I${BASE_DIRECTORY}/nvpro_core/nvp
        /external:I${BASE_DIRECTORY}/nvpro_core/nvvk
        /external:I${BASE_DIRECTORY}/nvpro_core/nvvkhl
        #/external:templates-  # can't enable due to nvvk::PushComputeDispatcher
    )
    target_compile_definitions(${SAMPLE_EXECUTABLE} PRIVATE WIN32_LEAN_AND_MEAN=1 NOMINMAX)
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${SAMPLE_EXECUTABLE})
else()
    target_compile_options(${SAMPLE_EXECUTABLE} PRIVATE
        -Wall
        -Wextra
        #-Wpedantic
        -Wconversion
        -Werror

        # Strange false positives/bug?
        -Wno-maybe-uninitialized
    )

    set_property(SOURCE src/renderer_rasterize.cpp APPEND_STRING PROPERTY COMPILE_OPTIONS
        -Wno-missing-field-initializers
        -Wno-unused-parameter
    )

    target_compile_definitions(${SAMPLE_EXECUTABLE} PRIVATE
        $<$<CONFIG:Debug>:_GLIBCXX_ASSERTIONS>
        # Can't use. Conflicts with std::vector use in the nvpro_core API
        #$<$<CONFIG:Debug>:_GLIBCXX_DEBUG>           # DANGER: ABI breaking!
        #$<$<CONFIG:Debug>:_GLIBCXX_DEBUG_BACKTRACE> # DANGER: ABI breaking!
    )

    # Ignore warnings from nvpro_core headers
    target_include_directories(${SAMPLE_EXECUTABLE} SYSTEM PRIVATE
      ${NVPRO_CORE_DIR}
      ${NVPRO_CORE_DIR}/third_party/glm)

    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        # for when you write too much code without building
        target_compile_options(${SAMPLE_EXECUTABLE} PRIVATE -fmax-errors=5)

        # Optional: faster repeat gdb launch at cost of post-build step
        if (CMAKE_BUILD_TYPE STREQUAL "Debug")
            add_custom_command(TARGET ${SAMPLE_EXECUTABLE} POST_BUILD
                COMMAND gdb-add-index $<TARGET_FILE:${SAMPLE_EXECUTABLE}>
                COMMENT "gdb-add-index ${SAMPLE_EXECUTABLE}"
            )
        endif()
    endif()
endif()

target_precompile_headers(${SAMPLE_EXECUTABLE} PRIVATE
  <algorithm>
  <any>
  <array>
  <assert.h>
  <bit>
  <cgltf.h>
  <condition_variable>
  <cstddef>
  <cstdint>
  <cstdlib>
  <deque>
  <filesystem>
  <functional>
  <glm/glm.hpp>
  <glm/gtc/type_ptr.hpp>
  <glm/gtx/hash.hpp>
  <imgui.h>
  <imgui/imgui_camera_widget.h>
  <imgui/imgui_helper.h>
  <initializer_list>
  <iostream>
  <memory>
  <memory_resource>
  <meshops_array_view.h>
  <mutex>
  <nvh/alignment.hpp>
  <nvh/commandlineparser.hpp>
  <nvh/nvprint.hpp>
  <nvvk/commands_vk.hpp>
  <nvvk/compute_vk.hpp>
  <nvvk/debug_util_vk.hpp>
  <nvvk/dynamicrendering_vk.hpp>
  <nvvkhl/alloc_vma.hpp>
  <nvvkhl/application.hpp>
  <nvvkhl/element_camera.hpp>
  <nvvkhl/element_gui.hpp>
  <nvvkhl/gbuffer.hpp>
  <nvvkhl/glsl_compiler.hpp>
  <nvvkhl/pipeline_container.hpp>
  <nvvk/images_vk.hpp>
  <nvvk/memallocator_vk.hpp>
  <nvvk/memallocator_vma_vk.hpp>
  <nvvk/pipeline_vk.hpp>
  <nvvk/resourceallocator_vk.hpp>
  <nvvk/sbtwrapper_vk.hpp>
  <nvvk/stagingmemorymanager_vk.hpp>
  <optional>
  <ostream>
  <ranges>
  <set>
  <span>
  <stb_image.h>
  <stdexcept>
  <string_view>
  <third_party/imgui/backends/imgui_impl_vulkan.h>
  <type_traits>
  <unordered_map>
  <utility>
  <vector>
  <vk_mem_alloc.h>
  )

set(DECODELESS_SEARCH_DEPENDENCIES OFF)
set(DECODELESS_FETCH_DEPENDENCIES ON)
set(BUILD_DECODELESS_TESTING OFF)
include(FetchContent)
FetchContent_Declare(
  decodeless_offset_ptr
  GIT_REPOSITORY https://github.com/decodeless/offset_ptr.git
  GIT_TAG 78693cb545ac155b9514c17caa4be671b791b8e5)
FetchContent_MakeAvailable(decodeless_offset_ptr)
FetchContent_Declare(
  decodeless_writer
  GIT_REPOSITORY https://github.com/decodeless/writer.git
  GIT_TAG e1626714146a8b7f7b9157bc1dbb3f339689e589)
FetchContent_MakeAvailable(decodeless_writer)

target_include_directories(${SAMPLE_EXECUTABLE} PRIVATE src)

target_link_libraries(${SAMPLE_EXECUTABLE} nvpro_core nvpro_core_extra vma_static nv_cluster_lod_builder cgltf_static stb_image_static ${PLATFORM_LIBRARIES} decodeless::writer decodeless::offset_ptr)
target_link_libraries(${SAMPLE_EXECUTABLE} debug ${LIBRARIES_DEBUG})
target_link_libraries(${SAMPLE_EXECUTABLE} optimized ${LIBRARIES_OPTIMIZED})

target_compile_definitions(${SAMPLE_EXECUTABLE} PRIVATE
  NVPRO_CORE_DIR="${NVPRO_CORE_DIR}"

  # Compile error from nvpro_core/nvp/perproject_globals.cpp without this,
  # if _add_project_definitions() is commented out
  PROJECT_NAME="${SAMPLE_EXECUTABLE}"

  GLM_FORCE_CTOR_INIT  # safety
  )

# Defines executable install() targets
_finalize_target(${SAMPLE_EXECUTABLE})

# Install shaders, including some shared headers from nvpro_core. Unlike some
# samples, these are loaded at runtime and not embedded in the binary
file(GLOB NVPRO_CORE_GLSL_FILES ${BASE_DIRECTORY}/nvpro_core/nvvkhl/shaders/*)
install(FILES ${GLSL_FILES} CONFIGURATIONS Release DESTINATION "bin_${ARCH}/GLSL_${SAMPLE_EXECUTABLE}")
install(FILES ${GLSL_FILES} CONFIGURATIONS Debug DESTINATION "bin_${ARCH}_debug/GLSL_${SAMPLE_EXECUTABLE}")
install(FILES ${NVPRO_CORE_GLSL_FILES} CONFIGURATIONS Release DESTINATION "bin_${ARCH}/GLSL_${SAMPLE_EXECUTABLE}/nvvkhl/shaders")
install(FILES ${NVPRO_CORE_GLSL_FILES} CONFIGURATIONS Debug DESTINATION "bin_${ARCH}_debug/GLSL_${SAMPLE_EXECUTABLE}/nvvkhl/shaders")

# Debug in the source directory to avoid having a set of relative shader include
# paths just for debugging with visual studio
set_property(TARGET ${SAMPLE_EXECUTABLE} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
