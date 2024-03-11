#
# Copyright (c) 2014-2024, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

file(GLOB shaders "*.hlsl")
file(GLOB sources "*.cpp" "*.h")

set(project work_graphs_d3d12)
set(folder "Examples/Work Graphs")

# Copy agility SDK bits to D3D12 folder right next to the executable
set (D3D12_AGILITY_SDK_RUNTIME_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/D3D12")

if (NOT EXISTS ${D3D12_AGILITY_SDK_RUNTIME_DIR})
    execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${D3D12_AGILITY_SDK_RUNTIME_DIR})
endif()

if (NOT EXISTS "${D3D12_AGILITY_SDK_RUNTIME_DIR}/D3D12Core.dll" OR NOT EXISTS "${D3D12_AGILITY_SDK_RUNTIME_DIR}/D3D12SDKLayers.dll")
    message(STATUS "Extracting Agility SDK runtime bits")
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf "${CMAKE_CURRENT_SOURCE_DIR}/AgilitySDK/d3d12_1.613.0.zip"
        WORKING_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/D3D12")
endif()

# Custom shader compilation using the agility SDK compiler rather than whatever is chosen during CMake config
set_source_files_properties(SHADER_SOURCES FOLDER ${folder} PROPERTIES VS_TOOL_OVERRIDE "None") 
add_custom_target(${project}_shaders DEPENDS ShaderMake SOURCES ${SHADER_SOURCES})

set(dxcCompilerCommand ShaderMake
    --config "${CMAKE_CURRENT_SOURCE_DIR}/shaders.cfg"
    --out "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/shaders/${project}/dxil"
    --platform DXIL
    --binaryBlob --outputExt .bin
    -I ${DONUT_SHADER_INCLUDE_DIR}
    --compiler "${DXC_PATH}"
    --shaderModel 6_8)

add_custom_command(TARGET ${project}_shaders PRE_BUILD COMMAND ${dxcCompilerCommand})
set_target_properties(${project}_shaders PROPERTIES FOLDER ${folder})


add_executable(${project} WIN32 ${sources})
target_include_directories(${project} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/AgilitySDK/include")
target_link_libraries(${project} donut_app donut_engine)
add_dependencies(${project} ${project}_shaders)
set_target_properties(${project} PROPERTIES FOLDER ${folder})

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3 /MP")
endif()