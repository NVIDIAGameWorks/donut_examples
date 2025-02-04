# Donut Samples

This repository provides a collection of example applications built using the [Donut framework](https://github.com/NVIDIA-RTX/Donut).

| Application                                               |        DX11        |        DX12        |        VK          | Description |
|-----------------------------------------------------------|:------------------:|:------------------:|:------------------:|-------------|
| [Feature Demo](feature_demo)                              | :white_check_mark: | :white_check_mark: | :white_check_mark: | A demo application that shows most of the raster-based features and effects available. |
| [Basic Triangle](examples/basic_triangle)                 | :white_check_mark: | :white_check_mark: | :white_check_mark: | The most basic example that draws a single triangle. |
| [Bindless Ray Tracing](examples/rt_bindless)              |                    | :white_check_mark: | :white_check_mark: | Renders a scene using ray tracing, starting from primary rays, and using bindless resources. Includes skeletal animation. |
| [Bindless Rendering](examples/bindless_rendering)         |                    | :white_check_mark: | :white_check_mark: | Renders a scene using bindless resources for minimal CPU overhead. |
| [Deferred Shading](examples/deferred_shading)             | :white_check_mark: | :white_check_mark: | :white_check_mark: | Draws a textured cube into a G-buffer and applies deferred shading to it. |
| [Headless Device](examples/headless)                      | :white_check_mark: | :white_check_mark: | :white_check_mark: | Tests operation of a graphics device without a window by adding some numbers. |
| [Meshlets](examples/meshlets)                             |                    | :white_check_mark: | :white_check_mark: | Renders a triangle using meshlets. |
| [Ray Traced Particles](examples/rt_particles)             |                    | :white_check_mark: | :white_check_mark: | Renders a particle system using ray tracing in an environment with mirrors. |
| [Ray Traced Reflections](examples/rt_reflections)         |                    | :white_check_mark: |                    | Rasterizes the G-buffer and renders basic ray traced reflections. Materials are accessed using local root signatures. |
| [Ray Traced Shadows](examples/rt_shadows)                 |                    | :white_check_mark: | :white_check_mark: | Rasterizes the G-buffer and renders basic ray traced directional shadows. |
| [Ray Traced Triangle](examples/rt_triangle)               |                    | :white_check_mark: | :white_check_mark: | Renders a triangle using ray tracing. |
| [Shader Specializations](examples/shader_specializations) |                    |                    | :white_check_mark: | Renders a few triangles using different specializations of the same shader. |
| [Threaded Rendering](examples/threaded_rendering)         |                    | :white_check_mark: | :white_check_mark: | Renders a cube map view of a scene using multiple threads, one per face. |
| [Variable Shading](examples/variable_shading)             |                    | :white_check_mark: | :white_check_mark: | Renders a scene with variable shading rate specified by a texture. |
| [Vertex Buffer](examples/vertex_buffer)                   | :white_check_mark: | :white_check_mark: | :white_check_mark: | Creates a vertex buffer for a cube and draws the cube. |
| [Work Graphs](examples/work_graphs)                       |                    | :white_check_mark: |                    | Demonstrates the new D3D12 work graphs API via a tiled deferred shading renderer that dynamically chooses shaders for each screen tile. Requires DXC with shader model 6.8 support. |

## Requirements

Same as the [requirements for Donut](https://github.com/NVIDIA-RTX/Donut).

## Build

1. Clone the repository **with all submodules**:
   
   `git clone --recursive <URL>`
   
2. Create a build folder.

   `cd donut_examples && mkdir build && cd build`

   * Any name works, but git is configured to ignore folders named 'build\*'
   * This folder should be placed under directory created by 'git clone' in step #1

3. Use CMake to configure the build and generate the project files.
   
   * Linux: `cmake ..`
   * Windows: use of CMake GUI is recommended. Make sure to select the x64 platform for the generator.

4. Build the solution generated by CMake in the build folder.

   * Linux: `make -j8` (example for an 8-core CPU)
   * Windows: Open the generated solution with Visual Studio and build it.

5. Run the examples. They should be built in the `bin` folder.

## Command Line

Most examples support multiple graphics APIs (on Windows). They are built with all APIs supported in the same executable,
and the API can be switched from the command line. Use these command line arguments:

- `-dx11` for D3D11
- `-dx12` for D3D12 (default)
- `-vk` for Vulkan

The Feature Demo supports additional command line arguments:

- `-debug` to enable the graphics API debug layer or runtime, and the [NVRHI](https://github.com/NVIDIA-RTX/NVRHI) validation layer.
- `-fullscreen` to start in full screen mode.
- `-no-vsync` to start without VSync (can be toggled in the GUI).
- `-print-graph` to print the scene graph into the output log on startup.
- `-width` and `-height` to set the window size.
- `<FileName>` to load any supported model or scene from the given file.


## License

Donut Examples are licensed under the [MIT License](LICENSE.txt).