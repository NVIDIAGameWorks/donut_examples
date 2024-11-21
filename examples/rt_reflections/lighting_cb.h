/*
* Copyright (c) 2014-2021, NVIDIA CORPORATION. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/

#ifndef LIGHTING_CB_H
#define LIGHTING_CB_H

#include <donut/shaders/light_cb.h>
#include <donut/shaders/view_cb.h>

#define REFLECTIONS_SPACE_GLOBAL                    0
#define REFLECTIONS_BINDING_MATERIAL_SAMPLER        0
#define REFLECTIONS_BINDING_LIGHTING_CONSTANTS      0
#define REFLECTIONS_BINDING_OUTPUT_UAV              0
#define REFLECTIONS_BINDING_SCENE_BVH               0
#define REFLECTIONS_BINDING_GBUFFER_DEPTH_TEXTURE   1
#define REFLECTIONS_BINDING_GBUFFER_0_TEXTURE       2
#define REFLECTIONS_BINDING_GBUFFER_1_TEXTURE       3
#define REFLECTIONS_BINDING_GBUFFER_2_TEXTURE       4
#define REFLECTIONS_BINDING_GBUFFER_3_TEXTURE       5

#define REFLECTIONS_SPACE_LOCAL                     1
#define REFLECTIONS_BINDING_MATERIAL_CONSTANTS      0
#define REFLECTIONS_BINDING_INDEX_BUFFER            0
#define REFLECTIONS_BINDING_TEX_COORD_BUFFER        1
#define REFLECTIONS_BINDING_NORMAL_BUFFER           2
#define REFLECTIONS_BINDING_DIFFUSE_TEXTURE         3
#define REFLECTIONS_BINDING_SPECULAR_TEXTURE        4
#define REFLECTIONS_BINDING_NORMAL_TEXTURE          5
#define REFLECTIONS_BINDING_EMISSIVE_TEXTURE        6
#define REFLECTIONS_BINDING_OCCLUSION_TEXTURE       7
#define REFLECTIONS_BINDING_TRANSMISSION_TEXTURE    8
#define REFLECTIONS_BINDING_OPACITY_TEXTURE         9

struct LightingConstants
{
    float4 ambientColor;

    LightConstants light;
    PlanarViewConstants view;
};

#endif // LIGHTING_CB_H