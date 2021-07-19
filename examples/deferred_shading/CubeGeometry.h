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

#include <donut/core/math/math.h>

static const donut::math::float3 g_Positions[] = {
    {-0.5f,  0.5f, -0.5f}, // front face
    { 0.5f, -0.5f, -0.5f},
    {-0.5f, -0.5f, -0.5f},
    { 0.5f,  0.5f, -0.5f},

    { 0.5f, -0.5f, -0.5f}, // right side face
    { 0.5f,  0.5f,  0.5f},
    { 0.5f, -0.5f,  0.5f},
    { 0.5f,  0.5f, -0.5f},

    {-0.5f,  0.5f,  0.5f}, // left side face
    {-0.5f, -0.5f, -0.5f},
    {-0.5f, -0.5f,  0.5f},
    {-0.5f,  0.5f, -0.5f},

    { 0.5f,  0.5f,  0.5f}, // back face
    {-0.5f, -0.5f,  0.5f},
    { 0.5f, -0.5f,  0.5f},
    {-0.5f,  0.5f,  0.5f},

    {-0.5f,  0.5f, -0.5f}, // top face
    { 0.5f,  0.5f,  0.5f},
    { 0.5f,  0.5f, -0.5f},
    {-0.5f,  0.5f,  0.5f},

    { 0.5f, -0.5f,  0.5f}, // bottom face
    {-0.5f, -0.5f, -0.5f},
    { 0.5f, -0.5f, -0.5f},
    {-0.5f, -0.5f,  0.5f},
};

static const donut::math::float2 g_TexCoords[] = {
    {0.0f, 0.0f}, // front face
    {1.0f, 1.0f},
    {0.0f, 1.0f},
    {1.0f, 0.0f},

    {0.0f, 1.0f}, // right side face
    {1.0f, 0.0f},
    {1.0f, 1.0f},
    {0.0f, 0.0f},

    {0.0f, 0.0f}, // left side face
    {1.0f, 1.0f},
    {0.0f, 1.0f},
    {1.0f, 0.0f},

    {0.0f, 0.0f}, // back face
    {1.0f, 1.0f},
    {0.0f, 1.0f},
    {1.0f, 0.0f},

    {0.0f, 1.0f}, // top face
    {1.0f, 0.0f},
    {1.0f, 1.0f},
    {0.0f, 0.0f},

    {1.0f, 1.0f}, // bottom face
    {0.0f, 0.0f},
    {1.0f, 0.0f},
    {0.0f, 1.0f},
};

static const donut::math::uint g_Normals[] = {
    donut::math::vectorToSnorm8(donut::math::float4(0.0f, 0.0f, -1.0f, 0.0f)), // front face
    donut::math::vectorToSnorm8(donut::math::float4(0.0f, 0.0f, -1.0f, 0.0f)),
    donut::math::vectorToSnorm8(donut::math::float4(0.0f, 0.0f, -1.0f, 0.0f)),
    donut::math::vectorToSnorm8(donut::math::float4(0.0f, 0.0f, -1.0f, 0.0f)),

    donut::math::vectorToSnorm8(donut::math::float4(1.0f, 0.0f, 0.0f, 0.0f)), // right side face
    donut::math::vectorToSnorm8(donut::math::float4(1.0f, 0.0f, 0.0f, 0.0f)),
    donut::math::vectorToSnorm8(donut::math::float4(1.0f, 0.0f, 0.0f, 0.0f)),
    donut::math::vectorToSnorm8(donut::math::float4(1.0f, 0.0f, 0.0f, 0.0f)),

    donut::math::vectorToSnorm8(donut::math::float4(-1.0f, 0.0f, 0.0f, 0.0f)), // left side face
    donut::math::vectorToSnorm8(donut::math::float4(-1.0f, 0.0f, 0.0f, 0.0f)),
    donut::math::vectorToSnorm8(donut::math::float4(-1.0f, 0.0f, 0.0f, 0.0f)),
    donut::math::vectorToSnorm8(donut::math::float4(-1.0f, 0.0f, 0.0f, 0.0f)),

    donut::math::vectorToSnorm8(donut::math::float4(0.0f, 0.0f, 1.0f, 0.0f)), // back face
    donut::math::vectorToSnorm8(donut::math::float4(0.0f, 0.0f, 1.0f, 0.0f)),
    donut::math::vectorToSnorm8(donut::math::float4(0.0f, 0.0f, 1.0f, 0.0f)),
    donut::math::vectorToSnorm8(donut::math::float4(0.0f, 0.0f, 1.0f, 0.0f)),

    donut::math::vectorToSnorm8(donut::math::float4(0.0f, 1.0f, 0.0f, 0.0f)), // top face
    donut::math::vectorToSnorm8(donut::math::float4(0.0f, 1.0f, 0.0f, 0.0f)),
    donut::math::vectorToSnorm8(donut::math::float4(0.0f, 1.0f, 0.0f, 0.0f)),
    donut::math::vectorToSnorm8(donut::math::float4(0.0f, 1.0f, 0.0f, 0.0f)),

    donut::math::vectorToSnorm8(donut::math::float4(0.0f, -1.0f, 0.0f, 0.0f)), // bottom face
    donut::math::vectorToSnorm8(donut::math::float4(0.0f, -1.0f, 0.0f, 0.0f)),
    donut::math::vectorToSnorm8(donut::math::float4(0.0f, -1.0f, 0.0f, 0.0f)),
    donut::math::vectorToSnorm8(donut::math::float4(0.0f, -1.0f, 0.0f, 0.0f)),
};

static const donut::math::uint g_Tangents[] = {
    donut::math::vectorToSnorm8(donut::math::float4(1.0f, 0.0f, 0.0f, 1.0f)), // front face
    donut::math::vectorToSnorm8(donut::math::float4(1.0f, 0.0f, 0.0f, 1.0f)),
    donut::math::vectorToSnorm8(donut::math::float4(1.0f, 0.0f, 0.0f, 1.0f)),
    donut::math::vectorToSnorm8(donut::math::float4(1.0f, 0.0f, 0.0f, 1.0f)),

    donut::math::vectorToSnorm8(donut::math::float4(0.0f, 0.0f, 1.0f, 1.0f)), // right side face
    donut::math::vectorToSnorm8(donut::math::float4(0.0f, 0.0f, 1.0f, 1.0f)),
    donut::math::vectorToSnorm8(donut::math::float4(0.0f, 0.0f, 1.0f, 1.0f)),
    donut::math::vectorToSnorm8(donut::math::float4(0.0f, 0.0f, 1.0f, 1.0f)),

    donut::math::vectorToSnorm8(donut::math::float4(0.0f, 0.0f, -1.0f, 1.0f)), // left side face
    donut::math::vectorToSnorm8(donut::math::float4(0.0f, 0.0f, -1.0f, 1.0f)),
    donut::math::vectorToSnorm8(donut::math::float4(0.0f, 0.0f, -1.0f, 1.0f)),
    donut::math::vectorToSnorm8(donut::math::float4(0.0f, 0.0f, -1.0f, 1.0f)),

    donut::math::vectorToSnorm8(donut::math::float4(-1.0f, 0.0f, 0.0f, 1.0f)), // back face
    donut::math::vectorToSnorm8(donut::math::float4(-1.0f, 0.0f, 0.0f, 1.0f)),
    donut::math::vectorToSnorm8(donut::math::float4(-1.0f, 0.0f, 0.0f, 1.0f)),
    donut::math::vectorToSnorm8(donut::math::float4(-1.0f, 0.0f, 0.0f, 1.0f)),

    donut::math::vectorToSnorm8(donut::math::float4(1.0f, 0.0f, 0.0f, 1.0f)), // top face
    donut::math::vectorToSnorm8(donut::math::float4(1.0f, 0.0f, 0.0f, 1.0f)),
    donut::math::vectorToSnorm8(donut::math::float4(1.0f, 0.0f, 0.0f, 1.0f)),
    donut::math::vectorToSnorm8(donut::math::float4(1.0f, 0.0f, 0.0f, 1.0f)),

    donut::math::vectorToSnorm8(donut::math::float4(1.0f, 0.0f, 0.0f, 1.0f)), // bottom face
    donut::math::vectorToSnorm8(donut::math::float4(1.0f, 0.0f, 0.0f, 1.0f)),
    donut::math::vectorToSnorm8(donut::math::float4(1.0f, 0.0f, 0.0f, 1.0f)),
    donut::math::vectorToSnorm8(donut::math::float4(1.0f, 0.0f, 0.0f, 1.0f)),
};

static const uint32_t g_Indices[] = {
     0,  1,  2,   0,  3,  1, // front face
     4,  5,  6,   4,  7,  5, // left face
     8,  9, 10,   8, 11,  9, // right face
    12, 13, 14,  12, 15, 13, // back face
    16, 17, 18,  16, 19, 17, // top face
    20, 21, 22,  20, 23, 21, // bottom face
};
