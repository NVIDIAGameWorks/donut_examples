/*
* Copyright (c) 2014-2022, NVIDIA CORPORATION. All rights reserved.
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

#ifndef PARTICLES_CB_H
#define PARTICLES_CB_H

#include <donut/shaders/view_cb.h>

struct GlobalConstants
{
    PlanarViewConstants view;

    float primaryRayConeAngle;
    uint reorientParticlesInPrimaryRays;
    uint reorientParticlesInSecondaryRays;
    uint orientationMode;

    int environmentMapTextureIndex;
};

struct ParticleInfo
{
    float3 center;
    float rotation;

    float3 xAxis;
    float inverseRadius;

    float3 yAxis;
    int textureIndex;

    float3 colorFactor;
    float opacityFactor;
};

#define INSTANCE_MASK_OPAQUE                1
#define INSTANCE_MASK_PARTICLE_GEOMETRY     2
#define INSTANCE_MASK_INTERSECTION_PARTICLE 4

#define ORIENTATION_MODE_AVT_MATRIX         0
#define ORIENTATION_MODE_QUATERNION         1
#define ORIENTATION_MODE_BEAM               2
#define ORIENTATION_MODE_BASIS              3

#endif // PARTICLES_CB_H
