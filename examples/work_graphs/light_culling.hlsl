/*
* Copyright (c) 2014-2024, NVIDIA CORPORATION. All rights reserved.
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

#include "scene_data.hlsli"
#include "lighting.hlsli"

// These are root 32-bit values
cbuffer InlineConstants : register(b0)
{
    uint g_LightTilesX, g_LightTilesY;
    uint g_LightCount;
};

Texture2D<float> t_DepthBuffer : register(t1);
StructuredBuffer<Light> t_LightData : register(t4);
RWStructuredBuffer<uint> u_CulledLightsDataRW : register(u0);

groupshared uint s_LightIsRelevant;
groupshared uint s_LightsAdded;

[numthreads(8, 4, 1)]
void CSMain(uint2 dispatchThreadId : SV_DispatchThreadID, uint2 groupThreadId : SV_GroupThreadID, uint2 groupId : SV_GroupID)
{
    const bool isThread0 = (groupThreadId.x == 0) && (groupThreadId.y == 0);
    const uint2 pixelXY = dispatchThreadId;
    const float depth = t_DepthBuffer.Load(uint3(pixelXY,0));
    const uint writeSlotStart = (groupId.y*g_LightTilesX + groupId.x) * c_MaxLightsPerTile;

    if (isThread0)
        s_LightsAdded = 0;

    for (uint i=0;i<g_LightCount;i++)
    {
        GroupMemoryBarrierWithGroupSync();
        if (s_LightsAdded == c_MaxLightsPerTile)
            break;

        const uint threadLightIsRelevant = PointInSpotLight(pixelXY, depth, t_LightData[i]) ? 1 : 0;
        const uint waveLightIsRelevant = WaveActiveBitOr(threadLightIsRelevant);

        if (isThread0)
            s_LightIsRelevant = 0; // Reset flag for this light
        GroupMemoryBarrierWithGroupSync();

        if (WaveIsFirstLane())
        {
            uint oldVal;
            InterlockedOr(s_LightIsRelevant, waveLightIsRelevant, oldVal);
        }

        GroupMemoryBarrierWithGroupSync();

        if (isThread0 && (s_LightIsRelevant != 0))
            u_CulledLightsDataRW[writeSlotStart + (s_LightsAdded++)] = i;
    }

    GroupMemoryBarrierWithGroupSync();
    if (isThread0 && s_LightsAdded<c_MaxLightsPerTile)
        u_CulledLightsDataRW[writeSlotStart + s_LightsAdded] = 0xFFFFFFFF; // Mark the last slot used for this tile
}