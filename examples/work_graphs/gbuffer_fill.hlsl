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

// This is a root 32-bit value
cbuffer InstanceConstantBuffer : register(b0)
{
    uint g_InstanceID;
};

StructuredBuffer<Instance> t_InstanceData : register(t0);
StructuredBuffer<Material> t_MaterialData : register(t3);
StructuredBuffer<AnimState> t_AnimStateData : register(t4);

struct PSInput
{
    float4 position : SV_Position;
    float3 normal : NORMAL;
    float2 sinCos : NRMROTY;
    uint faceted : FACETED;
    uint material : MATERIAL;
};

PSInput VSMain(float3 vertexPosition : POSITION, float3 vertexNormal : NORMAL)
{
    const Instance instanceData = t_InstanceData[g_InstanceID];
    const AnimState animStateData = t_AnimStateData[g_InstanceID];
    const Material material = t_MaterialData[instanceData.material];

    float3 scale = instanceData.size*animStateData.scale;
    float rotationY = instanceData.rotationY+animStateData.rotationY+(vertexPosition.y+0.5f)*animStateData.twist;
    float3 translation = instanceData.position+float3(0,(scale.y-instanceData.size.y)*0.5f+animStateData.offsetY,0);

    float2 sinCos;
    sincos(rotationY, sinCos.x, sinCos.y);

    // Transform position to world space, individual transform steps
    vertexPosition *= scale; // Scale
    vertexPosition = RotateY(vertexPosition, sinCos);
    vertexPosition += translation; // Translation

    // Transform normal to world space
    if (material.materialType != BT_Faceted)
    {
        vertexNormal = RotateY(vertexNormal, sinCos);
        sinCos = float2(0,0);
    }

    PSInput result;
    result.position = mul(float4(vertexPosition,1), viewProj);
    result.normal = vertexNormal;
    result.sinCos = sinCos;
    result.faceted = (material.materialType == BT_Faceted) ? 1 : 0;
    result.material = instanceData.material;

    return result;
}

uint4 PSMain(PSInput input) : SV_Target
{
    if (input.faceted)
    {
        const int sections=16;
	    const float longitude = atan2(input.normal.z,input.normal.x)/c_PI;
	    const float latitude = acos(input.normal.y)/c_PI;
	    const float facetedU = ((int)(latitude*sections))/(float)sections;
	    const float facetedV = ((int)(longitude*sections))/(float)sections;

	    float sinU,cosU,sinV,cosV;
	    sincos(facetedU*c_PI,sinU,cosU);
	    sincos(facetedV*c_PI,sinV,cosV);
	    float3 facetedNormal = float3(sinU*cosV,cosU,sinU*sinV);

        input.normal = RotateY(facetedNormal, input.sinCos); // To world space
    }

    return uint4((input.normal*0.5f+0.5f)*0xFFFF, input.material);
}
