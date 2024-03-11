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
#include "materials.hlsli"
#include "lighting.hlsli"

// These are root 32-bit values
cbuffer InlineConstants : register(b0)
{
    uint g_LightTilesX, g_LightTilesY;
    uint g_LightCount;
};

StructuredBuffer<Material> t_MaterialData : register(t0);
Texture2D<uint4> t_GBuffer : register(t1);
Texture2D<float> t_DepthBuffer : register(t2);
StructuredBuffer<uint> t_CulledLightsData : register(t3);
StructuredBuffer<Light> t_LightData : register(t4);
RWTexture2D<float4> u_LDRBuffer : register(u1);

[numthreads(8, 4, 1)]
void CSMain(uint2 dispatchThreadId : SV_DispatchThreadID, uint2 groupId : SV_GroupID)
{
    const uint2 pixelXY = dispatchThreadId;
    const float depth = t_DepthBuffer.Load(uint3(pixelXY,0));
    uint lightReadSlotIndex = (groupId.y*g_LightTilesX + groupId.x) * c_MaxLightsPerTile;

    if (depth == 1.0f)
    {
        u_LDRBuffer[pixelXY] = float4(EvaluateSky(pixelXY),1); // Sky
        return;
    }

    const uint4 gbufferData = t_GBuffer.Load(uint3(pixelXY,0));
    const float3 worldPosition = Unproject(pixelXY, depth);
    const float3 worldNormal = normalize(((gbufferData.xyz/(float)0xFFFF)-0.5f)*2.0f); // Decode normal from g-buffer
    const float3 camPosition = camPosAndSceneTime.xyz;
    const uint materialID = gbufferData.w;
    const Material material = t_MaterialData[materialID];

    static const bool useCulledLights = true;

    float3 color = float3(0,0,0);
    uint lightCount = useCulledLights ? c_MaxLightsPerTile : g_LightCount;

    for (uint i=0;i<lightCount;i++)
    {
        const uint lightIndex = useCulledLights ? t_CulledLightsData[lightReadSlotIndex+i] : i;
        if (lightIndex == 0xFFFFFFFF)
            break;

        const Light light = t_LightData[lightIndex];
        if (!PointInSpotLight(pixelXY, depth, light))
            continue;

        float3 lightToPointDir,lightColor;
        float lightAttenuation;
        EvaluateSpotLight(light, worldPosition, lightToPointDir, lightColor, lightAttenuation);

        switch (material.materialType)
        {
        case BT_Lambert:
            color += EvaluateMaterial_Lambert(material, worldPosition, worldNormal, lightToPointDir, lightColor, lightAttenuation);
            break;

        case BT_Phong:
            color += EvaluateMaterial_Phong(material, worldPosition, worldNormal, lightToPointDir, lightColor, lightAttenuation, camPosition);
            break;

        case BT_Metallic:
            color += EvaluateMaterial_Metallic(material, worldPosition, worldNormal, lightToPointDir, lightColor, lightAttenuation, camPosition);
            break;

        case BT_Velvet:
            color += EvaluateMaterial_Velvet(material, worldPosition, worldNormal, lightToPointDir, lightColor, lightAttenuation, camPosition);
            break;

        case BT_Flakes:
            color += EvaluateMaterial_Flakes(material, worldPosition, worldNormal, lightToPointDir, lightColor, lightAttenuation, camPosition);
            break;

        case BT_Stan:
            color += EvaluateMaterial_Stan(material, worldPosition, worldNormal, lightToPointDir, lightColor, lightAttenuation);
            break;

        case BT_Faceted:
            color += EvaluateMaterial_Faceted(material, worldPosition, worldNormal, lightToPointDir, lightColor, lightAttenuation, camPosition);
            break;

        case BT_Checker:
            color += EvaluateMaterial_Checker(material, worldPosition, worldNormal, lightToPointDir, lightColor, lightAttenuation, camPosition);
            break;
        }
    }

    u_LDRBuffer[pixelXY] = float4(color,1);
}
