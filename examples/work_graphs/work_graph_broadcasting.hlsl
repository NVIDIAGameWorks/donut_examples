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
    uint g_LightCount;
};

StructuredBuffer<Material> t_MaterialData : register(t0);
Texture2D<uint4> t_GBuffer : register(t1);
Texture2D<float> t_DepthBuffer : register(t2);
StructuredBuffer<Light> t_LightData : register(t4);
RWTexture2D<float4> u_LDRBuffer : register(u1);

struct SkyNodeRecord
{
    uint2 tileXY;
    uint uniformTile;
};

struct DarkTileNodeRecord
{
    uint2 tileXY;
};

struct DeferredShadingNodeRecord
{
    uint2 tileXY;
    uint culledLights[c_MaxLightsPerTile];
    uint uniformTile;
};

groupshared uint s_IsSky;
groupshared uint s_MaterialTypeMask;
groupshared uint s_LightsAdded;
groupshared uint s_LightIsRelevant;
groupshared uint s_CulledLights[c_MaxLightsPerTile];

[Shader("node")]
[NodeLaunch("broadcasting")]
[NodeIsProgramEntry]
[NodeDispatchGrid(1,1,1)] // This will be overriden during pipeline creation
[numthreads(8, 4, 1)]
void LightCull_Node(
    [MaxRecords(1)] NodeOutput<SkyNodeRecord> Sky_Node,
    [MaxRecords(1)] NodeOutput<DarkTileNodeRecord> DarkTile_Node,
    [MaxRecords(BT_COUNT)][NodeArraySize(BT_COUNT)] NodeOutputArray<DeferredShadingNodeRecord> Material_Nodes,
	uint2 dispatchThreadId : SV_DispatchThreadID,
    uint2 groupThreadId : SV_GroupThreadID,
    uint2 groupId : SV_GroupID)
{
    const bool isThread0 = (groupThreadId.x == 0) && (groupThreadId.y == 0);
    const uint groupThreadCount = 8 * 4;

    const uint2 pixelXY = dispatchThreadId;
    const float depth = t_DepthBuffer.Load(uint3(pixelXY,0));
    const uint4 gbufferData = t_GBuffer.Load(uint3(pixelXY,0));
    const uint materialID = gbufferData.w;
    const Material material = t_MaterialData[materialID];

    const uint threadIsSky = (depth == 1.0f) ? 1 : 0;
    const uint threadMaterialTypeMask = threadIsSky ? 0 : (1U << material.materialType);

    const uint waveIsSky = WaveActiveSum(threadIsSky == 1);
    const uint waveMaterialTypeMask = WaveActiveBitOr(threadMaterialTypeMask);

    if (isThread0)
    {
        s_LightsAdded = 0;
        s_IsSky = 0;
        s_MaterialTypeMask = 0;
    }

    GroupMemoryBarrierWithGroupSync();

    if (WaveIsFirstLane())
    {
        uint oldVal;
        InterlockedAdd(s_IsSky, waveIsSky, oldVal);
        InterlockedOr(s_MaterialTypeMask, waveMaterialTypeMask, oldVal);
    }

    GroupMemoryBarrierWithGroupSync();

    // Now all threads know the situation for the tile
    const uint tileIsSky = s_IsSky;
    const uint tileMaterialTypeMask = s_MaterialTypeMask;

    if (tileIsSky > 0)
    {
        GroupNodeOutputRecords<SkyNodeRecord> skyRecord = Sky_Node.GetGroupNodeOutputRecords(1);
        skyRecord[0].tileXY = groupId;
        skyRecord[0].uniformTile = (tileIsSky == groupThreadCount) ? 1 : 0;
        skyRecord.OutputComplete();
    }

    if (tileIsSky == groupThreadCount)
        return; // Tile is entirely sky. No need to go further

    // Cull lights
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
            s_CulledLights[s_LightsAdded++] = i;
    }

    GroupMemoryBarrierWithGroupSync();
    if (s_LightsAdded == 0)
    {
        GroupNodeOutputRecords<DarkTileNodeRecord> darkTileRecord = DarkTile_Node.GetGroupNodeOutputRecords(1);
        darkTileRecord[0].tileXY = groupId;
        darkTileRecord.OutputComplete();
        return; // Tile is entirely not receiving any lights. No need to go further
    }

    if (isThread0 && (s_LightsAdded < c_MaxLightsPerTile))
        s_CulledLights[s_LightsAdded] = 0xFFFFFFFF; // Mark the last slot used for this tile

    GroupMemoryBarrierWithGroupSync();

    const uint threadSpawnMaterialType = groupThreadId.x + groupThreadId.y*8;
    const uint threadSpawnMaterialTypeMask = (threadSpawnMaterialType < BT_COUNT) ? (1U << threadSpawnMaterialType) : 0;
    const uint threadWillOutputMaterial = (tileMaterialTypeMask & threadSpawnMaterialTypeMask) ? 1 : 0;
    const uint threadMaterialNodeIndex = min(threadSpawnMaterialType,BT_COUNT-1);

    ThreadNodeOutputRecords<DeferredShadingNodeRecord> deferredShadingRecord = Material_Nodes[threadMaterialNodeIndex].GetThreadNodeOutputRecords(threadWillOutputMaterial);
    if (threadWillOutputMaterial)
    {
        deferredShadingRecord.Get().tileXY = groupId;
        deferredShadingRecord.Get().culledLights = s_CulledLights;
        deferredShadingRecord.Get().uniformTile = (tileMaterialTypeMask == threadSpawnMaterialTypeMask) ? 1 : 0;

    }
    deferredShadingRecord.OutputComplete();
}

[Shader("node")]
[NodeLaunch("broadcasting")]
[NodeDispatchGrid(1,1,1)]
[numthreads(8, 4, 1)]
void Sky_Node(
	DispatchNodeInputRecord<SkyNodeRecord> inputData,
	uint2 groupThreadId : SV_GroupThreadID)
{
    const uint2 pixelXY = inputData.Get().tileXY * uint2(8,4) + groupThreadId;
    const float depth = inputData.Get().uniformTile ? 1.0f : t_DepthBuffer.Load(uint3(pixelXY,0));
    if (depth == 1.0f)
        u_LDRBuffer[pixelXY] = float4(EvaluateSky(pixelXY),1); // Sky
}

[Shader("node")]
[NodeLaunch("broadcasting")]
[NodeDispatchGrid(1,1,1)]
[numthreads(8, 4, 1)]
void DarkTile_Node(
	DispatchNodeInputRecord<DarkTileNodeRecord> inputData,
	uint2 groupThreadId : SV_GroupThreadID)
{
    const uint2 pixelXY = inputData.Get().tileXY * uint2(8,4) + groupThreadId;
    u_LDRBuffer[pixelXY] = float4(0,0,0,1);
}

void EvaluateMaterialColor(const DeferredShadingNodeRecord inputRecord, const uint2 groupThreadId, const uint materialType)
{
    const uint2 pixelXY = inputRecord.tileXY * uint2(8,4) + groupThreadId;
    const uint4 gbufferData = t_GBuffer.Load(uint3(pixelXY,0));
    const uint materialID = gbufferData.w;
    const Material material = t_MaterialData[materialID];

    const uint materialMatches = inputRecord.uniformTile ? 1 : (material.materialType == materialType);
    if (!materialMatches)
        return;

    const float depth = t_DepthBuffer.Load(uint3(pixelXY,0));
    if (depth == 1.0f)
        return;

    const float3 worldPosition = Unproject(pixelXY, depth);
    const float3 worldNormal = normalize(((gbufferData.xyz/(float)0xFFFF)-0.5f)*2.0f); // Decode normal from g-buffer
    const float3 camPosition = camPosAndSceneTime.xyz;

    float3 color = float3(0,0,0);
    for (uint i=0; i<c_MaxLightsPerTile; i++)
    {
        if (inputRecord.culledLights[i] == 0xFFFFFFFF)
            break;

        const Light light = t_LightData[inputRecord.culledLights[i]];
        if (!PointInSpotLight(pixelXY, depth, light))
            continue;

        float3 lightToPointDir,lightColor;
        float lightAttenuation;
        EvaluateSpotLight(light, worldPosition, lightToPointDir, lightColor, lightAttenuation);

        switch (materialType)
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

[Shader("node")]
[NodeID("Material_Nodes",BT_Lambert)]
[NodeLaunch("broadcasting")]
[NodeDispatchGrid(1,1,1)]
[numthreads(8, 4, 1)]
void Material_Lambert_Node(
	DispatchNodeInputRecord<DeferredShadingNodeRecord> inputData,
	uint2 groupThreadId : SV_GroupThreadID)
{
    EvaluateMaterialColor(inputData.Get(), groupThreadId, BT_Lambert);
}

[Shader("node")]
[NodeID("Material_Nodes",BT_Phong)]
[NodeLaunch("broadcasting")]
[NodeDispatchGrid(1,1,1)]
[numthreads(8, 4, 1)]
void Material_Phong_Node(
	DispatchNodeInputRecord<DeferredShadingNodeRecord> inputData,
	uint2 groupThreadId : SV_GroupThreadID)
{
    EvaluateMaterialColor(inputData.Get(), groupThreadId, BT_Phong);
}

[Shader("node")]
[NodeID("Material_Nodes",BT_Metallic)]
[NodeLaunch("broadcasting")]
[NodeDispatchGrid(1,1,1)]
[numthreads(8, 4, 1)]
void Material_Metallic_Node(
	DispatchNodeInputRecord<DeferredShadingNodeRecord> inputData,
	uint2 groupThreadId : SV_GroupThreadID)
{
    EvaluateMaterialColor(inputData.Get(), groupThreadId, BT_Metallic);
}

[Shader("node")]
[NodeID("Material_Nodes",BT_Velvet)]
[NodeLaunch("broadcasting")]
[NodeDispatchGrid(1,1,1)]
[numthreads(8, 4, 1)]
void Material_Velvet_Node(
	DispatchNodeInputRecord<DeferredShadingNodeRecord> inputData,
	uint2 groupThreadId : SV_GroupThreadID)
{
    EvaluateMaterialColor(inputData.Get(), groupThreadId, BT_Velvet);
}

[Shader("node")]
[NodeID("Material_Nodes",BT_Flakes)]
[NodeLaunch("broadcasting")]
[NodeDispatchGrid(1,1,1)]
[numthreads(8, 4, 1)]
void Material_Flakes_Node(
	DispatchNodeInputRecord<DeferredShadingNodeRecord> inputData,
	uint2 groupThreadId : SV_GroupThreadID)
{
    EvaluateMaterialColor(inputData.Get(), groupThreadId, BT_Flakes);
}

[Shader("node")]
[NodeID("Material_Nodes",BT_Faceted)]
[NodeLaunch("broadcasting")]
[NodeDispatchGrid(1,1,1)]
[numthreads(8, 4, 1)]
void Material_Faceted_Node(
	DispatchNodeInputRecord<DeferredShadingNodeRecord> inputData,
	uint2 groupThreadId : SV_GroupThreadID)
{
    EvaluateMaterialColor(inputData.Get(), groupThreadId, BT_Faceted);
}

[Shader("node")]
[NodeID("Material_Nodes",BT_Stan)]
[NodeLaunch("broadcasting")]
[NodeDispatchGrid(1,1,1)]
[numthreads(8, 4, 1)]
void Material_Stan_Node(
	DispatchNodeInputRecord<DeferredShadingNodeRecord> inputData,
	uint2 groupThreadId : SV_GroupThreadID)
{
    EvaluateMaterialColor(inputData.Get(), groupThreadId, BT_Stan);
}

[Shader("node")]
[NodeID("Material_Nodes",BT_Checker)]
[NodeLaunch("broadcasting")]
[NodeDispatchGrid(1,1,1)]
[numthreads(8, 4, 1)]
void Material_Checker_Node(
	DispatchNodeInputRecord<DeferredShadingNodeRecord> inputData,
	uint2 groupThreadId : SV_GroupThreadID)
{
    EvaluateMaterialColor(inputData.Get(), groupThreadId, BT_Checker);
}