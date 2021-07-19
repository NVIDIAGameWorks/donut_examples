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

#pragma pack_matrix(row_major)

#include <donut/shaders/bindless.h>
#include <donut/shaders/utils.hlsli>
#include <donut/shaders/vulkan.hlsli>
#include <donut/shaders/packing.hlsli>
#include <donut/shaders/surface.hlsli>
#include <donut/shaders/lighting.hlsli>
#include <donut/shaders/scene_material.hlsli>
#include "lighting_cb.h"

struct RayPayload
{
    float committedRayT;
    uint instanceID;
    uint primitiveIndex;
    uint geometryIndex;
    float2 barycentrics;
};

ConstantBuffer<LightingConstants> g_Const : register(b0);

RWTexture2D<float4> u_Output : register(u0);

RaytracingAccelerationStructure SceneBVH : register(t0);
StructuredBuffer<InstanceData> t_InstanceData : register(t1);
StructuredBuffer<GeometryData> t_GeometryData : register(t2);
StructuredBuffer<MaterialConstants> t_MaterialConstants : register(t3);

SamplerState s_MaterialSampler : register(s0);

VK_BINDING(0, 1) ByteAddressBuffer t_BindlessBuffers[] : register(t0, space1);
VK_BINDING(1, 1) Texture2D t_BindlessTextures[] : register(t0, space2);

RayDesc setupPrimaryRay(uint2 pixelPosition, PlanarViewConstants view)
{
    float2 uv = (float2(pixelPosition) + 0.5) * view.viewportSizeInv;
    float4 clipPos = float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.5, 1);
    float4 worldPos = mul(clipPos, view.matClipToWorld);
    worldPos.xyz /= worldPos.w;

    RayDesc ray;
    ray.Origin = view.cameraDirectionOrPosition.xyz;
    ray.Direction = normalize(worldPos.xyz - ray.Origin);
    ray.TMin = 0;
    ray.TMax = 1000;
    return ray;
}

RayDesc setupShadowRay(float3 surfacePos, float3 viewIncident)
{
    RayDesc ray;
    ray.Origin = surfacePos - viewIncident * 0.001;
    ray.Direction = -g_Const.light.direction;
    ray.TMin = 0;
    ray.TMax = 1000;
    return ray;
}

enum GeometryAttributes
{
    GeomAttr_Position   = 0x01,
    GeomAttr_TexCoord   = 0x02,
    GeomAttr_Normal     = 0x04,
    GeomAttr_Tangents   = 0x08,

    GeomAttr_All        = 0x0F
};

struct GeometrySample
{
    InstanceData instance;
    GeometryData geometry;
    MaterialConstants material;

    float3 vertexPositions[3];
    float2 vertexTexcoords[3];

    float3 objectSpacePosition;
    float2 texcoord;
    float3 flatNormal;
    float3 geometryNormal;
    float4 tangent;
};

GeometrySample getGeometryFromHit(
    uint instanceIndex,
    uint triangleIndex,
    uint geometryIndex,
    float2 rayBarycentrics,
    GeometryAttributes attributes,
    StructuredBuffer<InstanceData> instanceBuffer,
    StructuredBuffer<GeometryData> geometryBuffer,
    StructuredBuffer<MaterialConstants> materialBuffer)
{
    GeometrySample gs = (GeometrySample)0;

    gs.instance = instanceBuffer[instanceIndex];
    gs.geometry = geometryBuffer[gs.instance.firstGeometryIndex + geometryIndex];
    gs.material = materialBuffer[gs.geometry.materialIndex];

    ByteAddressBuffer indexBuffer = t_BindlessBuffers[NonUniformResourceIndex(gs.geometry.indexBufferIndex)];
    ByteAddressBuffer vertexBuffer = t_BindlessBuffers[NonUniformResourceIndex(gs.geometry.vertexBufferIndex)];

    float3 barycentrics;
    barycentrics.yz = rayBarycentrics;
    barycentrics.x = 1.0 - (barycentrics.y + barycentrics.z);

    uint3 indices = indexBuffer.Load3(gs.geometry.indexOffset + triangleIndex * c_SizeOfTriangleIndices);

    if (attributes & GeomAttr_Position)
    {
        gs.vertexPositions[0] = asfloat(vertexBuffer.Load3(gs.geometry.positionOffset + indices[0] * c_SizeOfPosition));
        gs.vertexPositions[1] = asfloat(vertexBuffer.Load3(gs.geometry.positionOffset + indices[1] * c_SizeOfPosition));
        gs.vertexPositions[2] = asfloat(vertexBuffer.Load3(gs.geometry.positionOffset + indices[2] * c_SizeOfPosition));
        gs.objectSpacePosition = interpolate(gs.vertexPositions, barycentrics);
    }

    if ((attributes & GeomAttr_TexCoord) && gs.geometry.texCoord1Offset != ~0u)
    {
        gs.vertexTexcoords[0] = asfloat(vertexBuffer.Load2(gs.geometry.texCoord1Offset + indices[0] * c_SizeOfTexcoord));
        gs.vertexTexcoords[1] = asfloat(vertexBuffer.Load2(gs.geometry.texCoord1Offset + indices[1] * c_SizeOfTexcoord));
        gs.vertexTexcoords[2] = asfloat(vertexBuffer.Load2(gs.geometry.texCoord1Offset + indices[2] * c_SizeOfTexcoord));
        gs.texcoord = interpolate(gs.vertexTexcoords, barycentrics);
    }

    if ((attributes & GeomAttr_Normal) && gs.geometry.normalOffset != ~0u)
    {
        float3 normals[3];
        normals[0] = Unpack_RGB8_SNORM(vertexBuffer.Load(gs.geometry.normalOffset + indices[0] * c_SizeOfNormal));
        normals[1] = Unpack_RGB8_SNORM(vertexBuffer.Load(gs.geometry.normalOffset + indices[1] * c_SizeOfNormal));
        normals[2] = Unpack_RGB8_SNORM(vertexBuffer.Load(gs.geometry.normalOffset + indices[2] * c_SizeOfNormal));
        gs.geometryNormal = interpolate(normals, barycentrics);
        gs.geometryNormal = mul(gs.instance.transform, float4(gs.geometryNormal, 0.0)).xyz;
        gs.geometryNormal = normalize(gs.geometryNormal);
    }

    if ((attributes & GeomAttr_Tangents) && gs.geometry.tangentOffset != ~0u)
    {
        float4 tangents[3];
        tangents[0] = Unpack_RGBA8_SNORM(vertexBuffer.Load(gs.geometry.tangentOffset + indices[0] * c_SizeOfNormal));
        tangents[1] = Unpack_RGBA8_SNORM(vertexBuffer.Load(gs.geometry.tangentOffset + indices[1] * c_SizeOfNormal));
        tangents[2] = Unpack_RGBA8_SNORM(vertexBuffer.Load(gs.geometry.tangentOffset + indices[2] * c_SizeOfNormal));
        gs.tangent.xyz = interpolate(tangents, barycentrics).xyz;
        gs.tangent.xyz = mul(gs.instance.transform, float4(gs.tangent.xyz, 0.0)).xyz;
        gs.tangent.xyz = normalize(gs.tangent.xyz);
        gs.tangent.w = tangents[0].w;
    }

    float3 objectSpaceFlatNormal = normalize(cross(
        gs.vertexPositions[1] - gs.vertexPositions[0],
        gs.vertexPositions[2] - gs.vertexPositions[0]));

    gs.flatNormal = normalize(mul(gs.instance.transform, float4(objectSpaceFlatNormal, 0.0)).xyz);

    return gs;
}

enum MaterialAttributes
{
    MatAttr_BaseColor    = 0x01,
    MatAttr_Emissive     = 0x02,
    MatAttr_Normal       = 0x04,
    MatAttr_MetalRough   = 0x08,
    MatAttr_Transmission = 0x10,

    MatAttr_All          = 0x1F
};

MaterialSample sampleGeometryMaterial(
    GeometrySample gs, 
    float2 texGrad_x, 
    float2 texGrad_y, 
    float mipLevel, // <-- Use a compile time constant for mipLevel, < 0 for aniso filtering
    MaterialAttributes attributes, 
    SamplerState materialSampler)
{
    MaterialTextureSample textures = DefaultMaterialTextures();

    if ((attributes & MatAttr_BaseColor) && (gs.material.baseOrDiffuseTextureIndex >= 0) && (gs.material.flags & MaterialFlags_UseBaseOrDiffuseTexture) != 0)
    {
        Texture2D diffuseTexture = t_BindlessTextures[NonUniformResourceIndex(gs.material.baseOrDiffuseTextureIndex)];
        
        if (mipLevel >= 0)
            textures.baseOrDiffuse = diffuseTexture.SampleLevel(materialSampler, gs.texcoord, mipLevel);
        else
            textures.baseOrDiffuse = diffuseTexture.SampleGrad(materialSampler, gs.texcoord, texGrad_x, texGrad_y);
    }

    if ((attributes & MatAttr_Emissive) && (gs.material.emissiveTextureIndex >= 0) && (gs.material.flags & MaterialFlags_UseEmissiveTexture) != 0)
    {
        Texture2D emissiveTexture = t_BindlessTextures[NonUniformResourceIndex(gs.material.emissiveTextureIndex)];
        
        if (mipLevel >= 0)
            textures.emissive = emissiveTexture.SampleLevel(materialSampler, gs.texcoord, mipLevel);
        else
            textures.emissive = emissiveTexture.SampleGrad(materialSampler, gs.texcoord, texGrad_x, texGrad_y);
    }
    
    if ((attributes & MatAttr_Normal) && (gs.material.normalTextureIndex >= 0) && (gs.material.flags & MaterialFlags_UseNormalTexture) != 0)
    {
        Texture2D normalsTexture = t_BindlessTextures[NonUniformResourceIndex(gs.material.normalTextureIndex)];
        
        if (mipLevel >= 0)
            textures.normal = normalsTexture.SampleLevel(materialSampler, gs.texcoord, mipLevel);
        else
            textures.normal = normalsTexture.SampleGrad(materialSampler, gs.texcoord, texGrad_x, texGrad_y);
    }

    if ((attributes & MatAttr_MetalRough) && (gs.material.metalRoughOrSpecularTextureIndex >= 0) && (gs.material.flags & MaterialFlags_UseMetalRoughOrSpecularTexture) != 0)
    {
        Texture2D specularTexture = t_BindlessTextures[NonUniformResourceIndex(gs.material.metalRoughOrSpecularTextureIndex)];

        if (mipLevel >= 0)
            textures.metalRoughOrSpecular = specularTexture.SampleLevel(materialSampler, gs.texcoord, mipLevel);
        else
            textures.metalRoughOrSpecular = specularTexture.SampleGrad(materialSampler, gs.texcoord, texGrad_x, texGrad_y);
    }

    if ((attributes & MatAttr_Transmission) && (gs.material.transmissionTextureIndex >= 0) && (gs.material.flags & MaterialFlags_UseTransmissionTexture) != 0)
    {
        Texture2D transmissionTexture = t_BindlessTextures[NonUniformResourceIndex(gs.material.transmissionTextureIndex)];

        if (mipLevel >= 0)
            textures.transmission = transmissionTexture.SampleLevel(materialSampler, gs.texcoord, mipLevel);
        else
            textures.transmission = transmissionTexture.SampleGrad(materialSampler, gs.texcoord, texGrad_x, texGrad_y);
    }

    return EvaluateSceneMaterial(gs.geometryNormal, gs.tangent, gs.material, textures);
}

float3 shadeSurface(
    uint2 pixelPosition,
    RayPayload payload,
    float3 viewDirection)
{
    GeometrySample gs = getGeometryFromHit(payload.instanceID, payload.primitiveIndex, payload.geometryIndex, payload.barycentrics, 
        GeomAttr_All, t_InstanceData, t_GeometryData, t_MaterialConstants);
    
    RayDesc ray_0 = setupPrimaryRay(pixelPosition, g_Const.view);
    RayDesc ray_x = setupPrimaryRay(pixelPosition + uint2(1, 0), g_Const.view);
    RayDesc ray_y = setupPrimaryRay(pixelPosition + uint2(0, 1), g_Const.view);
    float3 worldSpacePositions[3];
    worldSpacePositions[0] = mul(gs.instance.transform, float4(gs.vertexPositions[0], 1.0)).xyz;
    worldSpacePositions[1] = mul(gs.instance.transform, float4(gs.vertexPositions[1], 1.0)).xyz;
    worldSpacePositions[2] = mul(gs.instance.transform, float4(gs.vertexPositions[2], 1.0)).xyz;
    float3 bary_0 = computeRayIntersectionBarycentrics(worldSpacePositions, ray_0.Origin, ray_0.Direction);
    float3 bary_x = computeRayIntersectionBarycentrics(worldSpacePositions, ray_x.Origin, ray_x.Direction);
    float3 bary_y = computeRayIntersectionBarycentrics(worldSpacePositions, ray_y.Origin, ray_y.Direction);
    float2 texcoord_0 = interpolate(gs.vertexTexcoords, bary_0);
    float2 texcoord_x = interpolate(gs.vertexTexcoords, bary_x);
    float2 texcoord_y = interpolate(gs.vertexTexcoords, bary_y);
    float2 texGrad_x = texcoord_x - texcoord_0;
    float2 texGrad_y = texcoord_y - texcoord_0;

    MaterialSample ms = sampleGeometryMaterial(gs, texGrad_x, texGrad_y, -1, MatAttr_All, s_MaterialSampler);

    ms.shadingNormal = getBentNormal(gs.flatNormal, ms.shadingNormal, viewDirection);

    float3 worldPos = mul(gs.instance.transform, float4(gs.objectSpacePosition, 1.0)).xyz;

    float3 diffuseTerm = 0, specularTerm = 0;

    RayDesc shadowRay = setupShadowRay(worldPos, viewDirection);
    payload.instanceID = ~0u;
    TraceRay(SceneBVH, RAY_FLAG_NONE, 0xff, 0, 0, 0, shadowRay, payload);
    if (payload.instanceID == ~0u)
    {
        ShadeSurface(g_Const.light, ms, worldPos, viewDirection, diffuseTerm, specularTerm);
    }

    return diffuseTerm + specularTerm + ms.diffuseAlbedo * g_Const.ambientColor.rgb;
}
struct Attributes 
{
    float2 uv;
};

[shader("miss")]
void Miss(inout RayPayload payload : SV_RayPayload)
{
}

[shader("closesthit")]
void ClosestHit(inout RayPayload payload : SV_RayPayload, in Attributes attrib : SV_IntersectionAttributes)
{
    payload.committedRayT = RayTCurrent();
    payload.instanceID = InstanceID();
    payload.primitiveIndex = PrimitiveIndex();
    payload.geometryIndex = GeometryIndex();
    payload.barycentrics = attrib.uv;
}


[shader("anyhit")]
void AnyHit(inout RayPayload payload : SV_RayPayload, in Attributes attrib : SV_IntersectionAttributes)
{
    GeometrySample gs = getGeometryFromHit(InstanceID(), PrimitiveIndex(), GeometryIndex(), attrib.uv, 
        GeomAttr_TexCoord, t_InstanceData, t_GeometryData, t_MaterialConstants);
    
    MaterialSample ms = sampleGeometryMaterial(gs, 0, 0, 0, MatAttr_BaseColor | MatAttr_Transmission, s_MaterialSampler);

    bool alphaMask = ms.opacity >= gs.material.alphaCutoff;

    if (gs.material.domain == MaterialDomain_AlphaTested && !alphaMask)
        IgnoreHit();
}

[shader("raygeneration")]
void RayGen()
{
    uint2 pixelPosition = DispatchRaysIndex().xy;

    RayDesc ray = setupPrimaryRay(pixelPosition, g_Const.view);
    
    RayPayload payload;
    payload.committedRayT = 0;
    payload.instanceID = ~0u;
    payload.primitiveIndex = 0;
    payload.barycentrics = 0;

    TraceRay(SceneBVH, RAY_FLAG_NONE, 0xff, 0, 0, 0, ray, payload);

    if (payload.instanceID == ~0u)
    {
        u_Output[pixelPosition] = 0;
        return;
    }

    u_Output[pixelPosition] = float4(shadeSurface(pixelPosition, payload, ray.Direction), 0);
}
