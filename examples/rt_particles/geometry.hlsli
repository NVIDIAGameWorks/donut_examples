/*
* Copyright (c) 2014-2023, NVIDIA CORPORATION. All rights reserved.
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

// Estimate the mip level for sampling a texture on a triangle hit by a ray,
// assuming that the ray hits the triangle from a normal direction and that the texture
// mapping has uniform density in u and v dimensions.
// Ray footprint should be calculated as (primaryConeAngle * accumulatedHitDistance),
// which easily works across reflections and refractions, unlike ray differentials.
// For higher quality results, use anisotropic ray cones instead.
float getTextureMipLevel(GeometrySample gs, float rayFootprint, Texture2D textureObject)
{
    int2 textureSize;
    textureObject.GetDimensions(textureSize.x, textureSize.y);

    float2 vertexTexcoords[3];
    vertexTexcoords[0] = gs.vertexTexcoords[0] * float2(textureSize);
    vertexTexcoords[1] = gs.vertexTexcoords[1] * float2(textureSize);
    vertexTexcoords[2] = gs.vertexTexcoords[2] * float2(textureSize);

    float3 worldSpacePositions[3];
    worldSpacePositions[0] = mul(gs.instance.transform, float4(gs.vertexPositions[0], 1.0)).xyz;
    worldSpacePositions[1] = mul(gs.instance.transform, float4(gs.vertexPositions[1], 1.0)).xyz;
    worldSpacePositions[2] = mul(gs.instance.transform, float4(gs.vertexPositions[2], 1.0)).xyz;

    const float texelSpaceArea = length(cross(float3(vertexTexcoords[1] - vertexTexcoords[0], 0), float3(vertexTexcoords[2] - vertexTexcoords[0], 0)));
    const float worldSpaceArea = length(cross(worldSpacePositions[1] - worldSpacePositions[0], worldSpacePositions[2] - worldSpacePositions[0]));

    if (worldSpaceArea == 0)
        return 0;

    const float texelsPerWorldUnit = sqrt(texelSpaceArea / worldSpaceArea);

    return log2(texelsPerWorldUnit * rayFootprint);
}

MaterialSample sampleGeometryMaterial(
    GeometrySample gs,
    float rayFootprint,
    MaterialAttributes attributes, 
    SamplerState materialSampler)
{
    MaterialTextureSample textures = DefaultMaterialTextures();

    float mipLevel = 0;

    if ((attributes & MatAttr_BaseColor) && (gs.material.baseOrDiffuseTextureIndex >= 0) && (gs.material.flags & MaterialFlags_UseBaseOrDiffuseTexture) != 0)
    {
        Texture2D diffuseTexture = t_BindlessTextures[NonUniformResourceIndex(gs.material.baseOrDiffuseTextureIndex)];

        // Use the base color texture to get the mip level. Good enough for sample purposes.
        mipLevel = getTextureMipLevel(gs, rayFootprint, diffuseTexture);

        textures.baseOrDiffuse = diffuseTexture.SampleLevel(materialSampler, gs.texcoord, mipLevel);
    }

    if ((attributes & MatAttr_Emissive) && (gs.material.emissiveTextureIndex >= 0) && (gs.material.flags & MaterialFlags_UseEmissiveTexture) != 0)
    {
        Texture2D emissiveTexture = t_BindlessTextures[NonUniformResourceIndex(gs.material.emissiveTextureIndex)];
        
        textures.emissive = emissiveTexture.SampleLevel(materialSampler, gs.texcoord, mipLevel);
    }
    
    if ((attributes & MatAttr_Normal) && (gs.material.normalTextureIndex >= 0) && (gs.material.flags & MaterialFlags_UseNormalTexture) != 0)
    {
        Texture2D normalsTexture = t_BindlessTextures[NonUniformResourceIndex(gs.material.normalTextureIndex)];
        
        textures.normal = normalsTexture.SampleLevel(materialSampler, gs.texcoord, mipLevel);
    }

    if ((attributes & MatAttr_MetalRough) && (gs.material.metalRoughOrSpecularTextureIndex >= 0) && (gs.material.flags & MaterialFlags_UseMetalRoughOrSpecularTexture) != 0)
    {
        Texture2D specularTexture = t_BindlessTextures[NonUniformResourceIndex(gs.material.metalRoughOrSpecularTextureIndex)];
        
        textures.metalRoughOrSpecular = specularTexture.SampleLevel(materialSampler, gs.texcoord, mipLevel);
    }

    if ((attributes & MatAttr_Transmission) && (gs.material.transmissionTextureIndex >= 0) && (gs.material.flags & MaterialFlags_UseTransmissionTexture) != 0)
    {
        Texture2D transmissionTexture = t_BindlessTextures[NonUniformResourceIndex(gs.material.transmissionTextureIndex)];
        
        textures.transmission = transmissionTexture.SampleLevel(materialSampler, gs.texcoord, mipLevel);
    }

    return EvaluateSceneMaterial(gs.geometryNormal, gs.tangent, gs.material, textures);
}
