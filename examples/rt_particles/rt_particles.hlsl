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

#pragma pack_matrix(row_major)

#include <donut/shaders/bindless.h>
#include <donut/shaders/utils.hlsli>
#include <donut/shaders/vulkan.hlsli>
#include <donut/shaders/packing.hlsli>
#include <donut/shaders/surface.hlsli>
#include <donut/shaders/lighting.hlsli>
#include <donut/shaders/scene_material.hlsli>
#include "rt_particles_cb.h"
#include "mlab.hlsli"
#include "utils.hlsli"

VK_BINDING(0, 1) ByteAddressBuffer t_BindlessBuffers[] : register(t0, space1);
VK_BINDING(1, 1) Texture2D t_BindlessTextures[] : register(t0, space2);

#include "geometry.hlsli"

ConstantBuffer<GlobalConstants> g_Const : register(b0);

RWTexture2D<float4> u_Output : register(u0);

RaytracingAccelerationStructure SceneBVH : register(t0);
StructuredBuffer<InstanceData> t_InstanceData : register(t1);
StructuredBuffer<GeometryData> t_GeometryData : register(t2);
StructuredBuffer<MaterialConstants> t_MaterialConstants : register(t3);
StructuredBuffer<ParticleInfo> t_ParticleInfos : register(t4);

SamplerState s_MaterialSampler : register(s0);

// Estimate the mip level for sampling a texture on a particle billboard hit by a ray.
// Same assumptions and limitations as in getTextureMipLevel(...) - see geometry.hlsli
float getParticleMipLevel(ParticleInfo particle, float rayFootprint, Texture2D textureObject)
{
    int2 textureSize;
    textureObject.GetDimensions(textureSize.x, textureSize.y);

    // All particles map the entire texture to a square billboard with edge length = (2 * radius).
    const float texelsPerWorldUnit = max(textureSize.x, textureSize.y) * 0.5 * particle.inverseRadius;

    return log2(texelsPerWorldUnit * rayFootprint);
}

float3x3 getIdentityMatrix()
{
    return float3x3(
        1, 0, 0,
        0, 1, 0,
        0, 0, 1
    );
}

float3x3 getReflectionMatrix(float3 normal)
{
    // https://en.wikipedia.org/wiki/Transformation_matrix#Reflection_2
    // Note: This matrix is involutory, meaning it is equal to its own inverse. This means it can
    // be used in either direction for a transformation without issue.

    const float a = normal.x;
    const float b = normal.y;
    const float c = normal.z;

    return float3x3(
        1 - 2 * a * a, -2 * a * b, -2 * a * c,
        -2 * a * b, 1 - 2 * b * b, -2 * b * c,
        -2 * a * c, -2 * b * c, 1 - 2 * c * c
    );
}

// Returns the radiance and opacity (.a) of a geometric particle (i.e. represented by triangles)
// at the hit position specified by hitInfo.
float4 getGeometricParticleColor(RayHitInfo hitInfo, uint particleIndex, float accumulatedHitDistance)
{
    GeometrySample gs = getGeometryFromHit(hitInfo.instanceID, hitInfo.primitiveIndex, hitInfo.geometryIndex, hitInfo.barycentrics,
        GeomAttr_Position | GeomAttr_TexCoord, t_InstanceData, t_GeometryData, t_MaterialConstants);
    
    const float rayFootprint = (accumulatedHitDistance + hitInfo.hitT) * g_Const.primaryRayConeAngle;

    MaterialSample ms = sampleGeometryMaterial(gs, rayFootprint, MatAttr_BaseColor, s_MaterialSampler);

    ParticleInfo particle = t_ParticleInfos[particleIndex];

    // Apply the particle color.
    ms.baseColor *= particle.colorFactor;
    ms.opacity *= particle.opacityFactor;

    return float4(ms.baseColor, ms.opacity);
}

// Returns the radiance and opacity (.a) of an intersection particle (i.e. represented by the AABB)
// at the point where it intersects the given ray, if it does. If there is no intersection, returns 0.
float4 getIntersectionParticleColor(RayDesc ray, uint particleIndex, float accumulatedHitDistance,
    float3x3 accumulatedVectorTransform, out float hitT)
{
    ParticleInfo particle = t_ParticleInfos[particleIndex];
    
    // For secondary rays, we don't have the accumulatedVectorTransform, so get creative:
    // reorient the particle so that it faces the virtual camera, i.e. the image of the camera
    // below the surface where the ray originated from. Do not use the ray origin for the 
    // virtual camera position because that results in heavy distortions of particle shapes
    // in reflections.

    const float3 virtualCameraPosition = ray.Origin - ray.Direction * accumulatedHitDistance;

    // New normal
    const float3 cameraFacingNormal = normalize(virtualCameraPosition - particle.center);

    // Original normal
    float3 normal = normalize(cross(particle.xAxis, particle.yAxis));
    
    if (g_Const.orientationMode == ORIENTATION_MODE_AVT_MATRIX)
    {
        // If the AVT matrix is available, use it to transform the particle deterministically.
        particle.xAxis = mul(particle.xAxis, accumulatedVectorTransform);
        particle.yAxis = mul(particle.yAxis, accumulatedVectorTransform);
        normal = mul(normal, accumulatedVectorTransform);
    }
    else if (g_Const.orientationMode == ORIENTATION_MODE_BEAM)
    {
        // Beams - particle billboards that can rotate only around one axis, Y in our case.
        
        // Find the orientation of a cylinder section along the Y axis that provides the largest
        // angle between the section plane and the incoming ray.
        float3 xAxis = cross(particle.yAxis, cameraFacingNormal);

        // Use the normalized cross product as the X axis. If the ray is parallel to the Y axis,
        // just keep the existing X axis because it doesn't matter - there will be no intersection.
        float xLength = length(xAxis);
        if (xLength > 0)
            particle.xAxis = xAxis / xLength;

        normal = cross(particle.xAxis, particle.yAxis);
    }
    else if (g_Const.orientationMode == ORIENTATION_MODE_BASIS)
    {
        // Come up with a basis based on a default world "up" vector.
        // This is based on the "Billboard Ray Tracing for Impostors and Volumetric Effects"
        // chapter from the book Ray Tracing Gems 2.
        normal = cameraFacingNormal;

        float3 up = float3(0, 1, 0);
        if (abs(dot(up, normal)) >= 0.999)
            up = float3(1, 0, 0);

        float3 right = normalize(cross(up, normal));
        up = normalize(cross(normal, right));

        // Recalculate the particle orientation based on its rotation, as it's done on the host side.
        const float2 localRight = float2(cos(particle.rotation), sin(particle.rotation));
        const float2 localUp = float2(-localRight.y, localRight.x);

        particle.xAxis = localRight.x * right + localRight.y * up;
        particle.yAxis = localUp.x * right + localUp.y * up;
    }
    else // if (g_Const.orientationMode == ORIENTATION_MODE_QUATERNION)
    {
        // If the new normal is facing the opposite direction, just flip it, because
        // rotation becomes unstable near the opposite pole. We don't care much for the exact particle
        // rotation in reflections etc. but stability is desirable.
        if (dot(normal, cameraFacingNormal) < 0.0)
            normal = -normal;

        // Find a quaternion that transforms the original normal into the new one.
        const float4 rotationQuat = quaternionCreateOrientation(normal, cameraFacingNormal);

        // Use that quaternion to rotate the particle.
        // This is more predictable than re-creating a basis just from the normal.
        particle.xAxis = quaternionTransformVector(rotationQuat, particle.xAxis);
        particle.yAxis = quaternionTransformVector(rotationQuat, particle.yAxis);
        normal = cameraFacingNormal;
    }

    // Compute the T value of the ray-plane intersection.
    const float NdotD = dot(ray.Direction, normal);
    hitT = dot(particle.center - ray.Origin, normal) / NdotD;
    if (isinf(hitT) || isnan(hitT)) hitT = -1.0;

    // Make sure that the plane is within our ray's range.
    if (hitT < ray.TMin || hitT > ray.TMax)
        return 0;

    // Find the point on the plane where the ray intersects it.
    float3 intersectionPoint = ray.Origin + ray.Direction * hitT;

    // Compute the intersection point coordinates in the particle's 2D basis.
    float3 centerToIntersection = intersectionPoint - particle.center.xyz;
    float2 uv;
    uv.x = dot(centerToIntersection, particle.xAxis) * particle.inverseRadius;
    uv.y = dot(centerToIntersection, particle.yAxis) * particle.inverseRadius;

    // Make sure that the intersection is within the particle quad.
    if (max(abs(uv.x), abs(uv.y)) >= 1.0)
        return 0;

    // Translate from the [-1, 1] UV range to [0, 1] texture coordinates.
    // Real applications might want to store the basis vectors in texture space, too.
    uv.y = -uv.y;
    uv = uv * 0.5 + 0.5;

    // Get the bindless texture for this particle.
    Texture2D diffuseTexture = t_BindlessTextures[NonUniformResourceIndex(particle.textureIndex)];

    // Calculate the texture mip level.
    const float rayFootprint = (accumulatedHitDistance + hitT) * g_Const.primaryRayConeAngle;
    const float mipLevel = getParticleMipLevel(particle, rayFootprint, diffuseTexture);

    // Sample the textue.
    float4 baseColor = diffuseTexture.SampleLevel(s_MaterialSampler, uv, mipLevel);

    // Apply the particle color.
    baseColor.rgb *= particle.colorFactor;
    baseColor.a *= particle.opacityFactor;

    return baseColor;
}

// Traces a ray looking for particles, returns the accumulated radiance and transmittance.
BlendFragment accumulateParticles(RayDesc ray, float accumulatedHitDistance, float3x3 accumulatedVectorTransform, bool isSecondaryRay)
{
    // Use intersection particles if re-orientation is needed for this type of ray
    // (primary or secondary), per user settings. Primary and secondary rays generally
    // have different directions, which is what matters here.
    const bool useIntersectionPrimitives = isSecondaryRay
        ? g_Const.reorientParticlesInSecondaryRays
        : g_Const.reorientParticlesInPrimaryRays;

    // Select the right set of particles based on the primitive type we're looking for.
    // Could also use RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES or RAY_FLAG_SKIP_TRIANGLES.
    const uint rayMask = useIntersectionPrimitives
        ? INSTANCE_MASK_INTERSECTION_PARTICLE
        : INSTANCE_MASK_PARTICLE_GEOMETRY;

    RayQuery<RAY_FLAG_NONE> rayQuery;
    rayQuery.TraceRayInline(SceneBVH, RAY_FLAG_NONE, rayMask, ray);

    // Initialize the blending array.
    // See mlab.hlsli for more information.
    BlendFragment buffer[MLAB_FRAGMENTS];
    blendInit(buffer);

    while (rayQuery.Proceed())
    {
        float4 particleColor = 0;
        float particleDistance = 0;
        uint particleIndex = 0;

        if (rayQuery.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
        {
            RayHitInfo hitInfo;

            // Fill the hitInfo structure with candidate hit parameters.
            RAY_QUERY_CANDIDATE_HIT(hitInfo, rayQuery);

            // Load the ParticleInfo struture for this particle.
            // The particles are all stored in a single geometry, with two primitives per particle,
            // so the particle index is just (primitiveIndex / 2).
            particleIndex = hitInfo.primitiveIndex / 2;

            particleColor = getGeometricParticleColor(hitInfo, particleIndex, accumulatedHitDistance);
            particleDistance = hitInfo.hitT;
        }
        else if (rayQuery.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE)
        {
            // Particle index is stored in the instance's custom ID field.
            particleIndex = rayQuery.CandidateInstanceID();

            particleColor = getIntersectionParticleColor(ray, particleIndex, accumulatedHitDistance, accumulatedVectorTransform,
                /* out */ particleDistance);
        }

        // Skip fragments that are completely transparent.
        if (particleColor.a == 0)
            continue;
        
        // Insert the fragment into the blending array.
        BlendFragment f;
        f.color = particleColor.rgb * particleColor.a;
        f.attenuation = 1.0 - particleColor.a;
        f.depth = particleDistance;
        blendInsert(f, buffer);
    }

    // Integrate the blending array into one fragment.
    return blendIntegrate(buffer);
}

// Traces a ray looking for an opaque surface, returns the hit (if found) or instanceID = c_MissInstanceID (if not)
RayHitInfo findOpaqueSurface(RayDesc ray, uint rayMask, float accumulatedHitDistance)
{
    RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_CULL_BACK_FACING_TRIANGLES | RAY_FLAG_CULL_NON_OPAQUE> rayQuery;
    rayQuery.TraceRayInline(SceneBVH, RAY_FLAG_NONE, rayMask, ray);

    // Trace one ray segment for simplicity.
    // The ray query ignores procedural and non-opaque primitives per the ray flags above.
    rayQuery.Proceed();

    RayHitInfo hitInfo;
    hitInfo.instanceID = c_MissInstanceID;

    if (rayQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
        // Fill the hitInfo structure with hit parameters.
        RAY_QUERY_COMMITTED_HIT(hitInfo, rayQuery);
    }

    return hitInfo;
}

// Returns the radiance of an opaque surface at the ray hit, NOT including the following path segments
float3 shadeOpaqueSurface(
    RayHitInfo hitInfo,
    float accumulatedHitDistance,
    float3 viewDirection,
    out float3 normal)
{
    GeometrySample gs = getGeometryFromHit(hitInfo.instanceID, hitInfo.primitiveIndex, hitInfo.geometryIndex, hitInfo.barycentrics,
        GeomAttr_All, t_InstanceData, t_GeometryData, t_MaterialConstants);

    const float rayFootprint = (accumulatedHitDistance + hitInfo.hitT) * g_Const.primaryRayConeAngle;
    
    MaterialSample ms = sampleGeometryMaterial(gs, rayFootprint, MatAttr_BaseColor, s_MaterialSampler);

    normal = ms.shadingNormal;

    // Lighting is not important in this sample app
    return ms.diffuseAlbedo;
}

[numthreads(16, 16, 1)]
void main(uint2 pixelPosition : SV_DispatchThreadID)
{
    RayDesc ray = setupPrimaryRay(pixelPosition, g_Const.view);

    float3 finalColor = 0;
    float attenuation = 1.0;
    float accumulatedHitDistance = 0;
    float3x3 accumulatedVectorTransform = getIdentityMatrix();

    // Trace a path starting at the camera.
    for (int bounce = 0; bounce < 8; ++bounce)
    {
        RayHitInfo hitInfo = findOpaqueSurface(ray, INSTANCE_MASK_OPAQUE, accumulatedHitDistance);
        const bool hasHit = hitInfo.instanceID != c_MissInstanceID;

        // If we hit something with the primary or secondary ray, shade that.
        float3 surfaceColor = 0;
        float3 surfaceNormal = 0;
        if (hasHit)
        {
            surfaceColor = shadeOpaqueSurface(hitInfo, accumulatedHitDistance, ray.Direction, surfaceNormal);
            ray.TMax = hitInfo.hitT;
        }

        // Trace a ray looking for particles.
        BlendFragment particles = accumulateParticles(ray, accumulatedHitDistance, accumulatedVectorTransform, bounce > 0);

        // Blend the particles over the regular geometry.
        float3 segmentColor = particles.color + surfaceColor * particles.attenuation;
        finalColor += segmentColor * attenuation;
        attenuation *= particles.attenuation;

        // If there's no hit, stop the path.
        if (!hasHit)
            break;

        // Continue the path in the reflected direction.
        ray.Origin = ray.Origin + ray.Direction * hitInfo.hitT + surfaceNormal * 1e-3;
        ray.Direction = reflect(ray.Direction, surfaceNormal);
        ray.TMax = 1000;

        // Make the reflections progressively dimmer.
        // Should be Fresnel and other material parameters, but that's not important here.
        attenuation *= 0.8;

        // Accumulate the path length to make sure that particle orientation is correct
        // in the secondary rays and that proper texture mip levels are selected.
        accumulatedHitDistance += hitInfo.hitT;

        accumulatedVectorTransform = mul(accumulatedVectorTransform, getReflectionMatrix(surfaceNormal));
    }

    if (g_Const.environmentMapTextureIndex >= 0)
    {
        Texture2D environmentMap = t_BindlessTextures[g_Const.environmentMapTextureIndex];
        const float2 uv = directionToEquirectUV(ray.Direction);
        const float3 environmentRadiance = environmentMap.SampleLevel(s_MaterialSampler, uv, 0).rgb;
        finalColor += environmentRadiance * attenuation;
    }

    u_Output[pixelPosition] = float4(finalColor, 0);
}
