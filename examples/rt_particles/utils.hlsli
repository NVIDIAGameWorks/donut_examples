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

struct RayHitInfo
{
    float hitT;
    uint instanceID;
    uint primitiveIndex;
    uint geometryIndex;
    float2 barycentrics;
};

static const uint c_MissInstanceID = ~0u;

// Constructor to make RayHitInfo from a committed hit on a RayQuery.
// Can't make it a function because RayQuery<Flags> is a template.
#define RAY_QUERY_COMMITTED_HIT(hitInfo, rayQuery) \
    hitInfo.instanceID = rayQuery.CommittedInstanceID(); \
    hitInfo.primitiveIndex = rayQuery.CommittedPrimitiveIndex(); \
    hitInfo.geometryIndex = rayQuery.CommittedGeometryIndex(); \
    hitInfo.barycentrics = rayQuery.CommittedTriangleBarycentrics(); \
    hitInfo.hitT = rayQuery.CommittedRayT();

// Constructor to make RayHitInfo from a candidate hit on a RayQuery.
#define RAY_QUERY_CANDIDATE_HIT(hitInfo, rayQuery) \
    hitInfo.instanceID = rayQuery.CandidateInstanceID(); \
    hitInfo.primitiveIndex = rayQuery.CandidatePrimitiveIndex(); \
    hitInfo.geometryIndex = rayQuery.CandidateGeometryIndex(); \
    hitInfo.barycentrics = rayQuery.CandidateTriangleBarycentrics(); \
    hitInfo.hitT = rayQuery.CandidateTriangleRayT();

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

// Calculates the quaternion which transforms the normalized vector src to normalized vector dst
// Note: returns a zero quat if src == -dst, case when dot(src, dst) == -1 needs to be handled if this is expected.
float4 quaternionCreateOrientation(float3 src, float3 dst)
{
    // https://stackoverflow.com/questions/1171849/finding-quaternion-representing-the-rotation-from-one-vector-to-another
    float4 q;
    q.xyz = cross(src, dst);
    q.w = 1.0 + dot(src, dst);

    // Normalize the quaternion
    float len = length(q);
    q = (len > 0) ? q / len : float4(1.0, 0.0, 0.0, 0.0);

    return q;
}

// Transform a vector v with a quaternion q, v doesn't need to be normalized
float3 quaternionTransformVector(float4 q, float3 v)
{
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

// Converts a unit direction vector into texture coordinates for sampling a lat-long environment map
float2 directionToEquirectUV(float3 normalizedDirection)
{
    float elevation = asin(normalizedDirection.y);
    float azimuth = 0;
    if (abs(normalizedDirection.y) < 1.0)
        azimuth = atan2(normalizedDirection.z, normalizedDirection.x);

    const float c_pi = 3.1415926535;
    float2 uv;
    uv.x = azimuth / (2 * c_pi) - 0.25;
    uv.y = 0.5 - elevation / c_pi;

    return uv;
}
