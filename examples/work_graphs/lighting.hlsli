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

static const uint c_MaxLightsPerTile = 64; // This value must match DeferredShadingParam_MaxLightsPerTile defined in work_graphs_d3d12.cpp

float3 Unproject(uint2 pixelXY, float depth)
{
    const float3 clipSpacePos = float3(2.0f*pixelXY.x/viewportSizeXY.x-1.0f, -(2.0f*pixelXY.y/viewportSizeXY.y-1.0f), depth);
    float4 worldSpacePos = mul(float4(clipSpacePos,1), viewProjInverse);
    return worldSpacePos.xyz / worldSpacePos.w;
}

bool PointInSpotLight(uint2 pixelXY, float depth, Light light)
{
    const float3 worldPosition = Unproject(pixelXY, depth);

    const float3 lightTarget = light.target+light.targetOffset;
    const float lightLengthSq = dot(lightTarget-light.position, lightTarget-light.position);
    const float3 lightDir = (lightTarget-light.position);
    const float3 lightToPoint = (worldPosition-light.position);

    const float cosOuterAngle = cos(light.outerAngle*0.5f);
    const float cosAlpha = dot(lightToPoint, lightDir);
    return (cosAlpha >= cosOuterAngle) && (dot(lightToPoint, lightToPoint) <= lightLengthSq);
}

void EvaluateSpotLight(Light light, float3 worldPosition, out float3 outDirection, out float3 outColor, out float outAttenuation)
{
    const float3 lightTarget = light.target+light.targetOffset;
    const float lightLength = length(lightTarget-light.position);
    const float3 lightDir = normalize(lightTarget-light.position);
    const float3 lightToPoint = worldPosition-light.position; // Starting at light, pointing towards worldPosition (incoming onto worldPosition)
    const float3 lightToPointDir = normalize(lightToPoint);

    const float cosInnerAngle = cos(light.innerAngle*0.5f);
    const float cosOuterAngle = cos(light.outerAngle*0.5f);
    const float cosAlpha = saturate(dot(lightToPointDir, lightDir));

    const float distanceAttenuation = 1-saturate(length(lightToPoint)/lightLength);
    const float angleAttenuation = saturate((cosAlpha-cosOuterAngle)/(cosInnerAngle-cosOuterAngle));

    outDirection = lightToPointDir;
    outColor = light.color;
    outAttenuation = angleAttenuation*distanceAttenuation;
}

float3 EvaluateSky(uint2 pixelXY)
{
    float2 sinCosRotY;
    sincos(camPosAndSceneTime.w*-0.24f,sinCosRotY.x,sinCosRotY.y);
    const float3 worldPos = Unproject(pixelXY, 1);
    const float3 pointToCamDir = RotateY(normalize(worldPos-camPosAndSceneTime.xyz),sinCosRotY);

    const int sections = 125;
	const float2 uv = float2(
        (atan2(pointToCamDir.z,pointToCamDir.x)/c_PI+1)*0.5f,
	    acos(pointToCamDir.y)/c_PI);
	const uint2 sector = uint2(uv*sections);
	const float2 facetedUV = sector/(float2)sections;
    const float2 fractionUV = (uv-facetedUV)*sections*float2(2,1);

    const uint r0 = Random(sector.x);
	const uint r1 = Random(sector.y);
	const uint r2 = Random(r0+r1);
    const uint enable = ((r0 & 1) == (r1 & 1)) && ((sector.x % 2) == (sector.y % 3)) ? 1 : 0;
    const float timev = (r2%1003/1002.0f)*0.5f+0.5f;
    const float brightness = abs(fmod(camPosAndSceneTime.w,timev)-timev*0.5f);
    const float2 clr = uint2(r0,r1)%343/342.0f;

    const float d = pow(1-length(fractionUV-float2(0.25f,0.5f)),10) * 5.0f;
    return float3(clr*d*enable*brightness,0);
}