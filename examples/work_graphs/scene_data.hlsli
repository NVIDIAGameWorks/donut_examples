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

// enum AnimType
#define AT_Static 0
#define AT_RotateY 1
#define AT_Dance 2

// enum MaterialType
#define BT_Lambert 0
#define BT_Phong 1
#define BT_Metallic 2
#define BT_Velvet 3
#define BT_Flakes 4
#define BT_Faceted 5
#define BT_Stan 6
#define BT_Checker 7
#define BT_COUNT 8

// Root constant buffer
cbuffer SceneConstantBuffer : register(b1)
{
    float4x4 viewProj;
    float4x4 viewProjInverse;
    float4 camPosAndSceneTime;
    float4 camDir;
    float4 viewportSizeXY;
};

struct Instance
{
	float3 position;
	float rotationY;
	float3 size;
    uint meshType;
	uint material;
    uint animType;
};

struct Light
{
    float3 position;
	float3 target;
	float3 targetOffset;
	float3 color;
	float innerAngle;
	float outerAngle;
};

struct Material
{
	float3 baseColor;
	uint materialType;
	float3 param1;
	float param2;
	float param3;
};

struct AnimState
{
	uint state;
	uint stateRepeats;
	float statePeriod;
	float timeInState;

	float3 scale;
	float rotationY;
	float offsetY;
	float twist;
};

static const float c_PI = 3.1415926535897932384626433832795f;

uint Random(uint seed)
{
	// WangHash
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

float NormalizeRandom(uint rnd) { return rnd%0x43F2EC13 / (float)(0x43F2EC12); }

float3 RotateY(float3 v,float2 sinCos)
{
    return float3(v.x*sinCos.y - v.z*sinCos.x, v.y, v.x*sinCos.x + v.z*sinCos.y);
}
