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

// These are root 32-bit values
cbuffer InlineConstants : register(b0)
{
    float g_Time;
    float g_TimeDiff;
    uint g_ResetState;
};

StructuredBuffer<Instance> t_InstanceData : register(t0);
RWStructuredBuffer<AnimState> u_AnimStateData : register(u0);
RWStructuredBuffer<Light> u_LightData : register(u0);

#define DANCE_BUMP 0
#define DANCE_JUMP 1
#define DANCE_JUMPTWIST 2
#define DANCE_ROCK 3
#define DANCE_TWIST 4
#define DANCE_COUNT 5

float IntroOutro(float time,float introLength)
{
    if (time < introLength)
        return time/introLength;
    if (time > 1-introLength)
        return (1-time)/introLength;
    return 1.0f;
}

float CosineInterpolation(float v) { return (1-cos(v*c_PI))/2; }

void StepDance(inout AnimState state, uint rndSeed)
{
    state.timeInState += g_TimeDiff;

    if (state.timeInState >= state.statePeriod)
    {
        state.scale = float3(1,1,1);
        state.offsetY = 0;
        state.rotationY = 0;
        state.twist = 0;
        state.timeInState = 0;
        if (state.stateRepeats > 0)
            state.stateRepeats--;
        else
        {
            rndSeed = Random(rndSeed+state.state);
            uint oldState = state.state;
            state.state = rndSeed % DANCE_COUNT;
            if (oldState == state.state)
                state.state = (state.state+1) % DANCE_COUNT;
            rndSeed = Random(rndSeed);
            state.statePeriod = lerp(0.65f,0.8f,NormalizeRandom(rndSeed));
            rndSeed = Random(rndSeed);
            state.stateRepeats = 2 + rndSeed % 5;
        }
    }

    const float timeInStateNrm = state.timeInState / state.statePeriod;
    switch (state.state)
    {
    case DANCE_BUMP:
        state.scale.y = lerp(1,0.3f,CosineInterpolation(abs(timeInStateNrm*4)));
        break;

    case DANCE_JUMP:
    case DANCE_JUMPTWIST:
        if (timeInStateNrm < 0.4f)
        {
            const float compress = timeInStateNrm/0.4f;
            state.scale.y = lerp(1,0.3f,CosineInterpolation(compress));
            state.scale.xz = lerp(1.6,1,state.scale.y);
        }
        else if (timeInStateNrm < 0.6f) { }
        else if (timeInStateNrm < 0.9f)
        {
            const float shoot = (timeInStateNrm-0.6f)/0.3f;
            state.scale.y = lerp(0.3f,1.3f,sin(shoot*c_PI*0.5f));
            state.scale.xz = lerp(1.6,1,shoot);
            state.offsetY = shoot * 20.0f;
            state.rotationY = (state.state == DANCE_JUMPTWIST) ? lerp(0,c_PI*0.5f,CosineInterpolation(shoot)) : 0;
        }
        else
        {
            const float land = (timeInStateNrm-0.9f)/0.1f;
            state.offsetY = lerp(20,0,land);
            state.scale.y = lerp(1.3f,1,CosineInterpolation(land));
            state.rotationY = 0;
        }
        break;

    case DANCE_ROCK:
        {
            const float compress = sin(timeInStateNrm*c_PI);
            state.rotationY = lerp(0,c_PI*0.5f,compress)*((state.stateRepeats & 1)?1:-1);
            state.scale.y = lerp(1,0.5,compress*compress);
            state.scale.xz = lerp(1,1.3,compress*compress);
        }
        break;

    case DANCE_TWIST:
        state.twist = sin(timeInStateNrm*c_PI*4);
        state.scale.y = 1-IntroOutro(smoothstep(0,1,timeInStateNrm),0.25f)*0.75f;
        state.scale.xz = lerp(0.3f/state.scale.y,1,abs(timeInStateNrm-0.5f)); // Volume preservation
        {
            const float blendToOriginal = IntroOutro(timeInStateNrm,0.15f);
            state.twist *= blendToOriginal;
            state.scale = lerp(float3(1,1,1),state.scale,blendToOriginal);
            state.rotationY *= blendToOriginal;
            state.offsetY *= blendToOriginal;
        }
        break;
    }

}

[numthreads(32, 1, 1)]
void CSMainObjects(uint2 dispatchThreadId : SV_DispatchThreadID)
{
    const uint objectIndex = dispatchThreadId.y*0xFFFF*32 + dispatchThreadId.x;
    const uint rnd = Random(objectIndex);

    AnimState state;
    if (g_ResetState)
    {
        state = (AnimState)0;
        state.scale = float3(1,1,1);
    }
    else state = u_AnimStateData[objectIndex];

    const uint animType = t_InstanceData[objectIndex].animType;
    if (animType == AT_RotateY)
        state.rotationY += g_TimeDiff * lerp(0.4f, 1.0f, NormalizeRandom(rnd)) * ((rnd & 1) ? -1 : 1);
    else if (animType == AT_Dance)
        StepDance(state, rnd);

    u_AnimStateData[objectIndex] = state;
}

[numthreads(32, 1, 1)]
void CSMainLights(uint2 dispatchThreadId : SV_DispatchThreadID)
{
    const uint lightIndex = dispatchThreadId.y*0xFFFF*32 + dispatchThreadId.x;
    uint rnd = Random(lightIndex);

    Light light = u_LightData[lightIndex];
    if (g_ResetState)
        light.targetOffset = float3(0,0,0);
    else
    {
        const float radius = lerp(150,300,NormalizeRandom(rnd));
        rnd = Random(rnd);
        const float speed = lerp(1,3,NormalizeRandom(rnd));
        float2 sinCos;
        sincos(g_Time*speed,sinCos.x,sinCos.y);
        light.targetOffset = float3(sinCos.y*radius,0,sinCos.x*radius);
    }

    u_LightData[lightIndex] = light;
}