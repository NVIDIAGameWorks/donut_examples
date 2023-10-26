/*
* Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

// This file implements the Multi-Layer Alpha Blending (MLAB) algorithm.
// See M. Salvi, K. Vaidyanathan: "Multi-Layer Alpha Blending".

#ifndef MLAB_FRAGMENTS
#define MLAB_FRAGMENTS 4
#endif

struct BlendFragment
{
    float3 color;
    float attenuation;
    float depth;
};

// Initializes the blending array with empty fragments.
void blendInit(inout BlendFragment buffer[MLAB_FRAGMENTS])
{
    BlendFragment f;
    f.color = 0;
    f.attenuation = 1;
    f.depth = 1.#INF;

    for (int i = 0; i < MLAB_FRAGMENTS; ++i)
    {
        buffer[i] = f;
    }
}

// Inserts the new fragment f into the blending array.
// Based on Listing 1 in the paper, with one bug fix.
void blendInsert(BlendFragment f, inout BlendFragment buffer[MLAB_FRAGMENTS])
{
    // 1-pass bubble sort to insert fragment
    BlendFragment temp;
    for (int i = 0; i < MLAB_FRAGMENTS; ++i)
    {
        if (f.depth < buffer[i].depth)
        {
            temp = buffer[i];
            buffer[i] = f;
            f = temp;
        }
    }

    // Compression (merge last two rows)
    BlendFragment last = buffer[MLAB_FRAGMENTS-1];
    BlendFragment merged;
    merged.color = last.color + f.color * last.attenuation;
    merged.attenuation = last.attenuation * f.attenuation;
    merged.depth = last.depth;
    buffer[MLAB_FRAGMENTS-1] = merged;
}

// Integrates all fragments in the blending array into one fragment.
BlendFragment blendIntegrate(BlendFragment buffer[MLAB_FRAGMENTS])
{
    BlendFragment result = buffer[0];

    for (int i = 1; i < MLAB_FRAGMENTS; ++i)
    {
        BlendFragment f = buffer[i];
        result.color += f.color * result.attenuation;
        result.attenuation *= f.attenuation;
    }

    return result;
}
