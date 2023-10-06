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

Buffer<uint> t_Input : register(t0);
RWBuffer<uint> u_Output : register(u0);

static const uint GroupSize = 256;

groupshared uint s_ReductionData[GroupSize/2];

[numthreads(GroupSize, 1, 1)]
void main(uint threadIdx : SV_GroupThreadID)
{
    uint data = t_Input[threadIdx];
    
    // Simple parallel reduction implementation.
    // Process the data using groups of threads of reducing size.
    // Note: there is a faster way of doing group-wide reduction using wave intrinsics.
    for (uint size = GroupSize/2; size >= 1; size >>= 1)
    {
        // Upper half of the current group stores its data into the shared buffer
        if (size <= threadIdx && threadIdx < size * 2)
            s_ReductionData[threadIdx - size] = data;

        GroupMemoryBarrierWithGroupSync();

        // Lower half of the current group loads the data from the shared buffer and adds it to the accumulator
        if (threadIdx < size)
            data += s_ReductionData[threadIdx];

        GroupMemoryBarrierWithGroupSync();

        // Repeat with a smaller group...
    }

    if (threadIdx == 0)
        u_Output[0] = data;
}