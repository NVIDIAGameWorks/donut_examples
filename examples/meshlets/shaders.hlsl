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

#define MAX_PRIMS_PER_MESHLET 1
#define MAX_VERTICES_PER_MESHLET 3

static const float2 g_positions[] = {
	float2(-0.5, -0.5),
	float2(0, 0.5),
	float2(0.5, -0.5)
};

static const float3 g_colors[] = {
	float3(1, 0, 0),
	float3(0, 1, 0),
	float3(0, 0, 1)	
};

struct Payload
{
	int dummy;
};

struct Vertex
{
	float4 pos : SV_Position;
	float3 color : COLOR;
};

groupshared Payload s_payload;

[numthreads(1, 1, 1)]
void main_as(
	uint globalIdx : SV_DispatchThreadID)
{
	s_payload.dummy = 0;
	DispatchMesh(1, 1, 1, s_payload);
}

[numthreads(1,1,1)]
[outputtopology("triangle")]
void main_ms(
    uint threadId : SV_GroupThreadID,
    in payload Payload i_payload,
    out indices uint3 o_tris[MAX_PRIMS_PER_MESHLET],
    out vertices Vertex o_verts[MAX_VERTICES_PER_MESHLET])
{
    SetMeshOutputCounts(3, 1);
    o_tris[0] = uint3(0, 1, 2);

    for (uint i = 0; i < 3; i++)
    {
    	o_verts[i].pos = float4(g_positions[i], 0, 1);
    	o_verts[i].color = g_colors[i];
    }
}


void main_ps(
	in Vertex i_vertex,
	out float4 o_color : SV_Target0
)
{
	o_color = float4(i_vertex.color, 1);
}
