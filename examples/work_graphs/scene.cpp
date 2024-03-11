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

#include <donut/app/ApplicationBase.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/app/DeviceManager.h>
#include <donut/core/log.h>
#include <donut/core/vfs/VFS.h>
#include <donut/core/math/math.h>
#include <nvrhi/utils.h>
#include "scene.h"

using namespace donut::math;

// Below is a list of constants that can be used to control scene generation.

// Scene size and population (control scene dimensions, number of objects, lights and materials).
static const uint32_t SceneParam_MaterialCountOfEachType = 10;
static const uint32_t SceneParam_Floors = 3;
static const float SceneParam_FloorToCeilingHeight = 70;
static const float SceneParam_FloorSize = 500; // Larger means more objects and lights.
static const float SceneParam_ObjectRoomSize = 50;
static const float SceneParam_BallRoomSize = 120;
static const float SceneParam_BallSize = 15;
static const uint32_t SceneParam_LightsPerBall = 3; // Shaders can handle a max number of lights per tile. That must be adjusted according to this value too.

// Mesh density (control vertex processing cost).
static const uint16_t SceneParam_BoxSubdivisions = 100;
static const uint16_t SceneParam_SphereSides = 100;
static const uint16_t SceneParam_SphereSlices = 50;

// Materials visual look.
static const float3 SceneParam_GroundColor = {0.5f,0.5f,0.5f};
static const float SceneParam_PhongSpecularColorScale = 0.05f;
static const float SceneParam_PhongSpecularPowerMin = 15.0f;
static const float SceneParam_PhongSpecularPowerRange = 25.0f;
static const float SceneParam_VelvetRoughnessMin = 0.45f;
static const float SceneParam_VelvetRoughnessRange = 0.1f;
static const float SceneParam_FlakesSpecularColorScale = 0.05f;
static const float SceneParam_FlakesSpecularPowerMin = 15.0f;
static const float SceneParam_FlakesSpecularPowerRange = 25.0f;
static const float SceneParam_FlakesGranularityMin = 0.3f;
static const float SceneParam_FlakesGranularityRange = 0.1f;
static const float SceneParam_StanLineThicknessMin = 0.2f;
static const float SceneParam_StanLineThicknessRange = 0.4f;
static const float SceneParam_StanLineSpacingMin = 1.0f;
static const float SceneParam_StanLineSpacingRange = 3.0f;
static const float SceneParam_CheckersSize = 4.0f;
static const float SceneParam_CheckersSpecularPowerMin = 15.0f;
static const float SceneParam_CheckersSpecularPowerRange = 25.0f;

// All math and coordinate systems are left-handed.
struct MESH_DATA
{
	std::vector<float3> positions,normals;
	std::vector<uint16_t> indices;
};
static float3 RandomPosXZ(float extentsX,float y,float extentsZ);
static float3 RandomSize(float height,float size,float heightVariation,float sizeVariation);
static float3 RandomColor(bool normalized);
static float Random01();
static float RandomAngle();
static void GeneratePlane(MESH_DATA& outMesh);
static void GenerateBox(uint16_t faceSubdivisions,MESH_DATA& outMesh);
static void GenerateSphere(uint16_t sides,uint16_t slices,MESH_DATA& outMesh);

void Scene::CreateAssets(nvrhi::IDevice *device,nvrhi::ICommandList *commandList)
{
	// Generate geometry data.
	MESH_DATA meshSet[(int)MeshType::MT_COUNT];
	GeneratePlane(meshSet[(int)MeshType::MT_Plane]);
	GenerateBox(SceneParam_BoxSubdivisions, meshSet[(int)MeshType::MT_Box]);
	GenerateSphere(SceneParam_SphereSides, SceneParam_SphereSlices, meshSet[(int)MeshType::MT_Sphere]);

	PopulateWorld();

	// Create GPU buffers and record upload data commands.
	for (int i=0;i<(int)MeshType::MT_COUNT;i++)
	{
		const MESH_DATA& mesh = meshSet[i];

		// Generate vertex buffer data by interleaving positions and normals
		std::vector<float3> vertices(mesh.positions.size()*2);
		for (size_t j=0;j<mesh.positions.size();j++)
		{
			vertices[j*2+0] = mesh.positions[j];
			vertices[j*2+1] = mesh.normals[j];
		}

		// Interleave position and normal information in the vertex buffer
		const uint64_t vertexBufferSize = (mesh.positions.size()+mesh.normals.size()) * sizeof(float3);

		m_vertexBuffers[i] = device->createBuffer(
			nvrhi::BufferDesc().setByteSize(vertexBufferSize).
			setIsVertexBuffer(true).
			setInitialState(nvrhi::ResourceStates::VertexBuffer).
			setKeepInitialState(true).
			setDebugName("MeshVB"));

		commandList->writeBuffer(m_vertexBuffers[i], &vertices[0], vertexBufferSize);


		// Index buffer, 16-bit indices
		const uint64_t indexBufferSize = mesh.indices.size() * sizeof(uint16_t);

		m_indexBuffers[i] = device->createBuffer(
			nvrhi::BufferDesc().setByteSize(indexBufferSize).
			setIsIndexBuffer(true).
			setInitialState(nvrhi::ResourceStates::IndexBuffer).
			setKeepInitialState(true).
			setDebugName("MeshIB"));

		commandList->writeBuffer(m_indexBuffers[i], &mesh.indices[0], indexBufferSize);
	}

	// Materials data.
	{
		const uint64_t materialDataBufferSize = m_materials.size() * sizeof(Material);

		m_materialDataBuffer = device->createBuffer(
			nvrhi::BufferDesc().setByteSize(materialDataBufferSize).
			setCanHaveTypedViews(true).
			setStructStride(sizeof(Material)).
			setInitialState(nvrhi::ResourceStates::ShaderResource).
			setKeepInitialState(true).
			setDebugName("MaterialsData"));

		commandList->writeBuffer(m_materialDataBuffer, &m_materials[0], materialDataBufferSize);
	}

	// Instances data.
	{
		const uint64_t instanceDataBufferSize = m_worldObjects.size() * sizeof(Instance);

		m_instanceDataBuffer = device->createBuffer(
			nvrhi::BufferDesc().setByteSize(instanceDataBufferSize).
			setCanHaveTypedViews(true).
			setStructStride(sizeof(Instance)).
			setInitialState(nvrhi::ResourceStates::ShaderResource).
			setKeepInitialState(true).
			setDebugName("InstancesData"));

		commandList->writeBuffer(m_instanceDataBuffer, &m_worldObjects[0], instanceDataBufferSize);
	}

	// Lights data.
	{
		const uint64_t lightDataBufferSize = m_lights.size() * sizeof(Light);

		m_lightDataBuffer = device->createBuffer(
			nvrhi::BufferDesc().setByteSize(lightDataBufferSize).
			setCanHaveUAVs(true).
			setCanHaveTypedViews(true).
			setStructStride(sizeof(Light)).
			setInitialState(nvrhi::ResourceStates::UnorderedAccess).
			setKeepInitialState(true).
			setDebugName("LightsData"));

		commandList->writeBuffer(m_lightDataBuffer, &m_lights[0], lightDataBufferSize);
	}

	// Animation data.
	{
		const uint64_t animStateBufferSize = m_worldObjects.size() * sizeof(AnimState);

		m_animStateBuffer = device->createBuffer(
			nvrhi::BufferDesc().setByteSize(animStateBufferSize).
			setCanHaveUAVs(true).
			setCanHaveTypedViews(true).
			setStructStride(sizeof(AnimState)).
			setInitialState(nvrhi::ResourceStates::UnorderedAccess).
			setKeepInitialState(true).
			setDebugName("AnimState"));
	}
}

float Scene::GetSceneSize()
{
	return SceneParam_FloorSize;
}

float Scene::GetSceneHeight()
{
	return SceneParam_FloorToCeilingHeight * SceneParam_Floors;
}

void Scene::PopulateWorld()
{
	srand(0); // Determinstic world content.

	// Generate materials.
	m_materials.push_back(Material { SceneParam_GroundColor, MaterialType::BT_Lambert }); // Material 0 is lambert.
	m_materials.push_back(Material { {1,1,1}, MaterialType::BT_Faceted }); // Material 1 is faceted.
	const float materialCountOfEachType = 10;
	{
		// Lamberts.
		for (uint32_t i=0;i<SceneParam_MaterialCountOfEachType;i++)
			m_materials.push_back(Material { RandomColor(true), MaterialType::BT_Lambert });

		// Phongs.
		for (uint32_t i=0;i<SceneParam_MaterialCountOfEachType;i++)
		{
			Material mat = { RandomColor(true), MaterialType::BT_Phong };
			mat.phong.specularColor = RandomColor(true);
			mat.phong.specularColor = mat.phong.specularColor * SceneParam_PhongSpecularColorScale;
			mat.phong.specularPower = Random01()  *SceneParam_PhongSpecularPowerRange + SceneParam_PhongSpecularPowerMin;
			m_materials.push_back(mat);
		}

		// Metallics.
		for (uint32_t i=0;i<SceneParam_MaterialCountOfEachType;i++)
			m_materials.push_back(Material { RandomColor(true), MaterialType::BT_Metallic });

		// Velvets.
		for (uint32_t i=0;i<SceneParam_MaterialCountOfEachType;i++)
		{
			Material mat = { RandomColor(true), MaterialType::BT_Velvet };
			mat.velvet.roughness = Random01() * SceneParam_VelvetRoughnessRange + SceneParam_VelvetRoughnessMin;
			m_materials.push_back(mat);
		}

		// Flakes.
		for (uint32_t i=0;i<SceneParam_MaterialCountOfEachType;i++)
		{
			Material mat = { RandomColor(true), MaterialType::BT_Flakes };
			mat.flakes.specularColor = RandomColor(true);
			mat.phong.specularColor = mat.phong.specularColor * SceneParam_FlakesSpecularColorScale;
			mat.flakes.specularPower = Random01() * SceneParam_FlakesSpecularPowerRange + SceneParam_FlakesSpecularPowerMin;
			mat.flakes.granularity = Random01() * SceneParam_FlakesGranularityRange + SceneParam_FlakesGranularityMin;
			m_materials.push_back(mat);
		}

		// Stans.
		for (uint32_t i=0;i<SceneParam_MaterialCountOfEachType;i++)
		{
			Material mat = { RandomColor(true), MaterialType::BT_Stan };
			mat.stan.linesColor = RandomColor(false);
			mat.stan.linesThickness = Random01() * SceneParam_StanLineThicknessRange + SceneParam_StanLineThicknessMin;
			mat.stan.linesSpacing = Random01() * SceneParam_StanLineSpacingRange + SceneParam_StanLineSpacingMin;
			m_materials.push_back(mat);
		}

		// Checkers.
		for (uint32_t i=0;i<SceneParam_MaterialCountOfEachType;i++)
		{
			Material mat = { RandomColor(true), MaterialType::BT_Checker };
			mat.curvature.baseColor2 = RandomColor(false);
			mat.curvature.checkerSize = SceneParam_CheckersSize;
			mat.curvature.specularPower = Random01() * SceneParam_CheckersSpecularPowerRange + SceneParam_CheckersSpecularPowerMin;
			m_materials.push_back(mat);
		}
	} // Materials

	// Spawn multiple floors, each floor has a single plane, multiple glitter balls, and many cute dancers.
	for (uint32_t floor=0;floor<SceneParam_Floors;floor++)
	{
		const float floorHeight = floor * SceneParam_FloorToCeilingHeight;
		const float ceilingHeight = (floor+1) * SceneParam_FloorToCeilingHeight;

		// Ground.
		m_worldObjects.push_back(Instance {{0,floorHeight,0}, 0, {SceneParam_FloorSize,0,SceneParam_FloorSize}, MeshType::MT_Plane, 0, AnimType::AT_Static });

		// Multiple balls hung from the ceiling, emitting lights.
		{
			const int roomCount1D = (int)(SceneParam_FloorSize / SceneParam_BallRoomSize);
			const float ballHeight = ceilingHeight-SceneParam_BallSize*0.5f;
			for (int roomX=0;roomX<roomCount1D;roomX++)
			for (int roomZ=0;roomZ<roomCount1D;roomZ++)
			{
				const float roomCenterX = -SceneParam_FloorSize*0.5f + roomX * SceneParam_BallRoomSize + SceneParam_BallRoomSize*0.5f;
				const float roomCenterZ = -SceneParam_FloorSize*0.5f + roomZ * SceneParam_BallRoomSize + SceneParam_BallRoomSize*0.5f;
				float3 ballPos = RandomPosXZ((SceneParam_BallRoomSize-SceneParam_BallSize)*0.3f,ballHeight,(SceneParam_BallRoomSize-SceneParam_BallSize)*0.3f);
				ballPos.x += roomCenterX;
				ballPos.z += roomCenterZ;
				m_worldObjects.push_back(Instance {ballPos, RandomAngle(), {SceneParam_BallSize,SceneParam_BallSize,SceneParam_BallSize}, MeshType::MT_Sphere, 1, AnimType::AT_RotateY });

				// From each ball, generate a few lights.
				for (uint32_t light=0;light<SceneParam_LightsPerBall;light++)
				{
					const float3 dir = normalize(RandomSize(-1,0,0.8f,2.0f));
					const float length = Random01() * SceneParam_FloorSize*0.35f + SceneParam_FloorToCeilingHeight;
					const float3 tgt = {dir.x*length+ballPos.x, dir.y*length+ballPos.y, dir.z*length+ballPos.z};
					float angle1 = RandomAngle()*0.25f+0.25f; // Within 90-degree limit.
					float angle2 = RandomAngle()*0.25f+0.25f; // Within 90-degree limit.
					const float innerAngle = min(angle1,angle2);
					const float outerAngle = max(angle1,angle2)+RandomAngle()*0.1f;

					m_lights.push_back(Light {ballPos, tgt, float3(0,0,0), RandomColor(true), innerAngle, outerAngle});
				}
			}
		}

		// Many objects on the floor, sub-divide the plane into squares and place one object randomly within that square.
		{
			const int roomCount1D = (int)(SceneParam_FloorSize / SceneParam_ObjectRoomSize);
			for (int roomX=0;roomX<roomCount1D;roomX++)
			for (int roomZ=0;roomZ<roomCount1D;roomZ++)
			{
				const float roomCenterX = -SceneParam_FloorSize*0.5f + roomX * SceneParam_ObjectRoomSize + SceneParam_ObjectRoomSize*0.5f;
				const float roomCenterZ = -SceneParam_FloorSize*0.5f + roomZ * SceneParam_ObjectRoomSize + SceneParam_ObjectRoomSize*0.5f;
			
				float3 size = RandomSize(SceneParam_FloorToCeilingHeight*0.35f,SceneParam_ObjectRoomSize*0.20f,SceneParam_FloorToCeilingHeight*0.1f,SceneParam_ObjectRoomSize*0.05f);
				float3 pos = RandomPosXZ((SceneParam_ObjectRoomSize-size.x)*0.5f,floorHeight+size.y*0.5f,(SceneParam_ObjectRoomSize-size.z)*0.5f);
				pos.x += roomCenterX;
				pos.y += 0.01f; // Counter z-fighting.
				pos.z += roomCenterZ;
				const UINT32 material = (UINT32)(rand()%((int)m_materials.size()-2)+2); // Skip the first two hard-coded materials.

				m_worldObjects.push_back(Instance {pos, RandomAngle(), size, MeshType::MT_Box, material, AnimType::AT_Dance });
			}
		}
	}
}

#pragma region Randomization Functions
static float3 RandomPosXZ(float extentsX,float y,float extentsZ)
{
	return float3
	{
		((rand()/(float)RAND_MAX)-0.5f)*extentsX*2.0f,
		y,
		((rand()/(float)RAND_MAX)-0.5f)*extentsZ*2.0f
	};
};

static float3 RandomSize(float height,float size,float heightVariation,float sizeVariation)
{
	return float3
	{
		size + (rand()/(float)RAND_MAX-0.5f)*sizeVariation,
		height + (rand()/(float)RAND_MAX-0.5f)*heightVariation,
		size + (rand()/(float)RAND_MAX-0.5f)*sizeVariation,
	};
}

static float3 RandomColor(bool normalized)
{
	float3 clr = float3
	{
		rand()/(float)RAND_MAX,
		rand()/(float)RAND_MAX,
		rand()/(float)RAND_MAX
	};
	return normalized ? normalize(clr) : clr;
}

static float Random01()
{
	return rand()/(float)RAND_MAX;
}

static float RandomAngle()
{
	return (rand()/(float)RAND_MAX)*PI_f*2.0f;
};
#pragma endregion

#pragma region Geometry generation
static void GeneratePlaneInternal(float y,float sign,MESH_DATA& outMesh)
{
	const uint16_t baseVtx = (uint16_t)outMesh.positions.size();
	const float3 pos[] = { {-0.5f*sign,y,-0.5f},{-0.5f*sign,y,0.5f},{0.5f*sign,y,0.5f},{0.5f*sign,y,-0.5f} };
	const float3 nrm = {0,sign,0};
	const uint16_t indices[] = { uint16_t(baseVtx+0),uint16_t(baseVtx+1),uint16_t(baseVtx+2), uint16_t(baseVtx+2),uint16_t(baseVtx+3),uint16_t(baseVtx+0) };

	outMesh.positions.insert(outMesh.positions.end(),pos,pos+4);
	outMesh.normals.insert(outMesh.normals.end(),4,nrm);
	outMesh.indices.insert(outMesh.indices.end(),indices,indices+6);
}

static void GeneratePlane(MESH_DATA& outMesh)
{
	GeneratePlaneInternal(0, 1,outMesh);
	GeneratePlaneInternal(0,-1,outMesh);
}

static void GenerateBox(uint16_t faceSubdivisions,MESH_DATA& outMesh)
{
	auto GenerateSide = [&](int coord0,int coord1,const float3 posInit,const float3 nrm,float sign)
	{
		assert(outMesh.positions.size() < 0xFFFF);
		const uint16_t baseVtx = (uint16_t)outMesh.positions.size();
		float pos[] = {posInit.x,posInit.y,posInit.z};
		for (uint16_t y=0;y<faceSubdivisions+1;y++)
		{
			pos[coord1] = (y/(float)faceSubdivisions-0.5f);
			for (uint16_t x=0;x<faceSubdivisions+1;x++)
			{
				pos[coord0] = (x/(float)faceSubdivisions-0.5f)*sign;
				outMesh.positions.push_back({pos[0],pos[1],pos[2]});
				outMesh.normals.push_back(nrm);
			}
		}

		for (uint16_t y=0;y<faceSubdivisions;y++)
		for (uint16_t x=0;x<faceSubdivisions;x++)
		{
			const uint16_t faceBaseVtx = baseVtx+y*(faceSubdivisions+1)+x;
			outMesh.indices.push_back(faceBaseVtx+0);
			outMesh.indices.push_back(faceBaseVtx+(faceSubdivisions+1)+0);
			outMesh.indices.push_back(faceBaseVtx+(faceSubdivisions+1)+1);

			outMesh.indices.push_back(faceBaseVtx+(faceSubdivisions+1)+1);
			outMesh.indices.push_back(faceBaseVtx+1);
			outMesh.indices.push_back(faceBaseVtx+0);
		}
	};

	outMesh.positions.reserve(outMesh.positions.capacity()+(faceSubdivisions+1)*(faceSubdivisions+1)*4+8);
	outMesh.normals.reserve(outMesh.normals.capacity()+outMesh.positions.capacity());
	outMesh.indices.reserve(outMesh.indices.capacity()+((faceSubdivisions*faceSubdivisions)*4+4)*6);

	GenerateSide(0,1,{0,0,-0.5f},{0,0,-1}, 1); // Front side.
	GenerateSide(2,1,{ 0.5f,0,0},{ 1,0,0}, 1); // Right side.
	GenerateSide(0,1,{0,0, 0.5f},{0,0, 1},-1); // Back side.
	GenerateSide(2,1,{-0.5f,0,0},{-1,0,0},-1); // Left side.
	GeneratePlaneInternal(0.5f,1,outMesh); // Top side.
	GeneratePlaneInternal(-0.5f,-1,outMesh); // Bottom side.
}

static void GenerateSphere(uint16_t sides,uint16_t slices,MESH_DATA& outMesh)
{
	assert(sides >= 3);
	assert(slices >= 2);
	assert(outMesh.positions.size() < 0xFFFF);
	const uint16_t baseVtx = (uint16_t)outMesh.positions.size();

	outMesh.positions.push_back({0,-0.5f,0}); // Bottom vertex.
	outMesh.normals.push_back({0,-1,0}); // Bottom vertex.
	for (uint16_t y=1;y<slices;y++) // Trunk vertices.
	{
		float3 pos = {0,y/(float)slices-0.5f,0};
		const float ringRadius = sqrtf(1-pos.y*pos.y*4.0f)*0.5f;
		for (uint16_t x=0;x<sides;x++)
		{
			const float angle = (x/(float)sides) * PI_f*2.0f;
			pos.x = cosf(angle)*ringRadius;
			pos.z = sinf(angle)*ringRadius;
			outMesh.positions.push_back(pos);
			outMesh.normals.push_back(normalize(pos));
		}
	}
	const uint16_t capVtx = (uint16_t)outMesh.positions.size();
	outMesh.positions.push_back({0,0.5f,0}); // Top vertex.
	outMesh.normals.push_back({0,1,0}); // Bottom vertex.

	// Bottom cap.
	for (uint16_t i=0;i<sides;i++)
	{
		outMesh.indices.push_back(baseVtx+0);
		outMesh.indices.push_back(baseVtx+1+i);
		outMesh.indices.push_back(baseVtx+1+(i+1)%sides);
	}

	// Trunk.
	for (uint16_t y=0;y<slices-2;y++)
	{
		const uint16_t sliceBaseVtx = baseVtx+1+y*sides;
		for (uint16_t x=0;x<sides;x++)
		{
			outMesh.indices.push_back(sliceBaseVtx+x+0);
			outMesh.indices.push_back(sliceBaseVtx+x+0+sides);
			outMesh.indices.push_back(sliceBaseVtx+(x+1)%sides+sides);

			outMesh.indices.push_back(sliceBaseVtx+(x+1)%sides+sides);
			outMesh.indices.push_back(sliceBaseVtx+(x+1)%sides);
			outMesh.indices.push_back(sliceBaseVtx+x+0);
		}
	}

	// Top cap.
	const uint16_t capBaseVtx = baseVtx+1+(slices-2)*sides;
	for (uint16_t i=0;i<sides;i++)
	{
		outMesh.indices.push_back(capBaseVtx+i);
		outMesh.indices.push_back(capVtx);
		outMesh.indices.push_back(capBaseVtx+(i+1)%sides);
	}
}
#pragma endregion