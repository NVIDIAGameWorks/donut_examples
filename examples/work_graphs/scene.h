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

#pragma once

class Scene
{
public:

	enum class AnimType : uint32_t
	{
		AT_Static,
		AT_RotateY,
		AT_Dance,
		AT_COUNT
	};

	enum class MeshType : uint32_t 
	{
		MT_Plane,
		MT_Box,
		MT_Sphere,
		MT_COUNT
	};

	enum class MaterialType : uint32_t
	{
		BT_Lambert,
		BT_Phong,
		BT_Metallic,
		BT_Velvet,
		BT_Flakes,
		BT_Faceted,
		BT_Stan,
		BT_Checker,
		BT_COUNT
	};

	struct Material
	{
		struct PhongParams
		{
			dm::float3 specularColor;
			float specularPower;
		};
		struct VelvetParams
		{
			float roughness;
		};
		struct FlakesParams
		{
			dm::float3 specularColor;
			float specularPower;
			float granularity;
		};
		struct StanParams
		{
			dm::float3 linesColor;
			float linesThickness;
			float linesSpacing;
		};
		struct CheckerParams
		{
			dm::float3 baseColor2;
			float checkerSize;
			float specularPower;
		};

		dm::float3 baseColor;
		MaterialType materialType;
		union
		{
			PhongParams phong;
			VelvetParams velvet;
			FlakesParams flakes;
			StanParams stan;
			CheckerParams curvature;
		};
	};

	struct Instance
	{
		dm::float3 position;
		float rotationY;
		dm::float3 size;
		MeshType meshType;
		uint32_t material;
		AnimType animType;
	};

	struct Light
	{
		dm::float3 position;
		dm::float3 target;
		dm::float3 targetOffset;
		dm::float3 color;
		float innerAngle;
		float outerAngle;
	};

	struct AnimState
	{
		uint32_t state;
		uint32_t stateRepeats;
		float statePeriod;
		float timeInState;

		dm::float3 scale;
		float rotationY;
		float offsetY;
		float twist;
	};

	void CreateAssets(nvrhi::IDevice *device,nvrhi::ICommandList *commandList);

	const std::vector<Material>& GetMaterials() const { return m_materials; }
	const std::vector<Instance>& GetWorldObjects() const { return m_worldObjects; }
	const std::vector<Light>& GetLights() const { return m_lights; }
	nvrhi::BufferHandle GetMaterialsBuffer() const { return m_materialDataBuffer; }
	nvrhi::BufferHandle GetWorldObjectsBuffer() const { return m_instanceDataBuffer; }
	nvrhi::BufferHandle GetLightsBuffer() const { return m_lightDataBuffer; }
	nvrhi::BufferHandle GetAnimStateBuffer() const { return m_animStateBuffer; }
	nvrhi::BufferHandle GetMeshVertexBuffer(MeshType meshType) const { return m_vertexBuffers[(int)meshType]; }
	nvrhi::BufferHandle GetMeshIndexBuffer(MeshType meshType) const { return m_indexBuffers[(int)meshType]; }

	static float GetSceneSize();
	static float GetSceneHeight();

protected:
	void PopulateWorld();

	nvrhi::BufferHandle m_vertexBuffers[(int)MeshType::MT_COUNT];
	nvrhi::BufferHandle m_indexBuffers[(int)MeshType::MT_COUNT];
	nvrhi::BufferHandle m_materialDataBuffer;
	nvrhi::BufferHandle m_instanceDataBuffer;
	nvrhi::BufferHandle m_lightDataBuffer;
	nvrhi::BufferHandle m_animStateBuffer;

	std::vector<Material> m_materials;
	std::vector<Instance> m_worldObjects;
	std::vector<Light> m_lights;
};