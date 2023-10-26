/*
* Copyright (c) 2014-2022, NVIDIA CORPORATION. All rights reserved.
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
#include <donut/app/Camera.h>
#include <donut/app/imgui_renderer.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/engine/TextureCache.h>
#include <donut/engine/Scene.h>
#include <donut/engine/BindingCache.h>
#include <donut/engine/View.h>
#include <donut/app/DeviceManager.h>
#include <donut/core/log.h>
#include <donut/core/vfs/VFS.h>
#include <donut/core/math/math.h>
#include <nvrhi/utils.h>


using namespace donut;
using namespace donut::math;

#include "rt_particles_cb.h"

static const char* g_WindowTitle = "Donut Example: Ray Traced Particles";

constexpr uint32_t c_MaxParticles = 1024;
constexpr uint32_t c_IndicesPerQuad = 6;
constexpr uint32_t c_VerticesPerQuad = 4;

static float random()
{
    return float(std::rand()) / RAND_MAX;
}

static float3 random3()
{
    return float3(random(), random(), random());
}

struct ParticleEntity
{
    bool active = false;
    float3 position = 0.f;
    float3 velocity = 0.f;
    float3 color = 1.f;
    float radius = 0.f;
    float age = 0.f;
    float opacity = 1.f;
    float rotation = 0.f;
    
    void Emit(float3 emitterPosition)
    {
        active = true;
        position = emitterPosition;
        velocity = random3() - 0.5f;
        velocity.y = 1.f + velocity.y;
        radius = random() * 0.05f + 0.1f;
        age = 0.f;
        opacity = 1.f;
        color = random3() * 0.5f + 0.1f;
        rotation = random() * 6.28f;
    }

    void Animate(float time)
    {
        const float lifeTime = 2.f;

        position += velocity * time;
        velocity.y += 1.f * time;
        velocity.x += 1.f * time;
        age += time;
        radius += 0.5f * time;
        opacity = saturate((lifeTime - age) * 0.5f);

        if (age > lifeTime)
            active = false;
    }
};

enum class ParticleTexture
{
    Smoke = 0,
    Logo
};

struct UIData
{
    bool updatePipeline = true;
    bool enableAnimations = true;
    bool alwaysUpdateOrientation = true;
    bool reorientParticlesInPrimaryRays = false;
    bool reorientParticlesInSecondaryRays = true;
    uint orientationMode = ORIENTATION_MODE_QUATERNION;
    uint mlabFragments = 4;
    ParticleTexture particleTexture = ParticleTexture::Smoke;
    float3 emitterPosition = 0.f;
};

class RayTracedParticles : public app::ApplicationBase
{
private:
	std::shared_ptr<vfs::RootFileSystem> m_RootFS;

    nvrhi::ShaderHandle m_ComputeShader;
    nvrhi::ComputePipelineHandle m_ComputePipeline;
    nvrhi::CommandListHandle m_CommandList;
    nvrhi::BindingLayoutHandle m_BindingLayout;
    nvrhi::BindingSetHandle m_BindingSet;
    nvrhi::BindingLayoutHandle m_BindlessLayout;

    nvrhi::rt::AccelStructHandle m_TopLevelAS;

    nvrhi::BufferHandle m_ConstantBuffer;

    std::shared_ptr<engine::ShaderFactory> m_ShaderFactory;
    std::shared_ptr<engine::DescriptorTableManager> m_DescriptorTable;
    std::unique_ptr<engine::Scene> m_Scene;
    nvrhi::TextureHandle m_ColorBuffer;
    app::ThirdPersonCamera m_Camera;
    engine::PlanarView m_View;
    std::shared_ptr<engine::DirectionalLight> m_SunLight;
    std::unique_ptr<engine::BindingCache> m_BindingCache;
    
    std::shared_ptr<engine::BufferGroup> m_ParticleBuffers;
    std::shared_ptr<engine::MeshGeometry> m_ParticleGeometry;
    std::shared_ptr<engine::MeshInfo> m_ParticleMesh;
    std::shared_ptr<engine::MeshInstance> m_ParticleInstance;
    std::shared_ptr<engine::Material> m_ParticleMaterial;
    nvrhi::BufferHandle m_ParticleInfoBuffer;
    nvrhi::rt::AccelStructHandle m_ParticleIntersectionBLAS;
    
    std::vector<ParticleEntity> m_Particles;
    std::vector<ParticleInfo> m_ParticleInfoData;

    std::shared_ptr<engine::LoadedTexture> m_EnvironmentMap;
    std::shared_ptr<engine::LoadedTexture> m_SmokeTexture;
    std::shared_ptr<engine::LoadedTexture> m_LogoTexture;

    UIData* m_ui;
    float m_WallclockTime = 0.f;
    float m_LastEmitTime = 0.f;

public:
    RayTracedParticles(app::DeviceManager* deviceManager, UIData* ui)
        : ApplicationBase(deviceManager)
        , m_ui(ui)
    { }

    bool Init()
    {
        std::filesystem::path frameworkShaderPath = app::GetDirectoryWithExecutable() / "shaders/framework" / app::GetShaderTypeName(GetDevice()->getGraphicsAPI());
        std::filesystem::path appShaderPath = app::GetDirectoryWithExecutable() / "shaders/rt_particles" / app::GetShaderTypeName(GetDevice()->getGraphicsAPI());
        std::filesystem::path mediaPath = app::GetDirectoryWithExecutable().parent_path() / "media";

		m_RootFS = std::make_shared<vfs::RootFileSystem>();
		m_RootFS->mount("/shaders/donut", frameworkShaderPath);
        m_RootFS->mount("/shaders/app", appShaderPath);
        m_RootFS->mount("/media", mediaPath);

		m_ShaderFactory = std::make_shared<engine::ShaderFactory>(GetDevice(), m_RootFS, "/shaders");
		m_CommonPasses = std::make_shared<engine::CommonRenderPasses>(GetDevice(), m_ShaderFactory);
        m_BindingCache = std::make_unique<engine::BindingCache>(GetDevice());

        nvrhi::BindlessLayoutDesc bindlessLayoutDesc;
        bindlessLayoutDesc.visibility = nvrhi::ShaderType::All;
        bindlessLayoutDesc.firstSlot = 0;
        bindlessLayoutDesc.maxCapacity = 1024;
        bindlessLayoutDesc.registerSpaces = {
            nvrhi::BindingLayoutItem::RawBuffer_SRV(1),
            nvrhi::BindingLayoutItem::Texture_SRV(2)
        };
        m_BindlessLayout = GetDevice()->createBindlessLayout(bindlessLayoutDesc);

        nvrhi::BindingLayoutDesc globalBindingLayoutDesc;
        globalBindingLayoutDesc.visibility = nvrhi::ShaderType::All;
        globalBindingLayoutDesc.bindings = {
            nvrhi::BindingLayoutItem::VolatileConstantBuffer(0),
            nvrhi::BindingLayoutItem::RayTracingAccelStruct(0),
            nvrhi::BindingLayoutItem::StructuredBuffer_SRV(1),
            nvrhi::BindingLayoutItem::StructuredBuffer_SRV(2),
            nvrhi::BindingLayoutItem::StructuredBuffer_SRV(3),
            nvrhi::BindingLayoutItem::StructuredBuffer_SRV(4),
            nvrhi::BindingLayoutItem::Sampler(0),
            nvrhi::BindingLayoutItem::Texture_UAV(0)
        };
        m_BindingLayout = GetDevice()->createBindingLayout(globalBindingLayoutDesc);

        m_DescriptorTable = std::make_shared<engine::DescriptorTableManager>(GetDevice(), m_BindlessLayout);
        
        m_TextureCache = std::make_shared<engine::TextureCache>(GetDevice(), m_RootFS, m_DescriptorTable);

        m_CommandList = GetDevice()->createCommandList();
        
        CreateParticleMesh();
        m_Particles.resize(c_MaxParticles);
        m_ParticleInfoData.resize(c_MaxParticles);

        m_EnvironmentMap = m_TextureCache->LoadTextureFromFileDeferred("/media/rt_particles/environment-map.dds", false);
        m_SmokeTexture = m_TextureCache->LoadTextureFromFileDeferred("/media/rt_particles/smoke-particle.png", true);
        m_LogoTexture = m_TextureCache->LoadTextureFromFileDeferred("/media/nvidia-logo.png", true);

        SetAsynchronousLoadingEnabled(false);
        BeginLoadingScene(m_RootFS, "/media/rt_particles/ParticleScene.gltf");

        m_Scene->GetSceneGraph()->AttachLeafNode(m_Scene->GetSceneGraph()->GetRootNode(), m_ParticleInstance);
        m_Scene->RefreshSceneGraph(0);

        auto emitterNode = m_Scene->GetSceneGraph()->FindNode("/Emitter");
        if (emitterNode)
            m_ui->emitterPosition = emitterNode->GetLocalToWorldTransformFloat().m_translation;

        m_Scene->FinishedLoading(GetFrameIndex());

        m_Camera.SetTargetPosition(m_ui->emitterPosition + float3(0.f, 2.f, 0.f));
        m_Camera.SetDistance(6.f);
        m_Camera.SetRotation(radians(225.f), radians(20.f));
        m_Camera.SetMoveSpeed(3.f);

        m_ConstantBuffer = GetDevice()->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(
            sizeof(GlobalConstants), "LightingConstants", engine::c_MaxRenderPassConstantBufferVersions));
        
        m_CommandList->open();

        CreateAccelStructs(m_CommandList);
        BuildParticleIntersectionBLAS(m_CommandList);
                
        m_CommandList->close();
        GetDevice()->executeCommandList(m_CommandList);

        GetDevice()->waitForIdle();

        return true;
    }

    // Creates the buffers and initializes engine structures to attach a procedural particle mesh to the scene
    void CreateParticleMesh()
    {
        // BufferGroup
        m_ParticleBuffers = std::make_shared<engine::BufferGroup>();

        // Position and texcoord data ranges
        auto& positionRange = m_ParticleBuffers->getVertexBufferRange(engine::VertexAttribute::Position);
        auto& texcoordRange = m_ParticleBuffers->getVertexBufferRange(engine::VertexAttribute::TexCoord1);
        positionRange.byteOffset = 0;
        positionRange.byteSize = c_MaxParticles * c_VerticesPerQuad * sizeof(float3);
        texcoordRange.byteOffset = positionRange.byteOffset + positionRange.byteSize;
        texcoordRange.byteSize = c_MaxParticles * c_VerticesPerQuad * sizeof(float2);

        // Index buffer
        nvrhi::BufferDesc bufferDesc;
        bufferDesc.byteSize = c_MaxParticles * c_IndicesPerQuad * sizeof(uint32_t);
        bufferDesc.debugName = "ParticleIndices";
        bufferDesc.canHaveRawViews = true;
        bufferDesc.initialState = nvrhi::ResourceStates::ShaderResource | nvrhi::ResourceStates::AccelStructBuildInput;
        bufferDesc.keepInitialState = true;
        m_ParticleBuffers->indexBuffer = GetDevice()->createBuffer(bufferDesc);

        // Vertex buffer
        bufferDesc.byteSize = texcoordRange.byteOffset + texcoordRange.byteSize;
        bufferDesc.debugName = "ParticleVertices";
        m_ParticleBuffers->vertexBuffer = GetDevice()->createBuffer(bufferDesc);

        // Index and vertex buffer bindless descriptors
        m_ParticleBuffers->indexBufferDescriptor = std::make_shared<engine::DescriptorHandle>(
            m_DescriptorTable->CreateDescriptorHandle(nvrhi::BindingSetItem::RawBuffer_SRV(0, m_ParticleBuffers->indexBuffer)));

        m_ParticleBuffers->vertexBufferDescriptor = std::make_shared<engine::DescriptorHandle>(
            m_DescriptorTable->CreateDescriptorHandle(nvrhi::BindingSetItem::RawBuffer_SRV(0, m_ParticleBuffers->vertexBuffer)));

        // Material
        m_ParticleMaterial = std::make_shared<engine::Material>();
        m_ParticleMaterial->domain = engine::MaterialDomain::AlphaBlended;

        // Geometry
        m_ParticleGeometry = std::make_shared<engine::MeshGeometry>();
        m_ParticleGeometry->material = m_ParticleMaterial;

        // Set numVertices and numIndices to max possible to make sure that we create an appropriate BLAS before rendering
        m_ParticleGeometry->numVertices = c_MaxParticles * c_VerticesPerQuad;
        m_ParticleGeometry->numIndices = c_MaxParticles * c_IndicesPerQuad;
        m_ParticleBuffers->indexData.resize(m_ParticleGeometry->numIndices);
        m_ParticleBuffers->positionData.resize(m_ParticleGeometry->numVertices);
        m_ParticleBuffers->texcoord1Data.resize(m_ParticleGeometry->numVertices);

        // Mesh
        m_ParticleMesh = std::make_shared<engine::MeshInfo>();
        m_ParticleMesh->buffers = m_ParticleBuffers;
        m_ParticleMesh->geometries = { m_ParticleGeometry };
        m_ParticleMesh->name = "ParticleMesh";

        // Instance
        m_ParticleInstance = std::make_shared<engine::MeshInstance>(m_ParticleMesh);

        // Particle info buffer
        bufferDesc.byteSize = c_MaxParticles * sizeof(ParticleInfo);
        bufferDesc.canHaveRawViews = false;
        bufferDesc.structStride = sizeof(ParticleInfo);
        bufferDesc.debugName = "ParticleInfoBuffer";
        m_ParticleInfoBuffer = GetDevice()->createBuffer(bufferDesc);
    }

    // Updates particle geometry -- to be called before rendering every frame
    void BuildParticleGeometry(nvrhi::ICommandList* commandList)
    {
        commandList->beginMarker("Update Particles");
        
        // Get the camera plane vectors for particle orientation
        float3 cameraForward = m_Camera.GetDir();
        float3 cameraUp = m_Camera.GetUp();

        // To demonstrate beam orientation, we create vertical sprites that are free to rotate
        // around the world-space Y axis, simulating what old Doom-like games used.
        if (m_ui->orientationMode == ORIENTATION_MODE_BEAM)
        {
            if (fabsf(cameraForward.y) > 0.999f)
                cameraForward = cameraUp;

            cameraForward.y = 0.f;
            cameraForward = normalize(cameraForward);
            cameraUp = float3(0.f, 1.f, 0.f);
        }

        const float3 cameraRight = cross(cameraForward, cameraUp);

        // Generate the geometry for particles
        uint32_t numParticles = 0;
        for (uint32_t index = 0; index < m_Particles.size(); ++index)
        {
            const ParticleEntity* particle = &m_Particles[index];

            if (!particle->active)
                continue;

            uint32_t baseIndex = numParticles * c_IndicesPerQuad;
            uint32_t baseVertex = numParticles * c_VerticesPerQuad;

            // Indices for a quad
            m_ParticleBuffers->indexData[baseIndex + 0] = baseVertex + 0;
            m_ParticleBuffers->indexData[baseIndex + 1] = baseVertex + 1;
            m_ParticleBuffers->indexData[baseIndex + 2] = baseVertex + 2;
            m_ParticleBuffers->indexData[baseIndex + 3] = baseVertex + 0;
            m_ParticleBuffers->indexData[baseIndex + 4] = baseVertex + 2;
            m_ParticleBuffers->indexData[baseIndex + 5] = baseVertex + 3;

            // Compute the quad orientation in world space
            const float rotation = m_ui->orientationMode == ORIENTATION_MODE_BEAM ? 0.f : particle->rotation;
            const float2 localRight = float2(cosf(rotation), sinf(rotation));
            const float2 localUp    = float2(-localRight.y, localRight.x);
            const float3 worldRight = localRight.x * cameraRight + localRight.y * cameraUp;
            const float3 worldUp    = localUp.x    * cameraRight + localUp.y    * cameraUp;

            // Positions
            m_ParticleBuffers->positionData[baseVertex + 0] = particle->position - worldRight * particle->radius + worldUp * particle->radius ;
            m_ParticleBuffers->positionData[baseVertex + 1] = particle->position + worldRight * particle->radius + worldUp * particle->radius ;
            m_ParticleBuffers->positionData[baseVertex + 2] = particle->position + worldRight * particle->radius - worldUp * particle->radius ;
            m_ParticleBuffers->positionData[baseVertex + 3] = particle->position - worldRight * particle->radius - worldUp * particle->radius ;

            // Texture coordinates
            m_ParticleBuffers->texcoord1Data[baseVertex + 0] = float2(0.f, 0.f);
            m_ParticleBuffers->texcoord1Data[baseVertex + 1] = float2(1.f, 0.f);
            m_ParticleBuffers->texcoord1Data[baseVertex + 2] = float2(1.f, 1.f);
            m_ParticleBuffers->texcoord1Data[baseVertex + 3] = float2(0.f, 1.f);

            // Fill out the ParticleInfo structure for use in shaders, mostly in the intersection particle code path
            ParticleInfo* particleInfo = &m_ParticleInfoData[numParticles];
            particleInfo->center = particle->position;
            particleInfo->rotation = particle->rotation;
            particleInfo->colorFactor = particle->color;
            particleInfo->opacityFactor = particle->opacity;
            particleInfo->xAxis = worldRight;
            particleInfo->yAxis = worldUp;
            particleInfo->inverseRadius = 1.f / particle->radius;
            particleInfo->textureIndex = m_ParticleMaterial->baseOrDiffuseTexture->bindlessDescriptor.Get();

            ++numParticles;
        }

        m_ParticleGeometry->numIndices = numParticles * c_IndicesPerQuad;
        m_ParticleGeometry->numVertices = numParticles * c_VerticesPerQuad;

        // Copy the index and vertex data to the GPU
        commandList->writeBuffer(m_ParticleBuffers->indexBuffer, m_ParticleBuffers->indexData.data(), m_ParticleGeometry->numIndices * sizeof(uint32_t));
        commandList->writeBuffer(m_ParticleBuffers->vertexBuffer, m_ParticleBuffers->positionData.data(), m_ParticleGeometry->numVertices *  sizeof(float3),
            m_ParticleBuffers->getVertexBufferRange(engine::VertexAttribute::Position).byteOffset);
        commandList->writeBuffer(m_ParticleBuffers->vertexBuffer, m_ParticleBuffers->texcoord1Data.data(), m_ParticleGeometry->numVertices * sizeof(float2),
            m_ParticleBuffers->getVertexBufferRange(engine::VertexAttribute::TexCoord1).byteOffset);

        // Copy the particle info data to the GPU
        commandList->writeBuffer(m_ParticleInfoBuffer, m_ParticleInfoData.data(), numParticles * sizeof(ParticleInfo), 0);

        // Build the BLAS
        nvrhi::rt::AccelStructDesc blasDesc;
        GetMeshBlasDesc(*m_ParticleMesh, blasDesc);
        nvrhi::utils::BuildBottomLevelAccelStruct(commandList, m_ParticleMesh->accelStruct, blasDesc);
        
        commandList->endMarker();
    }

    void BuildParticleIntersectionBLAS(nvrhi::ICommandList* commandList)
    {
        // Only need to create and build the BLAS once, it's immutable
        if (m_ParticleIntersectionBLAS)
            return;

        // A small buffer to hold the AABB data
        nvrhi::BufferDesc aabbBufferDesc;
        aabbBufferDesc.byteSize = sizeof(nvrhi::rt::GeometryAABB);
        aabbBufferDesc.initialState = nvrhi::ResourceStates::CopyDest;
        aabbBufferDesc.keepInitialState = true;
        aabbBufferDesc.isAccelStructBuildInput = true;
        nvrhi::BufferHandle aabbBuffer = GetDevice()->createBuffer(aabbBufferDesc);

        // Write the AABB into the buffer
        nvrhi::rt::GeometryAABB aabb = { -1.f, -1.f, -1.f, 1.f, 1.f, 1.f };
        commandList->writeBuffer(aabbBuffer, &aabb, sizeof(aabb));

        // Create the BLAS with one AABB-type geometry
        nvrhi::rt::AccelStructDesc blasDesc;
        blasDesc.isTopLevel = false;
        blasDesc.debugName = "ParticleIntersectionBLAS";
        blasDesc.addBottomLevelGeometry(nvrhi::rt::GeometryDesc().setAABBs(
            nvrhi::rt::GeometryAABBs()
                .setBuffer(aabbBuffer)
                .setCount(1)));

        m_ParticleIntersectionBLAS = GetDevice()->createAccelStruct(blasDesc);

        // Build the BLAS
        nvrhi::utils::BuildBottomLevelAccelStruct(commandList, m_ParticleIntersectionBLAS, blasDesc);
    }

    bool LoadScene(std::shared_ptr<vfs::IFileSystem> fs, const std::filesystem::path& sceneFileName) override 
    {
        engine::Scene* scene = new engine::Scene(GetDevice(), *m_ShaderFactory, fs, m_TextureCache, m_DescriptorTable, nullptr);

        if (scene->Load(sceneFileName))
        {
            m_Scene = std::unique_ptr<engine::Scene>(scene);
            return true;
        }

        return false;
    }

    bool KeyboardUpdate(int key, int scancode, int action, int mods) override
    {
        m_Camera.KeyboardUpdate(key, scancode, action, mods);

        if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
        {
            m_ui->enableAnimations = !m_ui->enableAnimations;
            return true;
        }
        
        return true;
    }

    bool MousePosUpdate(double xpos, double ypos) override
    {
        m_Camera.MousePosUpdate(xpos, ypos);
        return true;
    }

    bool MouseButtonUpdate(int button, int action, int mods) override
    {
        m_Camera.MouseButtonUpdate(button, action, mods);
        return true;
    }

    bool MouseScrollUpdate(double xoffset, double yoffset) override
    {
        m_Camera.MouseScrollUpdate(xoffset, yoffset);
        return true;
    }

    std::shared_ptr<engine::ShaderFactory> GetShaderFactory() const
    {
        return m_ShaderFactory;
    }

    void Animate(float fElapsedTimeSeconds) override
    {
        m_Camera.Animate(fElapsedTimeSeconds);

        if (IsSceneLoaded() && m_ui->enableAnimations)
        {
            m_WallclockTime += fElapsedTimeSeconds;

            // Animate the live particles, also find the index of the first empty particle slot for spawning later.
            int firstEmptyParticle = -1;
            for (uint32_t i = 0; i < m_Particles.size(); ++i)
            {
                ParticleEntity* particle = &m_Particles[i];
                
                if (particle->active)
                {
                    particle->Animate(fElapsedTimeSeconds);
                }

                if (!particle->active)
                {
                    if (firstEmptyParticle < 0)
                        firstEmptyParticle = i;
                }
            }

            const float particlesPerSecond = 20.f;
            const float particleEmissionPeriod = 1.f / particlesPerSecond;

            // Emit a new particle if enough time has passed since the last emission.
            if (m_WallclockTime - m_LastEmitTime > particleEmissionPeriod && firstEmptyParticle >= 0)
            {
                ParticleEntity* particle = &m_Particles[firstEmptyParticle];

                particle->Emit(m_ui->emitterPosition);

                m_LastEmitTime = m_WallclockTime;
            }
        }
        
        GetDeviceManager()->SetInformativeWindowTitle(g_WindowTitle);
    }
    
    bool CreateComputePipeline(engine::ShaderFactory& shaderFactory)
    {
        char fragmentsText[4];
        snprintf(fragmentsText, sizeof(fragmentsText), "%d", m_ui->mlabFragments);
        std::vector<engine::ShaderMacro> defines = { { "MLAB_FRAGMENTS", fragmentsText } };

        m_ComputeShader = shaderFactory.CreateShader("app/rt_particles.hlsl", "main", &defines, nvrhi::ShaderType::Compute);

        if (!m_ComputeShader)
            return false;

        auto pipelineDesc = nvrhi::ComputePipelineDesc()
            .setComputeShader(m_ComputeShader)
            .addBindingLayout(m_BindingLayout)
            .addBindingLayout(m_BindlessLayout);

        m_ComputePipeline = GetDevice()->createComputePipeline(pipelineDesc);

        if (!m_ComputePipeline)
            return false;

        return true;
    }

    void GetMeshBlasDesc(engine::MeshInfo& mesh, nvrhi::rt::AccelStructDesc& blasDesc) const
    {
        blasDesc.isTopLevel = false;
        blasDesc.debugName = mesh.name;

        for (const auto& geometry : mesh.geometries)
        {
            nvrhi::rt::GeometryDesc geometryDesc;
            auto & triangles = geometryDesc.geometryData.triangles;
            triangles.indexBuffer = mesh.buffers->indexBuffer;
            triangles.indexOffset = (mesh.indexOffset + geometry->indexOffsetInMesh) * sizeof(uint32_t);
            triangles.indexFormat = nvrhi::Format::R32_UINT;
            triangles.indexCount = geometry->numIndices;
            triangles.vertexBuffer = mesh.buffers->vertexBuffer;
            triangles.vertexOffset = (mesh.vertexOffset + geometry->vertexOffsetInMesh) * sizeof(float3) + mesh.buffers->getVertexBufferRange(engine::VertexAttribute::Position).byteOffset;
            triangles.vertexFormat = nvrhi::Format::RGB32_FLOAT;
            triangles.vertexStride = sizeof(float3);
            triangles.vertexCount = geometry->numVertices;
            geometryDesc.geometryType = nvrhi::rt::GeometryType::Triangles;
            geometryDesc.flags = (geometry->material->domain == engine::MaterialDomain::Opaque)
                ? nvrhi::rt::GeometryFlags::Opaque
                : nvrhi::rt::GeometryFlags::None;
            blasDesc.bottomLevelGeometries.push_back(geometryDesc);
        }
        
        blasDesc.buildFlags = nvrhi::rt::AccelStructBuildFlags::PreferFastTrace;
    }

    void CreateAccelStructs(nvrhi::ICommandList* commandList)
    {
        for (const auto& mesh : m_Scene->GetSceneGraph()->GetMeshes())
        {
            nvrhi::rt::AccelStructDesc blasDesc;
            GetMeshBlasDesc(*mesh, blasDesc);

            nvrhi::rt::AccelStructHandle as = GetDevice()->createAccelStruct(blasDesc);

            // Build the BLAS if it's not the particle mesh - that one's dynamic
            if (mesh != m_ParticleMesh)
                nvrhi::utils::BuildBottomLevelAccelStruct(commandList, as, blasDesc);

            mesh->accelStruct = as;
        }
        
        nvrhi::rt::AccelStructDesc tlasDesc;
        tlasDesc.isTopLevel = true;
        // Note: the TLAS will include the scene geometries (including the single instance for geometric particles)
        // and many instances of the intersection BLAS, one instnace per particle.
        const uint32_t numSceneInstances = uint32_t(m_Scene->GetSceneGraph()->GetMeshInstances().size());
        tlasDesc.topLevelMaxInstances = numSceneInstances + c_MaxParticles;
        m_TopLevelAS = GetDevice()->createAccelStruct(tlasDesc);
    }

    void BuildTLAS(nvrhi::ICommandList* commandList, uint32_t frameIndex)
    {   
        std::vector<nvrhi::rt::InstanceDesc> instances;

        // Generate regular instances for scene meshes
        for (const auto& instance : m_Scene->GetSceneGraph()->GetMeshInstances())
        {
            nvrhi::rt::InstanceDesc instanceDesc;
            instanceDesc.bottomLevelAS = instance->GetMesh()->accelStruct;
            assert(instanceDesc.bottomLevelAS);
            instanceDesc.instanceMask = (instance->GetMesh() == m_ParticleMesh)
                ? INSTANCE_MASK_PARTICLE_GEOMETRY
                : INSTANCE_MASK_OPAQUE;
            instanceDesc.instanceID = instance->GetInstanceIndex();

            auto node = instance->GetNode();
            assert(node);
            affineToColumnMajor(node->GetLocalToWorldTransformFloat(), instanceDesc.transform);

            instances.push_back(instanceDesc);
        }

        // Generate intersection instances for active particles
        uint32_t particleIndex = 0;
        for (const auto& particle : m_Particles)
        {
            // Skip inactive particles
            if (!particle.active)
                continue;

            nvrhi::rt::InstanceDesc instanceDesc;
            instanceDesc.bottomLevelAS = m_ParticleIntersectionBLAS;
            instanceDesc.instanceMask = INSTANCE_MASK_INTERSECTION_PARTICLE;
            instanceDesc.instanceID = particleIndex;
            // Scale and translate the AABB to make it contain the particle billboard
            const affine3 transform = scaling(float3(particle.radius)) * translation(particle.position);
            affineToColumnMajor(transform, instanceDesc.transform);

            instances.push_back(instanceDesc);

            ++particleIndex;
        }
        
        commandList->beginMarker("TLAS Update");
        commandList->buildTopLevelAccelStruct(m_TopLevelAS, instances.data(), instances.size());
        commandList->endMarker();
    }


    void BackBufferResizing() override
    { 
        m_ColorBuffer = nullptr;
        m_BindingCache->Clear();
    }

    void Render(nvrhi::IFramebuffer* framebuffer) override
    {
        const auto& fbinfo = framebuffer->getFramebufferInfo();

        if (m_ui->updatePipeline)
        {
            if (!CreateComputePipeline(*m_ShaderFactory))
            {
                glfwSetWindowShouldClose(GetDeviceManager()->GetWindow(), 1);
                return;
            }

            m_ui->updatePipeline = false;
        }

        if (!m_ColorBuffer)
        {
            nvrhi::TextureDesc desc;
            desc.width = fbinfo.width;
            desc.height = fbinfo.height;
            desc.isUAV = true;
            desc.keepInitialState = true;
            desc.format = nvrhi::Format::RGBA16_FLOAT;
            desc.initialState = nvrhi::ResourceStates::UnorderedAccess;
            desc.debugName = "ColorBuffer";
            m_ColorBuffer = GetDevice()->createTexture(desc);

            nvrhi::BindingSetDesc bindingSetDesc;
            bindingSetDesc.bindings = {
                nvrhi::BindingSetItem::ConstantBuffer(0, m_ConstantBuffer),
                nvrhi::BindingSetItem::RayTracingAccelStruct(0, m_TopLevelAS),
                nvrhi::BindingSetItem::StructuredBuffer_SRV(1, m_Scene->GetInstanceBuffer()),
                nvrhi::BindingSetItem::StructuredBuffer_SRV(2, m_Scene->GetGeometryBuffer()),
                nvrhi::BindingSetItem::StructuredBuffer_SRV(3, m_Scene->GetMaterialBuffer()),
                nvrhi::BindingSetItem::StructuredBuffer_SRV(4, m_ParticleInfoBuffer),
                nvrhi::BindingSetItem::Sampler(0, m_CommonPasses->m_AnisotropicWrapSampler),
                nvrhi::BindingSetItem::Texture_UAV(0, m_ColorBuffer)
            };

            m_BindingSet = GetDevice()->createBindingSet(bindingSetDesc, m_BindingLayout);
        }

        auto particleTexture = (m_ui->particleTexture == ParticleTexture::Smoke) ? m_SmokeTexture : m_LogoTexture;
        if (m_ParticleMaterial->baseOrDiffuseTexture != particleTexture)
        {
            m_ParticleMaterial->baseOrDiffuseTexture = particleTexture;
            m_ParticleMaterial->dirty = true;
        }

        nvrhi::Viewport windowViewport(float(fbinfo.width), float(fbinfo.height));
        m_View.SetViewport(windowViewport);
        const float verticalFovRadians = PI_f * 0.25f;
        m_View.SetMatrices(m_Camera.GetWorldToViewMatrix(), perspProjD3DStyleReverse(
            verticalFovRadians, windowViewport.width() / windowViewport.height(), 0.1f));
        m_View.UpdateCache();
        m_Camera.SetView(m_View);

        m_CommandList->open();
        
        if (m_ui->enableAnimations || m_ui->alwaysUpdateOrientation || m_ParticleMaterial->dirty)
        {
            m_Scene->Refresh(m_CommandList, GetFrameIndex());
            BuildParticleGeometry(m_CommandList);
            BuildTLAS(m_CommandList, GetFrameIndex());
        }
        
        GlobalConstants constants = {};
        m_View.FillPlanarViewConstants(constants.view);
        constants.primaryRayConeAngle = verticalFovRadians / float(windowViewport.height());
        constants.reorientParticlesInPrimaryRays = m_ui->reorientParticlesInPrimaryRays;
        constants.reorientParticlesInSecondaryRays = m_ui->reorientParticlesInSecondaryRays;
        constants.orientationMode = m_ui->orientationMode;
        constants.environmentMapTextureIndex = m_EnvironmentMap->bindlessDescriptor.Get();
        m_CommandList->writeBuffer(m_ConstantBuffer, &constants, sizeof(constants));
        
        nvrhi::ComputeState state;
        state.pipeline = m_ComputePipeline;
        state.bindings = { m_BindingSet, m_DescriptorTable->GetDescriptorTable() };
        m_CommandList->setComputeState(state);

        m_CommandList->dispatch(
            div_ceil(fbinfo.width, 16),
            div_ceil(fbinfo.height, 16));
        
        m_CommonPasses->BlitTexture(m_CommandList, framebuffer, m_ColorBuffer, m_BindingCache.get());

        m_CommandList->close();
        GetDevice()->executeCommandList(m_CommandList);
    }
};

class UserInterface : public app::ImGui_Renderer
{
private:
    UIData* m_ui;

public:
    UserInterface(app::DeviceManager* deviceManager, UIData* ui)
        : ImGui_Renderer(deviceManager)
        , m_ui(ui)
    {
        ImGui::GetIO().IniFilename = nullptr;
    }

    void buildUI() override
    {
        ImGui::SetNextWindowPos(ImVec2(10.f, 10.f), 0);
        ImGui::Begin("Settings", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

        ImGui::Checkbox("Animate particles (Space)", &m_ui->enableAnimations);
        ImGui::Checkbox("Update orientation when paused", &m_ui->alwaysUpdateOrientation);
        ImGui::Separator();

        ImGui::Text("Orientation mode:");
        ImGui::Indent();
        ImGui::Combo("##orientationMode", (int*)&m_ui->orientationMode,
            "Accumulated Vector Transform\0"
            "Quaternion Rotation\0"
            "Beam or Vertical Sprite\0"
            "Basis (RTG2)\0");
        ImGui::Unindent();
        ImGui::Separator();

        ImGui::Text("Reorient particles:");
        ImGui::Indent();
        ImGui::Checkbox("In primary rays", &m_ui->reorientParticlesInPrimaryRays);
        ImGui::Checkbox("In secondary rays", &m_ui->reorientParticlesInSecondaryRays);
        ImGui::Unindent();
        ImGui::Separator();

        // MLAB fragment count combo-box
        uint allowedMlabFragmentCounts[] = {1, 2, 4, 8};
        char fragmentsText[4];
        snprintf(fragmentsText, sizeof(fragmentsText), "%d", m_ui->mlabFragments);
        ImGui::PushItemWidth(40.f);
        uint newFragmentCount = m_ui->mlabFragments;
        if (ImGui::BeginCombo("Blending fragments", fragmentsText))
        {
            for (uint fragmentCount : allowedMlabFragmentCounts)
            {
                snprintf(fragmentsText, sizeof(fragmentsText), "%d", fragmentCount);
                if (ImGui::Selectable(fragmentsText, newFragmentCount == fragmentCount))
                    newFragmentCount = fragmentCount;
            }
            ImGui::EndCombo();
        }
        ImGui::PopItemWidth();
        if (newFragmentCount != m_ui->mlabFragments)
        {
            m_ui->mlabFragments = newFragmentCount;
            m_ui->updatePipeline = true;
        }
        ImGui::Separator();

        ImGui::Text("Emitter position:");
        ImGui::Indent();
        ImGui::DragFloat3("##emitterPosition", &m_ui->emitterPosition.x, 0.01f);
        ImGui::Unindent();

        ImGui::Text("Particle texture:");
        ImGui::Indent();
        ImGui::Combo("##particleTexture", (int*)&m_ui->particleTexture, "Smoke\0Logo\0");
        ImGui::Unindent();

        // End of window
        ImGui::End();
    }
};

#ifdef WIN32
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
#else
int main(int __argc, const char** __argv)
#endif
{
    nvrhi::GraphicsAPI api = app::GetGraphicsAPIFromCommandLine(__argc, __argv);
    app::DeviceManager* deviceManager = app::DeviceManager::Create(api);

    app::DeviceCreationParameters deviceParams;
    deviceParams.enableRayTracingExtensions = true;
    
    for (int i = 1; i < __argc; i++)
    {
        if (strcmp(__argv[i], "-debug") == 0)
        {
            deviceParams.enableDebugRuntime = true;
            deviceParams.enableNvrhiValidationLayer = true;
        }
    }

    if (!deviceManager->CreateWindowDeviceAndSwapChain(deviceParams, g_WindowTitle))
    {
        log::fatal("Cannot initialize a graphics device with the requested parameters");
        return 1;
    }
    
    if (!deviceManager->GetDevice()->queryFeatureSupport(nvrhi::Feature::RayQuery))
    {
        log::fatal("The graphics device does not support Ray Queries");
        return 1;
    }

    {
        UIData uiData;
        RayTracedParticles example(deviceManager, &uiData);
        UserInterface gui(deviceManager, &uiData);

        if (example.Init() && gui.Init(example.GetShaderFactory()))
        {
            deviceManager->AddRenderPassToBack(&example);
            deviceManager->AddRenderPassToBack(&gui);
            deviceManager->RunMessageLoop();
            deviceManager->RemoveRenderPass(&gui);
            deviceManager->RemoveRenderPass(&example);
        }
    }
    
    deviceManager->Shutdown();

    delete deviceManager;

    return 0;
}
