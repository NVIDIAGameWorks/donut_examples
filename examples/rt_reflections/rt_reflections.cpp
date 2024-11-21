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

#include <donut/render/GBufferFillPass.h>
#include <donut/render/ForwardShadingPass.h>
#include <donut/render/DrawStrategy.h>
#include <donut/app/ApplicationBase.h>
#include <donut/app/Camera.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/engine/TextureCache.h>
#include <donut/engine/Scene.h>
#include <donut/engine/FramebufferFactory.h>
#include <donut/engine/BindingCache.h>
#include <donut/app/DeviceManager.h>
#include <donut/core/log.h>
#include <donut/core/vfs/VFS.h>
#include <donut/core/math/math.h>
#include <nvrhi/utils.h>

using namespace donut;
using namespace donut::math;

#include "lighting_cb.h"

static const char* g_WindowTitle = "Donut Example: Ray Traced Reflections";

class RenderTargets
{
public:
    nvrhi::TextureHandle m_Depth;
    nvrhi::TextureHandle m_GBufferDiffuse;
    nvrhi::TextureHandle m_GBufferSpecular;
    nvrhi::TextureHandle m_GBufferNormals;
    nvrhi::TextureHandle m_GBufferEmissive;
    nvrhi::TextureHandle m_HdrColor;

    std::shared_ptr<engine::FramebufferFactory> m_HdrFramebuffer;
    std::shared_ptr<engine::FramebufferFactory> m_HdrFramebufferDepth;
    std::shared_ptr<engine::FramebufferFactory> m_GBufferFramebuffer;
    
    int2 m_Size;
    
    RenderTargets(nvrhi::IDevice* device, int2 size)
        : m_Size(size)
    {
        nvrhi::TextureDesc desc;
        desc.width = size.x;
        desc.height = size.y;
        desc.isRenderTarget = true;
        desc.useClearValue = true;
        desc.clearValue = nvrhi::Color(0.f);
        desc.keepInitialState = true;

        desc.isTypeless = true;
        desc.format = nvrhi::Format::D24S8;
        desc.initialState = nvrhi::ResourceStates::DepthWrite;
        desc.debugName = "DepthBuffer";
        m_Depth = device->createTexture(desc);

        desc.clearValue = nvrhi::Color(0.f);
        desc.isTypeless = false;
        desc.format = nvrhi::Format::RGBA16_FLOAT;
        desc.initialState = nvrhi::ResourceStates::RenderTarget;
        desc.isUAV = true;
        desc.debugName = "HdrColor";
        m_HdrColor = device->createTexture(desc);

        desc.format = nvrhi::Format::SRGBA8_UNORM;
        desc.isUAV = false;
        desc.debugName = "GBufferDiffuse";
        m_GBufferDiffuse = device->createTexture(desc);

        desc.format = nvrhi::Format::SRGBA8_UNORM;
        desc.debugName = "GBufferSpecular";
        m_GBufferSpecular = device->createTexture(desc);

        desc.format = nvrhi::Format::RGBA16_SNORM;
        desc.debugName = "GBufferNormals";
        m_GBufferNormals = device->createTexture(desc);

        desc.format = nvrhi::Format::RGBA16_FLOAT;
        desc.debugName = "GBufferEmissive";
        m_GBufferEmissive = device->createTexture(desc);

        m_GBufferFramebuffer = std::make_shared<engine::FramebufferFactory>(device);
        m_GBufferFramebuffer->RenderTargets = { m_GBufferDiffuse, m_GBufferSpecular, m_GBufferNormals, m_GBufferEmissive };
        m_GBufferFramebuffer->DepthTarget = m_Depth;

        m_HdrFramebuffer = std::make_shared<engine::FramebufferFactory>(device);
        m_HdrFramebuffer->RenderTargets = { m_HdrColor };

        m_HdrFramebufferDepth = std::make_shared<engine::FramebufferFactory>(device);
        m_HdrFramebufferDepth->RenderTargets = { m_HdrColor };
        m_HdrFramebufferDepth->DepthTarget = m_Depth;
    }

    bool IsUpdateRequired(int2 size)
    {
        if (any(m_Size != size))
            return true;

        return false;
    }

    void Clear(nvrhi::ICommandList* commandList)
    {
        commandList->clearDepthStencilTexture(m_Depth, nvrhi::AllSubresources, true, 0.f, true, 0);
        commandList->clearTextureFloat(m_HdrColor, nvrhi::AllSubresources, nvrhi::Color(0.f));
        commandList->clearTextureFloat(m_GBufferDiffuse, nvrhi::AllSubresources, nvrhi::Color(0.f));
        commandList->clearTextureFloat(m_GBufferSpecular, nvrhi::AllSubresources, nvrhi::Color(0.f));
        commandList->clearTextureFloat(m_GBufferNormals, nvrhi::AllSubresources, nvrhi::Color(0.f));
        commandList->clearTextureFloat(m_GBufferEmissive, nvrhi::AllSubresources, nvrhi::Color(0.f));
    }

    const int2& GetSize()
    {
        return m_Size;
    }
};

class VariableRateShading : public app::ApplicationBase
{
private:
	std::shared_ptr<vfs::RootFileSystem> m_RootFS;

    nvrhi::ShaderLibraryHandle m_ShaderLibrary;
    nvrhi::rt::PipelineHandle m_Pipeline;
    nvrhi::rt::ShaderTableHandle m_ShaderTable;
    nvrhi::CommandListHandle m_CommandList;
    nvrhi::BindingLayoutHandle m_GlobalBindingLayout;
    nvrhi::BindingLayoutHandle m_LocalBindingLayout;
    nvrhi::BindingSetHandle m_BindingSet;

    nvrhi::rt::AccelStructHandle m_BottomLevelAS;
    nvrhi::rt::AccelStructHandle m_TopLevelAS;

    nvrhi::BufferHandle m_ConstantBuffer;

    std::shared_ptr<engine::ShaderFactory> m_ShaderFactory;
    std::unique_ptr<engine::Scene> m_Scene;
    std::unique_ptr<render::GBufferFillPass> m_GBufferPass;
    std::unique_ptr<render::ForwardShadingPass> m_ForwardPass;
    std::unique_ptr<RenderTargets> m_RenderTargets;
    app::FirstPersonCamera m_Camera;
    engine::PlanarView m_View;
    std::shared_ptr<engine::DirectionalLight> m_SunLight;
    std::unique_ptr<render::InstancedOpaqueDrawStrategy> m_OpaqueDrawStrategy;
    std::unique_ptr<render::TransparentDrawStrategy> m_TransparentDrawStrategy;
    std::unique_ptr<engine::BindingCache> m_BindingCache;

public:
    using ApplicationBase::ApplicationBase;

    bool Init()
    {
        std::filesystem::path sceneFileName = app::GetDirectoryWithExecutable().parent_path() / "media/glTF-Sample-Assets/Models/Sponza/glTF/Sponza.gltf";
        std::filesystem::path frameworkShaderPath = app::GetDirectoryWithExecutable() / "shaders/framework" / app::GetShaderTypeName(GetDevice()->getGraphicsAPI());
        std::filesystem::path appShaderPath = app::GetDirectoryWithExecutable() / "shaders/rt_reflections" / app::GetShaderTypeName(GetDevice()->getGraphicsAPI());
        
		m_RootFS = std::make_shared<vfs::RootFileSystem>();
		m_RootFS->mount("/shaders/donut", frameworkShaderPath);
		m_RootFS->mount("/shaders/app", appShaderPath);

		m_ShaderFactory = std::make_shared<engine::ShaderFactory>(GetDevice(), m_RootFS, "/shaders");
        m_CommonPasses = std::make_shared<engine::CommonRenderPasses>(GetDevice(), m_ShaderFactory);
        m_BindingCache = std::make_unique<engine::BindingCache>(GetDevice());

        auto nativeFS = std::make_shared<vfs::NativeFileSystem>();
        m_TextureCache = std::make_shared<engine::TextureCache>(GetDevice(), nativeFS, nullptr);
        
        SetAsynchronousLoadingEnabled(false);
        BeginLoadingScene(nativeFS, sceneFileName);

        m_SunLight = std::make_shared<engine::DirectionalLight>();
        m_Scene->GetSceneGraph()->AttachLeafNode(m_Scene->GetSceneGraph()->GetRootNode(), m_SunLight);

        m_SunLight->SetDirection(double3(0.1, -1.0, 0.15));
        m_SunLight->angularSize = 0.53f;
        m_SunLight->irradiance = 1.f;

        m_Scene->FinishedLoading(GetFrameIndex());

        m_OpaqueDrawStrategy = std::make_unique<render::InstancedOpaqueDrawStrategy>();
        m_TransparentDrawStrategy = std::make_unique<render::TransparentDrawStrategy>();

        m_Camera.LookAt(float3(0.f, 1.8f, 0.f), float3(1.f, 1.8f, 0.f));
        m_Camera.SetMoveSpeed(3.f);

        m_ConstantBuffer = GetDevice()->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(sizeof(LightingConstants), "LightingConstants", engine::c_MaxRenderPassConstantBufferVersions));

        if (!CreateRayTracingPipeline(*m_ShaderFactory))
            return false;

        m_CommandList = GetDevice()->createCommandList();

        m_CommandList->open();

        CreateAccelStruct(m_CommandList);

        m_CommandList->close();
        GetDevice()->executeCommandList(m_CommandList);

        GetDevice()->waitForIdle();

        return true;
    }

    bool LoadScene(std::shared_ptr<vfs::IFileSystem> fs, const std::filesystem::path& sceneFileName) override 
    {
        std::unique_ptr<engine::Scene> scene = std::make_unique<engine::Scene>(GetDevice(),
            *m_ShaderFactory, fs, m_TextureCache, nullptr, nullptr);

        if (scene->Load(sceneFileName))
        {
            m_Scene = std::move(scene);
            return true;
        }

        return false;
    }

    bool KeyboardUpdate(int key, int scancode, int action, int mods) override
    {
        m_Camera.KeyboardUpdate(key, scancode, action, mods);
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

    void Animate(float fElapsedTimeSeconds) override
    {
        m_Camera.Animate(fElapsedTimeSeconds);
        GetDeviceManager()->SetInformativeWindowTitle(g_WindowTitle);
    }

    bool CreateRayTracingPipeline(engine::ShaderFactory& shaderFactory)
    {
        m_ShaderLibrary = shaderFactory.CreateShaderLibrary("app/rt_reflections.hlsl", nullptr);

        if (!m_ShaderLibrary)
            return false;

        nvrhi::BindingLayoutDesc globalBindingLayoutDesc;
        globalBindingLayoutDesc.visibility = nvrhi::ShaderType::All;
        globalBindingLayoutDesc.registerSpace = REFLECTIONS_SPACE_GLOBAL;
        globalBindingLayoutDesc.bindings = {
            nvrhi::BindingLayoutItem::VolatileConstantBuffer(REFLECTIONS_BINDING_LIGHTING_CONSTANTS),
            nvrhi::BindingLayoutItem::RayTracingAccelStruct(REFLECTIONS_BINDING_SCENE_BVH),
            nvrhi::BindingLayoutItem::Texture_SRV(REFLECTIONS_BINDING_GBUFFER_DEPTH_TEXTURE),
            nvrhi::BindingLayoutItem::Texture_SRV(REFLECTIONS_BINDING_GBUFFER_0_TEXTURE),
            nvrhi::BindingLayoutItem::Texture_SRV(REFLECTIONS_BINDING_GBUFFER_1_TEXTURE),
            nvrhi::BindingLayoutItem::Texture_SRV(REFLECTIONS_BINDING_GBUFFER_2_TEXTURE),
            nvrhi::BindingLayoutItem::Texture_SRV(REFLECTIONS_BINDING_GBUFFER_3_TEXTURE),
            nvrhi::BindingLayoutItem::Texture_UAV(REFLECTIONS_BINDING_OUTPUT_UAV),
            nvrhi::BindingLayoutItem::Sampler(REFLECTIONS_BINDING_MATERIAL_SAMPLER)
        };

        m_GlobalBindingLayout = GetDevice()->createBindingLayout(globalBindingLayoutDesc);

        nvrhi::BindingLayoutDesc localBindingLayoutDesc;
        localBindingLayoutDesc.visibility = nvrhi::ShaderType::All;
        localBindingLayoutDesc.registerSpace = REFLECTIONS_SPACE_LOCAL;
        localBindingLayoutDesc.bindings = {
            nvrhi::BindingLayoutItem::TypedBuffer_SRV(REFLECTIONS_BINDING_INDEX_BUFFER),
            nvrhi::BindingLayoutItem::TypedBuffer_SRV(REFLECTIONS_BINDING_TEX_COORD_BUFFER),
            nvrhi::BindingLayoutItem::TypedBuffer_SRV(REFLECTIONS_BINDING_NORMAL_BUFFER),
            nvrhi::BindingLayoutItem::Texture_SRV(REFLECTIONS_BINDING_DIFFUSE_TEXTURE),
            nvrhi::BindingLayoutItem::Texture_SRV(REFLECTIONS_BINDING_SPECULAR_TEXTURE),
            nvrhi::BindingLayoutItem::Texture_SRV(REFLECTIONS_BINDING_NORMAL_TEXTURE),
            nvrhi::BindingLayoutItem::Texture_SRV(REFLECTIONS_BINDING_EMISSIVE_TEXTURE),
            nvrhi::BindingLayoutItem::Texture_SRV(REFLECTIONS_BINDING_OCCLUSION_TEXTURE),
            nvrhi::BindingLayoutItem::Texture_SRV(REFLECTIONS_BINDING_TRANSMISSION_TEXTURE),
            nvrhi::BindingLayoutItem::Texture_SRV(REFLECTIONS_BINDING_OPACITY_TEXTURE),
            nvrhi::BindingLayoutItem::ConstantBuffer(REFLECTIONS_BINDING_MATERIAL_CONSTANTS)
        };

        m_LocalBindingLayout = GetDevice()->createBindingLayout(localBindingLayoutDesc);

        nvrhi::rt::PipelineDesc pipelineDesc;
        pipelineDesc.globalBindingLayouts = { m_GlobalBindingLayout };
        pipelineDesc.shaders = {
            { "", m_ShaderLibrary->getShader("RayGen", nvrhi::ShaderType::RayGeneration), nullptr },
            { "", m_ShaderLibrary->getShader("ShadowMiss", nvrhi::ShaderType::Miss), nullptr },
            { "", m_ShaderLibrary->getShader("ReflectionMiss", nvrhi::ShaderType::Miss), nullptr }
        };

        pipelineDesc.hitGroups = { 
            { 
                "ShadowHitGroup",
                nullptr, // closestHitShader
                nullptr, // anyHitShader
                nullptr, // intersectionShader
                nullptr, // bindingLayout
                false  // isProceduralPrimitive
            },
            { 
                "ReflectionHitGroup", 
                m_ShaderLibrary->getShader("ReflectionClosestHit", nvrhi::ShaderType::ClosestHit), 
                nullptr, // anyHitShader
                nullptr, // intersectionShader
                m_LocalBindingLayout, 
                false // isProceduralPrimitive
            },
        };

        pipelineDesc.maxPayloadSize = sizeof(dm::float4);
        pipelineDesc.maxRecursionDepth = 2;

        m_Pipeline = GetDevice()->createRayTracingPipeline(pipelineDesc);

        m_ShaderTable = m_Pipeline->createShaderTable();
        m_ShaderTable->setRayGenerationShader("RayGen");
        m_ShaderTable->addMissShader("ShadowMiss");
        m_ShaderTable->addMissShader("ReflectionMiss");

        for (const auto& mesh : m_Scene->GetSceneGraph()->GetMeshes())
        {
            for (const auto& geometry : mesh->geometries)
            {
                nvrhi::BindingSetDesc bindingSetDesc;
                bindingSetDesc.bindings = {
                    nvrhi::BindingSetItem::TypedBuffer_SRV(
                        REFLECTIONS_BINDING_INDEX_BUFFER,
                        mesh->buffers->indexBuffer,
                        nvrhi::Format::R32_UINT,
                        nvrhi::BufferRange((mesh->indexOffset + geometry->indexOffsetInMesh) * sizeof(uint32_t), geometry->numIndices * sizeof(uint32_t))),
                    nvrhi::BindingSetItem::TypedBuffer_SRV(
                        REFLECTIONS_BINDING_TEX_COORD_BUFFER,
                        mesh->buffers->vertexBuffer,
                        nvrhi::Format::RG32_FLOAT,
                        nvrhi::BufferRange((mesh->vertexOffset + geometry->vertexOffsetInMesh) * sizeof(float2) + mesh->buffers->getVertexBufferRange(engine::VertexAttribute::TexCoord1).byteOffset, geometry->numVertices * sizeof(float2))),
                    nvrhi::BindingSetItem::TypedBuffer_SRV(
                        REFLECTIONS_BINDING_NORMAL_BUFFER,
                        mesh->buffers->vertexBuffer,
                        nvrhi::Format::RGBA8_SNORM,
                        nvrhi::BufferRange((mesh->vertexOffset + geometry->vertexOffsetInMesh) * sizeof(uint32_t) + mesh->buffers->getVertexBufferRange(engine::VertexAttribute::Normal).byteOffset, geometry->numVertices * sizeof(uint32_t))),
                    nvrhi::BindingSetItem::Texture_SRV(
                        REFLECTIONS_BINDING_DIFFUSE_TEXTURE,
                        geometry->material->baseOrDiffuseTexture && geometry->material->baseOrDiffuseTexture->texture ? geometry->material->baseOrDiffuseTexture->texture : m_CommonPasses->m_WhiteTexture),
                    nvrhi::BindingSetItem::Texture_SRV(
                        REFLECTIONS_BINDING_SPECULAR_TEXTURE,
                        geometry->material->metalRoughOrSpecularTexture && geometry->material->metalRoughOrSpecularTexture->texture ? geometry->material->metalRoughOrSpecularTexture->texture : m_CommonPasses->m_WhiteTexture),
                    nvrhi::BindingSetItem::Texture_SRV(
                        REFLECTIONS_BINDING_NORMAL_TEXTURE,
                        geometry->material->normalTexture && geometry->material->normalTexture->texture ? geometry->material->normalTexture->texture : m_CommonPasses->m_BlackTexture),
                    nvrhi::BindingSetItem::Texture_SRV(
                        REFLECTIONS_BINDING_EMISSIVE_TEXTURE,
                        geometry->material->emissiveTexture && geometry->material->emissiveTexture->texture ? geometry->material->emissiveTexture->texture : m_CommonPasses->m_BlackTexture),
                    nvrhi::BindingSetItem::Texture_SRV(
                        REFLECTIONS_BINDING_OCCLUSION_TEXTURE,
                        geometry->material->occlusionTexture && geometry->material->occlusionTexture->texture ? geometry->material->occlusionTexture->texture : m_CommonPasses->m_WhiteTexture),
                    nvrhi::BindingSetItem::Texture_SRV(
                        REFLECTIONS_BINDING_TRANSMISSION_TEXTURE,
                        geometry->material->transmissionTexture && geometry->material->transmissionTexture->texture ? geometry->material->transmissionTexture->texture : m_CommonPasses->m_BlackTexture),
                    nvrhi::BindingSetItem::Texture_SRV(
                        REFLECTIONS_BINDING_OPACITY_TEXTURE,
                        geometry->material->opacityTexture && geometry->material->opacityTexture->texture ? geometry->material->opacityTexture->texture : m_CommonPasses->m_WhiteTexture),
                    nvrhi::BindingSetItem::ConstantBuffer(
                        REFLECTIONS_BINDING_MATERIAL_CONSTANTS,
                        geometry->material->materialConstants)
                };

                nvrhi::BindingSetHandle localBindingSet = GetDevice()->createBindingSet(bindingSetDesc, m_LocalBindingLayout);

                int hitGroupIndex = m_ShaderTable->addHitGroup("ShadowHitGroup", nullptr);
                assert(hitGroupIndex == geometry->globalGeometryIndex * 2);

                m_ShaderTable->addHitGroup("ReflectionHitGroup", localBindingSet);
            }
        }

        return true;
    }

    void CreateAccelStruct(nvrhi::ICommandList* commandList)
    {
        for (const auto& mesh : m_Scene->GetSceneGraph()->GetMeshes())
        {
            nvrhi::rt::AccelStructDesc blasDesc;
            blasDesc.isTopLevel = false;

            for (const auto& geometry : mesh->geometries)
            {
                nvrhi::rt::GeometryDesc geometryDesc;
                auto& triangles = geometryDesc.geometryData.triangles;
                triangles.indexBuffer = mesh->buffers->indexBuffer;
                triangles.indexOffset = (mesh->indexOffset + geometry->indexOffsetInMesh) * sizeof(uint32_t);
                triangles.indexFormat = nvrhi::Format::R32_UINT;
                triangles.indexCount = geometry->numIndices;
                triangles.vertexBuffer = mesh->buffers->vertexBuffer;
                triangles.vertexOffset = (mesh->vertexOffset + geometry->vertexOffsetInMesh) * sizeof(float3) + mesh->buffers->getVertexBufferRange(engine::VertexAttribute::Position).byteOffset;
                triangles.vertexFormat = nvrhi::Format::RGB32_FLOAT;
                triangles.vertexStride = sizeof(float3);
                triangles.vertexCount = geometry->numVertices;
                geometryDesc.geometryType = nvrhi::rt::GeometryType::Triangles;
                geometryDesc.flags = nvrhi::rt::GeometryFlags::Opaque;
                blasDesc.bottomLevelGeometries.push_back(geometryDesc);
            }

            nvrhi::rt::AccelStructHandle as = GetDevice()->createAccelStruct(blasDesc);
            nvrhi::utils::BuildBottomLevelAccelStruct(commandList, as, blasDesc);

            mesh->accelStruct = as;
        }


        nvrhi::rt::AccelStructDesc tlasDesc;
        tlasDesc.isTopLevel = true;

        std::vector<nvrhi::rt::InstanceDesc> instances;
        for (const auto& instance : m_Scene->GetSceneGraph()->GetMeshInstances())
        {
            const auto& mesh = instance->GetMesh();
            
            nvrhi::rt::InstanceDesc instanceDesc;
            instanceDesc.bottomLevelAS = mesh->accelStruct;
            assert(instanceDesc.bottomLevelAS);
            instanceDesc.instanceMask = 1;
            instanceDesc.instanceContributionToHitGroupIndex = mesh->geometries[0]->globalGeometryIndex * 2;
            
            auto node = instance->GetNode();
            assert(node);
            dm::affineToColumnMajor(node->GetLocalToWorldTransformFloat(), instanceDesc.transform);
            
            instances.push_back(instanceDesc);
        }

        tlasDesc.topLevelMaxInstances = instances.size();
        m_TopLevelAS = GetDevice()->createAccelStruct(tlasDesc);

        commandList->buildTopLevelAccelStruct(m_TopLevelAS, instances.data(), instances.size());
    }
    
    void BackBufferResizing() override
    { 
        m_RenderTargets = nullptr;
        m_BindingCache->Clear();
        m_GBufferPass = nullptr;
        m_ForwardPass = nullptr;
    }

    void Render(nvrhi::IFramebuffer* framebuffer) override
    {
        const auto& fbinfo = framebuffer->getFramebufferInfo();

        if (!m_RenderTargets)
        {
            m_RenderTargets = std::make_unique<RenderTargets>(GetDevice(), int2(fbinfo.width, fbinfo.height));

            nvrhi::BindingSetDesc bindingSetDesc;
            bindingSetDesc.bindings = {
                nvrhi::BindingSetItem::ConstantBuffer(REFLECTIONS_BINDING_LIGHTING_CONSTANTS, m_ConstantBuffer),
                nvrhi::BindingSetItem::RayTracingAccelStruct(REFLECTIONS_BINDING_SCENE_BVH, m_TopLevelAS),
                nvrhi::BindingSetItem::Texture_SRV(REFLECTIONS_BINDING_GBUFFER_DEPTH_TEXTURE, m_RenderTargets->m_Depth),
                nvrhi::BindingSetItem::Texture_SRV(REFLECTIONS_BINDING_GBUFFER_0_TEXTURE, m_RenderTargets->m_GBufferDiffuse),
                nvrhi::BindingSetItem::Texture_SRV(REFLECTIONS_BINDING_GBUFFER_1_TEXTURE, m_RenderTargets->m_GBufferSpecular),
                nvrhi::BindingSetItem::Texture_SRV(REFLECTIONS_BINDING_GBUFFER_2_TEXTURE, m_RenderTargets->m_GBufferNormals),
                nvrhi::BindingSetItem::Texture_SRV(REFLECTIONS_BINDING_GBUFFER_3_TEXTURE, m_RenderTargets->m_GBufferEmissive),
                nvrhi::BindingSetItem::Texture_UAV(REFLECTIONS_BINDING_OUTPUT_UAV, m_RenderTargets->m_HdrColor),
                nvrhi::BindingSetItem::Sampler(REFLECTIONS_BINDING_MATERIAL_SAMPLER, m_CommonPasses->m_LinearWrapSampler)
            };

            m_BindingSet = GetDevice()->createBindingSet(bindingSetDesc, m_GlobalBindingLayout);
        }

        if (!m_GBufferPass)
        {
            m_GBufferPass = std::make_unique<render::GBufferFillPass>(GetDevice(), m_CommonPasses);

            render::GBufferFillPass::CreateParameters gbufferParams;
            m_GBufferPass->Init(*m_ShaderFactory, gbufferParams);
        }

        if (!m_ForwardPass)
        {
            m_ForwardPass = std::make_unique<render::ForwardShadingPass>(GetDevice(), m_CommonPasses);

            render::ForwardShadingPass::CreateParameters forwardParams;
            m_ForwardPass->Init(*m_ShaderFactory, forwardParams);
        }

        nvrhi::Viewport windowViewport(float(fbinfo.width), float(fbinfo.height));
        m_View.SetViewport(windowViewport);
        m_View.SetMatrices(m_Camera.GetWorldToViewMatrix(), perspProjD3DStyleReverse(dm::PI_f * 0.25f, windowViewport.width() / windowViewport.height(), 0.1f));
        m_View.UpdateCache();


        m_CommandList->open();

        m_RenderTargets->Clear(m_CommandList);
        render::GBufferFillPass::Context gbufferContext;
        render::RenderCompositeView(m_CommandList, &m_View, &m_View, *m_RenderTargets->m_GBufferFramebuffer, 
            m_Scene->GetSceneGraph()->GetRootNode(), *m_OpaqueDrawStrategy, *m_GBufferPass, gbufferContext);

        LightingConstants constants = {};
        constants.ambientColor = float4(0.2f);
        m_View.FillPlanarViewConstants(constants.view);
        m_SunLight->FillLightConstants(constants.light);
        m_CommandList->writeBuffer(m_ConstantBuffer, &constants, sizeof(constants));

        nvrhi::rt::State state;
        state.shaderTable = m_ShaderTable;
        state.bindings = { m_BindingSet };
        m_CommandList->setRayTracingState(state);

        nvrhi::rt::DispatchRaysArguments args;
        args.width = fbinfo.width;
        args.height = fbinfo.height;
        m_CommandList->dispatchRays(args);

        render::ForwardShadingPass::Context forwardContext;
        m_ForwardPass->PrepareLights(forwardContext, m_CommandList, m_Scene->GetSceneGraph()->GetLights(), constants.ambientColor, constants.ambientColor, {});
        render::RenderCompositeView(m_CommandList, &m_View, &m_View, *m_RenderTargets->m_HdrFramebufferDepth,
            m_Scene->GetSceneGraph()->GetRootNode(), *m_TransparentDrawStrategy, *m_ForwardPass, forwardContext);
        
        m_CommonPasses->BlitTexture(m_CommandList, framebuffer, m_RenderTargets->m_HdrColor, m_BindingCache.get());

        m_CommandList->close();
        GetDevice()->executeCommandList(m_CommandList);

        GetDeviceManager()->SetVsyncEnabled(true);
    }

};

#ifdef WIN32
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
#else
int main(int __argc, const char** __argv)
#endif
{
    app::DeviceManager* deviceManager = app::DeviceManager::Create(nvrhi::GraphicsAPI::D3D12);

    app::DeviceCreationParameters deviceParams;
#ifdef _DEBUG
    deviceParams.enableDebugRuntime = true; 
    deviceParams.enableNvrhiValidationLayer = true;
#endif

    if (!deviceManager->CreateWindowDeviceAndSwapChain(deviceParams, g_WindowTitle))
    {
        log::error("Cannot initialize a graphics device with the requested parameters");
        return 1;
    }

    if (!deviceManager->GetDevice()->queryFeatureSupport(nvrhi::Feature::RayTracingPipeline))
    {
        log::error("The graphics device does not support Ray Tracing Pipelines");
        return 1;
    }
    
    {
        VariableRateShading example(deviceManager);
        if (example.Init())
        {
            deviceManager->AddRenderPassToBack(&example);
            deviceManager->RunMessageLoop();
            deviceManager->RemoveRenderPass(&example);
        }
    }
    
    deviceManager->Shutdown();

    delete deviceManager;

    return 0;
}
