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

#include <donut/app/ApplicationBase.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/engine/BindingCache.h>
#include <donut/app/DeviceManager.h>
#include <donut/core/log.h>
#include <donut/core/vfs/VFS.h>
#include <donut/core/math/math.h>
#include <nvrhi/utils.h>


using namespace donut;
using namespace donut::math;

static const char* g_WindowTitle = "Donut Example: Ray Traced Triangle";

class RayTracedTriangle : public app::IRenderPass
{
private:
    nvrhi::ShaderLibraryHandle m_ShaderLibrary;
    nvrhi::rt::PipelineHandle m_Pipeline;
    nvrhi::rt::ShaderTableHandle m_ShaderTable;
    nvrhi::CommandListHandle m_CommandList;
    nvrhi::BindingLayoutHandle m_BindingLayout;
    nvrhi::BindingSetHandle m_BindingSet;
    nvrhi::rt::AccelStructHandle m_BottomLevelAS;
    nvrhi::rt::AccelStructHandle m_TopLevelAS;
    nvrhi::TextureHandle m_RenderTarget;
    std::shared_ptr<engine::CommonRenderPasses> m_CommonPasses;
    std::unique_ptr<engine::BindingCache> m_BindingCache;

public:
    using IRenderPass::IRenderPass;

    bool Init()
    {
        auto nativeFS = std::make_shared<vfs::NativeFileSystem>();

        std::filesystem::path frameworkShaderPath = app::GetDirectoryWithExecutable() / "shaders/framework" / app::GetShaderTypeName(GetDevice()->getGraphicsAPI());
        std::filesystem::path appShaderPath = app::GetDirectoryWithExecutable() / "shaders/rt_triangle" / app::GetShaderTypeName(GetDevice()->getGraphicsAPI());
        
		std::shared_ptr<vfs::RootFileSystem> rootFS = std::make_shared<vfs::RootFileSystem>();
		rootFS->mount("/shaders/donut", frameworkShaderPath);
		rootFS->mount("/shaders/app", appShaderPath);

        std::shared_ptr<engine::ShaderFactory> shaderFactory = std::make_shared<engine::ShaderFactory>(GetDevice(), rootFS, "/shaders");
        m_ShaderLibrary = shaderFactory->CreateShaderLibrary("app/rt_triangle.hlsl", nullptr);

        if (!m_ShaderLibrary)
        {
            return false;
        }

        m_BindingCache = std::make_unique<engine::BindingCache>(GetDevice());

        m_CommonPasses = std::make_shared<engine::CommonRenderPasses>(GetDevice(), shaderFactory);

        nvrhi::BindingLayoutDesc globalBindingLayoutDesc;
        globalBindingLayoutDesc.visibility = nvrhi::ShaderType::All;
        globalBindingLayoutDesc.bindings = {
            { 0, nvrhi::ResourceType::RayTracingAccelStruct },
            { 0, nvrhi::ResourceType::Texture_UAV }
        };

        m_BindingLayout = GetDevice()->createBindingLayout(globalBindingLayoutDesc);

        nvrhi::rt::PipelineDesc pipelineDesc;
        pipelineDesc.globalBindingLayouts = { m_BindingLayout };
        pipelineDesc.shaders = {
            { "", m_ShaderLibrary->getShader("RayGen", nvrhi::ShaderType::RayGeneration), nullptr },
            { "", m_ShaderLibrary->getShader("Miss", nvrhi::ShaderType::Miss), nullptr }
        };

        pipelineDesc.hitGroups = { { 
            "HitGroup", 
            m_ShaderLibrary->getShader("ClosestHit", nvrhi::ShaderType::ClosestHit), 
            nullptr, // anyHitShader
            nullptr, // intersectionShader
            nullptr, // bindingLayout
            false  // isProceduralPrimitive
        } };

        pipelineDesc.maxPayloadSize = sizeof(dm::float4);

        m_Pipeline = GetDevice()->createRayTracingPipeline(pipelineDesc);

        m_ShaderTable = m_Pipeline->createShaderTable();
        m_ShaderTable->setRayGenerationShader("RayGen");
        m_ShaderTable->addHitGroup("HitGroup");
        m_ShaderTable->addMissShader("Miss");


        m_CommandList = GetDevice()->createCommandList();

        m_CommandList->open();

        nvrhi::BufferDesc bufferDesc;
        bufferDesc.byteSize = sizeof(uint) * 3;
        bufferDesc.initialState = nvrhi::ResourceStates::ShaderResource;
        bufferDesc.keepInitialState = true;
        bufferDesc.isAccelStructBuildInput = true;
        nvrhi::BufferHandle indexBuffer = GetDevice()->createBuffer(bufferDesc);
        bufferDesc.byteSize = sizeof(float3) * 3;
        nvrhi::BufferHandle vertexBuffer = GetDevice()->createBuffer(bufferDesc);

        uint indices[3] = { 0, 1, 2 };
        m_CommandList->writeBuffer(indexBuffer, indices, sizeof(indices));
        float3 vertices[3] = { float3(0, -1, 1), float3(-1, 1, 1), float3(1, 1, 1) };
        m_CommandList->writeBuffer(vertexBuffer, vertices, sizeof(vertices));

        nvrhi::rt::AccelStructDesc blasDesc;
        blasDesc.isTopLevel = false;
        nvrhi::rt::GeometryDesc geometryDesc;
        auto& triangles = geometryDesc.geometryData.triangles;
        triangles.indexBuffer = indexBuffer;
        triangles.vertexBuffer = vertexBuffer;
        triangles.indexFormat = nvrhi::Format::R32_UINT;
        triangles.indexCount = 3;
        triangles.vertexFormat = nvrhi::Format::RGB32_FLOAT;
        triangles.vertexStride = sizeof(float3);
        triangles.vertexCount = 3;
        geometryDesc.geometryType = nvrhi::rt::GeometryType::Triangles;
        geometryDesc.flags = nvrhi::rt::GeometryFlags::Opaque;
        blasDesc.bottomLevelGeometries.push_back(geometryDesc);

        m_BottomLevelAS = GetDevice()->createAccelStruct(blasDesc);
        nvrhi::utils::BuildBottomLevelAccelStruct(m_CommandList, m_BottomLevelAS, blasDesc);

        nvrhi::rt::AccelStructDesc tlasDesc;
        tlasDesc.isTopLevel = true;
        tlasDesc.topLevelMaxInstances = 1;
        
        m_TopLevelAS = GetDevice()->createAccelStruct(tlasDesc);

        nvrhi::rt::InstanceDesc instanceDesc;
        instanceDesc.bottomLevelAS = m_BottomLevelAS;
        instanceDesc.instanceMask = 1;
        instanceDesc.flags = nvrhi::rt::InstanceFlags::TriangleFrontCounterclockwise;
        float3x4 transform = float3x4::identity();
        memcpy(instanceDesc.transform, &transform, sizeof(transform));

        m_CommandList->buildTopLevelAccelStruct(m_TopLevelAS, &instanceDesc, 1);

        m_CommandList->close();
        GetDevice()->executeCommandList(m_CommandList);

        return true;
    }

    void Animate(float fElapsedTimeSeconds) override
    {
        GetDeviceManager()->SetInformativeWindowTitle(g_WindowTitle);
    }

    void BackBufferResizing() override
    { 
        m_RenderTarget = nullptr;
        m_BindingCache->Clear();
    }

    void Render(nvrhi::IFramebuffer* framebuffer) override
    {
        if (!m_RenderTarget)
        {
            nvrhi::TextureDesc textureDesc = framebuffer->getDesc().colorAttachments[0].texture->getDesc();
            textureDesc.isUAV = true;
            textureDesc.isRenderTarget = false;
            textureDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
            textureDesc.keepInitialState = true;
            textureDesc.format = nvrhi::Format::RGBA8_UNORM;
            m_RenderTarget = GetDevice()->createTexture(textureDesc);

            nvrhi::BindingSetDesc bindingSetDesc;
            bindingSetDesc.bindings = {
                nvrhi::BindingSetItem::RayTracingAccelStruct(0, m_TopLevelAS),
                nvrhi::BindingSetItem::Texture_UAV(0, m_RenderTarget)
            };

            m_BindingSet = GetDevice()->createBindingSet(bindingSetDesc, m_BindingLayout);
        }

        const auto& fbinfo = framebuffer->getFramebufferInfo();

        m_CommandList->open();

        nvrhi::rt::State state;
        state.shaderTable = m_ShaderTable;
        state.bindings = { m_BindingSet };
        m_CommandList->setRayTracingState(state);

        nvrhi::rt::DispatchRaysArguments args;
        args.width = fbinfo.width;
        args.height = fbinfo.height;
        m_CommandList->dispatchRays(args);

        nvrhi::Viewport windowViewport(float(fbinfo.width), float(fbinfo.height));
        m_CommonPasses->BlitTexture(m_CommandList, framebuffer, m_RenderTarget, m_BindingCache.get());
        
        m_CommandList->close();
        GetDevice()->executeCommandList(m_CommandList);
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
#ifdef _DEBUG
    deviceParams.enableDebugRuntime = true; 
    deviceParams.enableNvrhiValidationLayer = true;
#endif

    if (!deviceManager->CreateWindowDeviceAndSwapChain(deviceParams, g_WindowTitle))
    {
        log::fatal("Cannot initialize a graphics device with the requested parameters");
        return 1;
    }

    if (!deviceManager->GetDevice()->queryFeatureSupport(nvrhi::Feature::RayTracingPipeline))
    {
        log::fatal("The graphics device does not support Ray Tracing Pipelines");
        return 1;
    }
    
    {
        RayTracedTriangle example(deviceManager);
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
