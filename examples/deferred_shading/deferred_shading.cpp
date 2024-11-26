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
#include <donut/app/DeviceManager.h>
#include <donut/core/log.h>
#include <donut/core/vfs/VFS.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/engine/SceneGraph.h>
#include <donut/engine/FramebufferFactory.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/TextureCache.h>
#include <donut/render/DeferredLightingPass.h>
#include <donut/render/GBuffer.h>
#include <donut/render/GBufferFillPass.h>
#include <donut/render/DrawStrategy.h>

using namespace donut;
using namespace donut::math;
using namespace donut::engine;
using namespace donut::render;

#include <donut/shaders/material_cb.h>
#include <donut/shaders/bindless.h>

#include "CubeGeometry.h"

static const char* g_WindowTitle = "Donut Example: Deferred Shading";

class RenderTargets : public GBufferRenderTargets
{
public:
    nvrhi::TextureHandle ShadedColor;

    void Init(
        nvrhi::IDevice* device,
        dm::uint2 size,
        dm::uint sampleCount,
        bool enableMotionVectors,
        bool useReverseProjection) override
    {
        GBufferRenderTargets::Init(device, size, sampleCount, enableMotionVectors, useReverseProjection);

        nvrhi::TextureDesc textureDesc;
        textureDesc.dimension = nvrhi::TextureDimension::Texture2D;
        textureDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
        textureDesc.keepInitialState = true;
        textureDesc.debugName = "ShadedColor";
        textureDesc.isUAV = true;
        textureDesc.format = nvrhi::Format::RGBA16_FLOAT;
        textureDesc.width = size.x;
        textureDesc.height = size.y;
        textureDesc.sampleCount = sampleCount;
        ShadedColor = device->createTexture(textureDesc);
    }
};

class SimpleScene
{
public:

    bool Init(nvrhi::IDevice* device, nvrhi::ICommandList* commandList, TextureCache* textureCache)
    {
        commandList->open();

        m_Buffers = std::make_shared<BufferGroup>();
        m_Buffers->indexBuffer = CreateGeometryBuffer(device, commandList, "IndexBuffer", g_Indices, sizeof(g_Indices), false, false);

        uint64_t vertexBufferSize = 0;
        m_Buffers->getVertexBufferRange(VertexAttribute::Position).setByteOffset(vertexBufferSize).setByteSize(sizeof(g_Positions)); vertexBufferSize += sizeof(g_Positions);
        m_Buffers->getVertexBufferRange(VertexAttribute::TexCoord1).setByteOffset(vertexBufferSize).setByteSize(sizeof(g_TexCoords)); vertexBufferSize += sizeof(g_TexCoords);
        m_Buffers->getVertexBufferRange(VertexAttribute::Normal).setByteOffset(vertexBufferSize).setByteSize(sizeof(g_Normals)); vertexBufferSize += sizeof(g_Normals);
        m_Buffers->getVertexBufferRange(VertexAttribute::Tangent).setByteOffset(vertexBufferSize).setByteSize(sizeof(g_Tangents)); vertexBufferSize += sizeof(g_Tangents);
        m_Buffers->vertexBuffer = CreateGeometryBuffer(device, commandList, "VertexBuffer", nullptr, vertexBufferSize, true, false);

        commandList->beginTrackingBufferState(m_Buffers->vertexBuffer, nvrhi::ResourceStates::CopyDest);
        commandList->writeBuffer(m_Buffers->vertexBuffer, g_Positions, sizeof(g_Positions), m_Buffers->getVertexBufferRange(VertexAttribute::Position).byteOffset);
        commandList->writeBuffer(m_Buffers->vertexBuffer, g_TexCoords, sizeof(g_TexCoords), m_Buffers->getVertexBufferRange(VertexAttribute::TexCoord1).byteOffset);
        commandList->writeBuffer(m_Buffers->vertexBuffer, g_Normals, sizeof(g_Normals), m_Buffers->getVertexBufferRange(VertexAttribute::Normal).byteOffset);
        commandList->writeBuffer(m_Buffers->vertexBuffer, g_Tangents, sizeof(g_Tangents), m_Buffers->getVertexBufferRange(VertexAttribute::Tangent).byteOffset);
        commandList->setPermanentBufferState(m_Buffers->vertexBuffer, nvrhi::ResourceStates::ShaderResource);
        
        InstanceData instance{};
        instance.transform = math::float3x4(transpose(math::affineToHomogeneous(math::affine3::identity())));
        instance.prevTransform = instance.transform;
        m_Buffers->instanceBuffer = CreateGeometryBuffer(device, commandList, "VertexBufferTransform", &instance, sizeof(instance), false, true);

        std::filesystem::path textureFileName = app::GetDirectoryWithExecutable().parent_path() / "media/nvidia-logo.png";

        m_Material = std::make_shared<Material>();
        m_Material->name = "CubeMaterial";
        m_Material->useSpecularGlossModel = true;
        m_Material->enableBaseOrDiffuseTexture = true;
        m_Material->baseOrDiffuseTexture = textureCache->LoadTextureFromFile(textureFileName, true, nullptr, commandList);
        m_Material->materialConstants = CreateMaterialConstantBuffer(device, commandList, m_Material);

        commandList->close();
        device->executeCommandList(commandList);

        if (!m_Material->baseOrDiffuseTexture->texture)
        {
            log::error("Couldn't load the texture");
            return false;
        }

        auto geometry = std::make_shared<MeshGeometry>();
        geometry->material = m_Material;
        geometry->numIndices = dim(g_Indices);
        geometry->numVertices = dim(g_Positions);

        m_MeshInfo = std::make_shared<MeshInfo>();
        m_MeshInfo->name = "CubeMesh";
        m_MeshInfo->buffers = m_Buffers;
        m_MeshInfo->objectSpaceBounds = math::box3(math::float3(-0.5f), math::float3(0.5f));;
        m_MeshInfo->totalIndices = geometry->numIndices;
        m_MeshInfo->totalVertices = geometry->numVertices;
        m_MeshInfo->geometries.push_back(geometry);

        m_SceneGraph = std::make_shared<SceneGraph>();
        auto node = std::make_shared<SceneGraphNode>();
        m_SceneGraph->SetRootNode(node);

        m_MeshInstance = std::make_shared<MeshInstance>(m_MeshInfo);
        node->SetLeaf(m_MeshInstance);
        node->SetName("CubeNode");

        std::shared_ptr<DirectionalLight> sunLight = std::make_shared<DirectionalLight>();
        m_SceneGraph->AttachLeafNode(node, sunLight);

        sunLight->SetDirection(double3(0.1, -1.0, 0.2));
        sunLight->angularSize = 0.53f;
        sunLight->irradiance = 1.f;
        sunLight->SetName("Sun");

        m_SceneGraph->Refresh(0);

        PrintSceneGraph(m_SceneGraph->GetRootNode());

        return true;
    }

    const std::shared_ptr<MeshInstance>& GetMeshInstance() const
    {
        return m_MeshInstance;
    }

    const std::shared_ptr<SceneGraph>& GetSceneGraph() const
    {
        return m_SceneGraph;
    }
    
    const std::vector<std::shared_ptr<Light>>& GetLights() const
    {
        return m_SceneGraph->GetLights();
    }

private:
    std::shared_ptr<BufferGroup> m_Buffers;
    std::shared_ptr<Material> m_Material;
    std::shared_ptr<MeshInfo> m_MeshInfo;
    std::shared_ptr<MeshInstance> m_MeshInstance;
    std::shared_ptr<SceneGraph> m_SceneGraph;

    nvrhi::BufferHandle CreateGeometryBuffer(nvrhi::IDevice* device, nvrhi::ICommandList* commandList,
        const char* debugName, const void* data, uint64_t dataSize, bool isVertexBuffer, bool isInstanceBuffer)
    {
        nvrhi::BufferHandle bufHandle;

        // The G-buffer fill pass accesses instance buffers as structured on DX12 and Vulkan, and as raw on DX11.
        bool const needStructuredBuffer = isInstanceBuffer && device->getGraphicsAPI() != nvrhi::GraphicsAPI::D3D11;
        
        nvrhi::BufferDesc desc;
        desc.byteSize = dataSize;
        desc.isIndexBuffer = !isVertexBuffer && !isInstanceBuffer;
        desc.canHaveRawViews = isVertexBuffer || isInstanceBuffer;
        desc.structStride = needStructuredBuffer ? sizeof(InstanceData) : 0;
        desc.debugName = debugName;
        desc.initialState = nvrhi::ResourceStates::CopyDest;
        bufHandle = device->createBuffer(desc);

        if (data)
        {
            commandList->beginTrackingBufferState(bufHandle, nvrhi::ResourceStates::CopyDest);
            commandList->writeBuffer(bufHandle, data, dataSize);
            commandList->setPermanentBufferState(bufHandle, (isVertexBuffer || isInstanceBuffer)
                ? nvrhi::ResourceStates::ShaderResource
                : nvrhi::ResourceStates::IndexBuffer);
        }

        return bufHandle;
    }

    nvrhi::BufferHandle CreateMaterialConstantBuffer(nvrhi::IDevice* device, nvrhi::ICommandList* commandList, const std::shared_ptr<Material> material)
    {
        nvrhi::BufferDesc bufferDesc;
        bufferDesc.byteSize = sizeof(MaterialConstants);
        bufferDesc.debugName = material->name;
        bufferDesc.isConstantBuffer = true;
        bufferDesc.initialState = nvrhi::ResourceStates::ConstantBuffer;
        bufferDesc.keepInitialState = true;
        nvrhi::BufferHandle buffer = device->createBuffer(bufferDesc);

        MaterialConstants constants;
        material->FillConstantBuffer(constants);
        commandList->writeBuffer(buffer, &constants, sizeof(constants));

        return buffer;
    }
};

class DeferredShading : public app::IRenderPass
{
private:
    std::shared_ptr<ShaderFactory> m_ShaderFactory;
    std::shared_ptr<TextureCache> m_TextureCache;
    std::shared_ptr<CommonRenderPasses> m_CommonPasses;
    std::unique_ptr<engine::BindingCache> m_BindingCache;

    std::shared_ptr<RenderTargets> m_RenderTargets;
    std::unique_ptr<GBufferFillPass> m_GBufferPass;
    std::unique_ptr<DeferredLightingPass> m_DeferredLightingPass;
    
    PlanarView m_View;

    SimpleScene m_Scene;

    nvrhi::CommandListHandle m_CommandList;
    float m_Rotation = 0.f;

public:
    using IRenderPass::IRenderPass;

    void SetupView()
    {
        float2 renderTargetSize = float2(m_RenderTargets->GetSize());

        math::affine3 viewMatrix = math::yawPitchRoll(m_Rotation, 0.f, 0.f)
            * math::yawPitchRoll(0.f, math::radians(-30.f), 0.f)
            * math::translation(math::float3(0, 0, 2));

        float4x4 projection = math::perspProjD3DStyle(math::radians(60.f), renderTargetSize.x / renderTargetSize.y, 0.1f, 10.f);

        m_View.SetViewport(nvrhi::Viewport(renderTargetSize.x, renderTargetSize.y));
        m_View.SetMatrices(viewMatrix, projection);
        m_View.UpdateCache();
    }
    
    bool Init()
    {
        auto nativeFS = std::make_shared<vfs::NativeFileSystem>();

        std::filesystem::path frameworkShaderPath = app::GetDirectoryWithExecutable() / "shaders/framework" / app::GetShaderTypeName(GetDevice()->getGraphicsAPI());
        
        std::shared_ptr<vfs::RootFileSystem> rootFS = std::make_shared<vfs::RootFileSystem>();
        rootFS->mount("/shaders/donut", frameworkShaderPath);
        m_ShaderFactory = std::make_shared<ShaderFactory>(GetDevice(), rootFS, "/shaders");
        m_CommonPasses = std::make_shared<CommonRenderPasses>(GetDevice(), m_ShaderFactory);
        m_BindingCache = std::make_unique<engine::BindingCache>(GetDevice());

        m_DeferredLightingPass = std::make_unique<DeferredLightingPass>(GetDevice(), m_CommonPasses);
        m_DeferredLightingPass->Init(m_ShaderFactory);

        m_TextureCache = std::make_shared<TextureCache>(GetDevice(), nativeFS, nullptr);

        m_CommandList = GetDevice()->createCommandList();

        return m_Scene.Init(GetDevice(), m_CommandList, m_TextureCache.get());
    }

    void Animate(float seconds) override
    {
        m_Rotation += seconds * 1.1f;
        GetDeviceManager()->SetInformativeWindowTitle(g_WindowTitle);
    }

    void BackBufferResizing() override
    {

    }

    void Render(nvrhi::IFramebuffer* framebuffer) override
    {
        const nvrhi::FramebufferInfoEx& fbinfo = framebuffer->getFramebufferInfo();

        uint2 size = uint2(fbinfo.width, fbinfo.height);

        if (!m_RenderTargets || any(m_RenderTargets->GetSize() != size))
        {
            m_RenderTargets = nullptr;
            m_BindingCache->Clear();
            m_DeferredLightingPass->ResetBindingCache();

            m_GBufferPass.reset();

            m_RenderTargets = std::make_shared<RenderTargets>();
            m_RenderTargets->Init(GetDevice(), size, 1, false, false);
        }

        SetupView();
        
        if (!m_GBufferPass)
        {
            GBufferFillPass::CreateParameters GBufferParams;
            m_GBufferPass = std::make_unique<GBufferFillPass>(GetDevice(), m_CommonPasses);
            m_GBufferPass->Init(*m_ShaderFactory, GBufferParams);
        }

        m_CommandList->open();

        m_RenderTargets->Clear(m_CommandList);

        render::DrawItem drawItem;
        drawItem.instance = m_Scene.GetMeshInstance().get();
        drawItem.mesh = drawItem.instance->GetMesh().get();
        drawItem.geometry = drawItem.mesh->geometries[0].get();
        drawItem.material = drawItem.geometry->material.get();
        drawItem.buffers = drawItem.mesh->buffers.get();
        drawItem.distanceToCamera = 0;
        drawItem.cullMode = nvrhi::RasterCullMode::Back;

        render::PassthroughDrawStrategy drawStrategy;
        drawStrategy.SetData(&drawItem, 1);

        GBufferFillPass::Context context;

        RenderView(
            m_CommandList,
            &m_View,
            &m_View,
            m_RenderTargets->GBufferFramebuffer->GetFramebuffer(m_View),
            drawStrategy,
            *m_GBufferPass,
            context,
            false);


        DeferredLightingPass::Inputs deferredInputs;
        deferredInputs.SetGBuffer(*m_RenderTargets);
        deferredInputs.ambientColorTop = 0.2f;
        deferredInputs.ambientColorBottom = deferredInputs.ambientColorTop * float3(0.3f, 0.4f, 0.3f);
        deferredInputs.lights = &m_Scene.GetLights();
        deferredInputs.output = m_RenderTargets->ShadedColor;
        
        m_DeferredLightingPass->Render(m_CommandList, m_View, deferredInputs);

        m_CommonPasses->BlitTexture(m_CommandList, framebuffer, m_RenderTargets->ShadedColor, m_BindingCache.get());

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
#ifdef _DEBUG
    deviceParams.enableDebugRuntime = true; 
    deviceParams.enableNvrhiValidationLayer = true;
#endif

    if (!deviceManager->CreateWindowDeviceAndSwapChain(deviceParams, g_WindowTitle))
    {
        log::fatal("Cannot initialize a graphics device with the requested parameters");
        return 1;
    }

    {
        DeferredShading example(deviceManager);
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
