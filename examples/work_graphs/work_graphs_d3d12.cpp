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

#include "d3dx12/d3dx12.h"
#include <donut/app/ApplicationBase.h>
#include <donut/engine/FramebufferFactory.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/app/DeviceManager.h>
#include <donut/app/imgui_renderer.h>
#include <donut/core/log.h>
#include <donut/core/vfs/VFS.h>
#include <donut/core/math/math.h>
#include <wrl.h>
#include "scene.h"

using namespace donut;
using namespace donut::app;
using namespace donut::engine;
using namespace donut::math;
using Microsoft::WRL::ComPtr;

static const char* g_WindowTitle = "Donut Example: Work Graphs";
#define WORKGRAPH_NAME L"D3D12WorkGraphs"


// Constants used by deferred shading. Ensure these values are matched with the shaders.
static const uint32_t DeferredShadingParam_MaxLightsPerTile = 64; // If changed, make sure to also change the constant c_MaxLightsPerTile in lighting.hlsli
static const uint32_t DeferredShadingParam_TileWidth = 8;
static const uint32_t DeferredShadingParam_TileHeight = 4;

// Simulation and camera control constants.
static const float Animation_SpeedMultiplier = 1.0f;
static const float Camera_PositionOrbitSpeed = 0.1f;
static const float Camera_TargetOrbitSpeed = 0.03f;
static const float Camera_PositionRadiusRatio = 0.75f;
static const float Camera_TargetRadiusRatio = 0.1f;
static const float Camera_ClimbSpeed = 0.1f;
static const float Camera_ClimbRatio = 0.6f;
static const float Camera_VerticalFOV = (dm::PI_f/4.0f)*1.15f; // In radians.
static const float Camera_NearClipDistance = 0.5f;


struct UIData
{
    bool ShowUI = true;
    int CurrentTechnique = 0;
    bool Paused = false;
    bool ResetAnim = false;
    float GPUFrameTime = 0.0f;
    float GPUShadingTime = 0.0f;
};


struct RenderTargets
{
    nvrhi::TextureHandle m_Depth;
    nvrhi::TextureHandle m_LDRBuffer;
    nvrhi::TextureHandle m_GBuffer;
    nvrhi::IFramebuffer* m_FrameBufferGB;
    std::shared_ptr<engine::FramebufferFactory> m_GBufferDepth;

    int2 m_Size;

    RenderTargets(nvrhi::IDevice* device, int2 size)
        : m_Size(size)
    {
        nvrhi::TextureDesc desc;
        desc.width = size.x;
        desc.height = size.y;
        desc.keepInitialState = true;

        // Depth buffer
        desc.useClearValue = true;
        desc.clearValue = nvrhi::Color(1.f);
        desc.isRenderTarget = true;
        desc.isTypeless = true;
        desc.format = nvrhi::Format::D32;
        desc.initialState = nvrhi::ResourceStates::ShaderResource;
        desc.debugName = "DepthBuffer";
        m_Depth = device->createTexture(desc);

        // G buffer
        desc.format = nvrhi::Format::RGBA16_UINT;
        desc.clearValue = nvrhi::Color(0.f);
        desc.useClearValue = false;
        desc.isTypeless = false;
        desc.initialState = nvrhi::ResourceStates::ShaderResource;
        desc.debugName = "GBuffer";
        m_GBuffer = device->createTexture(desc);

        // LDR buffer
        desc.format = nvrhi::Format::RGBA8_UNORM;
        desc.isRenderTarget = false;
        desc.isUAV = true;
        desc.initialState = nvrhi::ResourceStates::UnorderedAccess;
        desc.debugName = "LDRBuffer";
        m_LDRBuffer = device->createTexture(desc);

        m_GBufferDepth = std::make_shared<engine::FramebufferFactory>(device);
        m_GBufferDepth->RenderTargets = { m_GBuffer };
        m_GBufferDepth->DepthTarget = m_Depth;

        m_FrameBufferGB = m_GBufferDepth->GetFramebuffer(nvrhi::TextureSubresourceSet());
    }

    bool IsUpdateRequired(int2 size) const { return any(m_Size != size); }
};


class WorkGraphs : public app::IRenderPass
{
private:
    enum class ScenePass
    {
        AnimateObjects,
        AnimateLights,
        GBufferFill,
        LightCulling,
        DeferredShading,
        WorkGraph,

        COUNT
    };

    enum class Techniques
    {
        WorkGraphBroadcastingLaunch,
        Dispatch,

        COUNT,
    };

    std::unique_ptr<RenderTargets> m_RenderTargets;
    nvrhi::InputLayoutHandle m_InputLayout;
    nvrhi::BindingLayoutHandle m_BindingLayout;
    nvrhi::BindingSetHandle m_BindingSets[(int)ScenePass::COUNT];

    Scene m_Scene;
    nvrhi::CommandListHandle m_CommandList;

    // Pipeline state objects.
    nvrhi::ComputePipelineHandle m_AnimateObjectsPSO;
    nvrhi::ComputePipelineHandle m_AnimateLightsPSO;
    nvrhi::GraphicsPipelineHandle m_GBufferFillPSO;
    nvrhi::ComputePipelineHandle m_CullLightsPSO;
    nvrhi::ComputePipelineHandle m_ShadePSO;

    // Work graph objects.
    ComPtr<ID3D12StateObject> m_WorkGraphBroadcastingSO;

    D3D12_PROGRAM_IDENTIFIER m_workGraphBroadcastingIdentifier;

    nvrhi::BufferHandle m_WorkGraphBackingMemory;

    // Resources.
    nvrhi::BufferHandle m_ConstantBuffer;
    nvrhi::BufferHandle m_CulledLightsBuffer;

    nvrhi::BufferHandle m_NullSRVBuffer;
    nvrhi::BufferHandle m_NullUAVBuffer;
    nvrhi::TextureHandle m_NullSRVTexture;
    nvrhi::TextureHandle m_NullUAVTexture;

    // State.
    Techniques m_CurrentTechnique = Techniques::WorkGraphBroadcastingLaunch;
    bool m_InitWorkGraphBackingMemory = true;
    UIData& m_UI;

    // Timing.
    static const uint32_t QueuedFramesCount = 10;
    nvrhi::TimerQueryHandle m_FrameTimers[QueuedFramesCount];
    nvrhi::TimerQueryHandle m_ShadingTimers[QueuedFramesCount];
    int m_NextTimerToUse = 0;
    float m_TimeInSeconds = 0.0f;
    float m_TimeDiffThisFrame = 0.0f;
    bool m_ForceResetAnimation = true;

    // Constant buffer definition.
    struct SceneConstantBuffer
    {
        float4x4 viewProj;
        float4x4 viewProjInverse;
        float4 camPosAndSceneTime;
        float4 camDir;
        float4 viewportSizeXY;

        // Constant buffers are 256-byte aligned. Add padding in the struct to allow multiple buffers
        // to be array-indexed.
        float padding[20];
    };

    // Utility functions.
    static inline bool HRSuccess(HRESULT hr) { assert(SUCCEEDED(hr)); return SUCCEEDED(hr); }
    static inline D3D12_SHADER_BYTECODE getShaderLibD3D12Bytecode(const nvrhi::ShaderLibraryHandle& shaderLib)
    {
        D3D12_SHADER_BYTECODE bc = {};
        shaderLib->getBytecode(&bc.pShaderBytecode, &bc.BytecodeLength);
        return bc;
    };
    static inline uint32_t GetLightTileCountX(uint32_t viewportWidth) { return (viewportWidth+DeferredShadingParam_TileWidth-1)/DeferredShadingParam_TileWidth; };
    static inline uint32_t GetLightTileCountY(uint32_t viewportHeight) { return (viewportHeight+DeferredShadingParam_TileHeight-1)/DeferredShadingParam_TileHeight; };
    static inline uint32_t GetLightTileCount(uint32_t viewportWidth, uint32_t viewportHeight) { return GetLightTileCountX(viewportWidth) * GetLightTileCountY(viewportHeight); };
    float GetLastValidQueryTimer(nvrhi::TimerQueryHandle timers[QueuedFramesCount])
    {
        nvrhi::IDevice *device = GetDevice();
        for (int i=m_NextTimerToUse-1;i>=0;i--)
        {
            if (device->pollTimerQuery(timers[i]))
                return device->getTimerQueryTime(timers[i])*1000.0f;
        }

        for (int i=QueuedFramesCount-1;i>m_NextTimerToUse;i--)
        {
            if (device->pollTimerQuery(timers[i]))
                return device->getTimerQueryTime(timers[i])*1000.0f;
        }
        return -1.0f;
    }
    static inline float4x4 lookToD3DStyle(const float3& eyePosition, const float3& focusPosition, const float3& upDirection)
    {
        float3 eyeDirection = focusPosition - eyePosition;
        float3 negEyePosition = -eyePosition;
        float3 z = normalize(eyeDirection);
        float3 x = normalize(cross(upDirection, z));
        float3 y = cross(z, x);

        float4x4 m;
        m.row0 = float4(x, dot(x, negEyePosition));
        m.row1 = float4(y, dot(y, negEyePosition));
        m.row2 = float4(z, dot(z, negEyePosition));
        m.row3 = float4(0,0,0,1);
        return transpose(m);
    }

public:
    using IRenderPass::IRenderPass;

    WorkGraphs(DeviceManager* deviceManager, UIData& ui) :
        IRenderPass(deviceManager),
        m_UI(ui)
    {}

    bool Init()
    {
        ID3D12Device *deviceD3D12 = GetDevice()->getNativeObject(nvrhi::ObjectTypes::D3D12_Device);

        // Check for device support for work graphs.
        D3D12_FEATURE_DATA_D3D12_OPTIONS21 options = {};
        if (!HRSuccess(deviceD3D12->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS21, &options, sizeof(options))))
        {
            log::fatal("Failed to check D3D12 feature support for work graphs");
            return false;
        }
        if (options.WorkGraphsTier == D3D12_WORK_GRAPHS_TIER_NOT_SUPPORTED)
        {
            log::fatal("D3D12 device reports it has no support for work graphs. This sample cannot run.\n"
                "Please make sure you download the latest graphics driver with support for work graphs, "
                "and that the hardware does support this feature.");
            return false;
        }

        m_CommandList = GetDevice()->createCommandList();

        // Resources used to fill unused shader binding slots (null resources).
        m_NullSRVBuffer = GetDevice()->createBuffer(nvrhi::BufferDesc()
            .setByteSize(512).setStructStride(16).setKeepInitialState(true)
            .setInitialState(nvrhi::ResourceStates::ShaderResource).setDebugName("NullSRVBuffer"));
        m_NullUAVBuffer = GetDevice()->createBuffer(nvrhi::BufferDesc()
            .setByteSize(512).setStructStride(16).setKeepInitialState(true)
            .setInitialState(nvrhi::ResourceStates::UnorderedAccess).setCanHaveUAVs(true).setDebugName("NullUAVBuffer"));
        m_NullSRVTexture = GetDevice()->createTexture(nvrhi::TextureDesc()
            .setFormat(nvrhi::Format::RGBA8_UNORM).setKeepInitialState(true)
            .setInitialState(nvrhi::ResourceStates::ShaderResource).setDebugName("NullSRVTexture"));
        m_NullUAVTexture = GetDevice()->createTexture(nvrhi::TextureDesc()
            .setFormat(nvrhi::Format::RGBA8_UNORM).setKeepInitialState(true)
            .setInitialState(nvrhi::ResourceStates::UnorderedAccess).setIsUAV(true).setDebugName("NullUAVTexture"));

        for (uint32_t i=0; i<QueuedFramesCount; i++)
        {
            m_FrameTimers[i] = GetDevice()->createTimerQuery();
            m_ShadingTimers[i] = GetDevice()->createTimerQuery();
        }
        
        // Create the scene procedurally.
        m_CommandList->open();
        m_Scene.CreateAssets(GetDevice(), m_CommandList);
        m_CommandList->close();
        GetDevice()->executeCommandList(m_CommandList);
        GetDevice()->waitForIdle();

        return true;
    }

    bool LoadScenePipelines(nvrhi::IFramebuffer* gBufferFramebuffer,nvrhi::IFramebuffer* backBufferFramebuffer)
    {
        std::filesystem::path appShaderPath = app::GetDirectoryWithExecutable() / "shaders/work_graphs_d3d12" /  app::GetShaderTypeName(GetDevice()->getGraphicsAPI());
        
        auto nativeFS = std::make_shared<vfs::NativeFileSystem>();
        engine::ShaderFactory shaderFactory(GetDevice(), nativeFS, appShaderPath);

        nvrhi::ShaderHandle animateObjects_computeShader = shaderFactory.CreateShader("animation.hlsl", "CSMainObjects", nullptr, nvrhi::ShaderType::Compute);
        nvrhi::ShaderHandle animateLights_computeShader = shaderFactory.CreateShader("animation.hlsl", "CSMainLights", nullptr, nvrhi::ShaderType::Compute);
        nvrhi::ShaderHandle gbuffer_vertexShader = shaderFactory.CreateShader("gbuffer_fill.hlsl", "VSMain", nullptr, nvrhi::ShaderType::Vertex);
        nvrhi::ShaderHandle gbuffer_pixelShader = shaderFactory.CreateShader("gbuffer_fill.hlsl", "PSMain", nullptr, nvrhi::ShaderType::Pixel);
        nvrhi::ShaderHandle lightCulling_computeShader = shaderFactory.CreateShader("light_culling.hlsl", "CSMain", nullptr, nvrhi::ShaderType::Compute);
        nvrhi::ShaderHandle deferredShading_computeShader = shaderFactory.CreateShader("deferred_shading.hlsl", "CSMain", nullptr, nvrhi::ShaderType::Compute);

        if (!animateObjects_computeShader || !animateLights_computeShader ||
            !gbuffer_vertexShader || !gbuffer_pixelShader ||
            !lightCulling_computeShader || !deferredShading_computeShader)
        {
            return false;
        }

		auto bindingLayoutDesc = nvrhi::BindingLayoutDesc()
            .setRegisterSpace(0)
            .setVisibility(nvrhi::ShaderType::All)
            .addItem(nvrhi::BindingLayoutItem::PushConstants(0,sizeof(int3)))
            .addItem(nvrhi::BindingLayoutItem::VolatileConstantBuffer(1))
			.addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(0))
			.addItem(nvrhi::BindingLayoutItem::Texture_SRV(1))
			.addItem(nvrhi::BindingLayoutItem::Texture_SRV(2))
			.addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(3))
			.addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(4))
			.addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(0))
            .addItem(nvrhi::BindingLayoutItem::Texture_UAV(1));
        m_BindingLayout = GetDevice()->createBindingLayout(bindingLayoutDesc);

        nvrhi::VertexAttributeDesc attributes[] = {
            nvrhi::VertexAttributeDesc()
                .setName("POSITION")
                .setFormat(nvrhi::Format::RGB32_FLOAT)
                .setOffset(0)
                .setElementStride(sizeof(float3)*2),
            nvrhi::VertexAttributeDesc()
                .setName("NORMAL")
                .setFormat(nvrhi::Format::RGB32_FLOAT)
                .setOffset(sizeof(float3))
                .setElementStride(sizeof(float3)*2),
            };
        m_InputLayout = GetDevice()->createInputLayout(attributes, uint32_t(std::size(attributes)), gbuffer_vertexShader);

        // Create pipeine states
        {
            nvrhi::GraphicsPipelineDesc psoGfxDesc;
            psoGfxDesc.inputLayout = m_InputLayout;
            psoGfxDesc.bindingLayouts = { m_BindingLayout };
            psoGfxDesc.VS = gbuffer_vertexShader;
            psoGfxDesc.PS = gbuffer_pixelShader;

            m_GBufferFillPSO = GetDevice()->createGraphicsPipeline(psoGfxDesc, gBufferFramebuffer);
        }

        nvrhi::ComputePipelineDesc psoCSDesc;
        psoCSDesc.bindingLayouts = { m_BindingLayout };

        m_AnimateObjectsPSO = GetDevice()->createComputePipeline(psoCSDesc.setComputeShader(animateObjects_computeShader));
        m_AnimateLightsPSO = GetDevice()->createComputePipeline(psoCSDesc.setComputeShader(animateLights_computeShader));
        m_CullLightsPSO = GetDevice()->createComputePipeline(psoCSDesc.setComputeShader(lightCulling_computeShader));
        m_ShadePSO = GetDevice()->createComputePipeline(psoCSDesc.setComputeShader(deferredShading_computeShader));

        // Create the culled lights buffer.
        {
            uint2 framebufferSize = uint2(gBufferFramebuffer->getFramebufferInfo().width, gBufferFramebuffer->getFramebufferInfo().height);
            const uint32_t tileCount = GetLightTileCount(framebufferSize.x, framebufferSize.y);

            nvrhi::BufferDesc bufferDesc;
            bufferDesc.byteSize = tileCount * DeferredShadingParam_MaxLightsPerTile * sizeof(UINT32);
            bufferDesc.structStride = sizeof(UINT32);
            bufferDesc.canHaveUAVs = true;
            bufferDesc.debugName = "CulledLights";
            bufferDesc.initialState = nvrhi::ResourceStates::ShaderResource;
            bufferDesc.keepInitialState = true;
            m_CulledLightsBuffer = GetDevice()->createBuffer(bufferDesc);
        }

        // Create the constant buffer.
        {
            nvrhi::BufferDesc bufferDesc;
            bufferDesc.byteSize = sizeof(SceneConstantBuffer);
            bufferDesc.maxVersions = 16;
            bufferDesc.isConstantBuffer = true;
            bufferDesc.isVolatile = true;
            bufferDesc.debugName = "SceneConstants";
            bufferDesc.initialState = nvrhi::ResourceStates::ShaderResource;
            bufferDesc.keepInitialState = true;
            m_ConstantBuffer = GetDevice()->createBuffer(bufferDesc);
        }

        // Create the resource binding sets for each pass. The resource registers must match with
        // assignments used in the shader files. Donut internally takes care of resource states and transition barriers.
        m_BindingSets[(int)ScenePass::AnimateObjects] = GetDevice()->createBindingSet(nvrhi::BindingSetDesc()
            .addItem(nvrhi::BindingSetItem::PushConstants(0, sizeof(uint3)))
            .addItem(nvrhi::BindingSetItem::ConstantBuffer(1, m_ConstantBuffer))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(0, m_Scene.GetWorldObjectsBuffer()))
            .addItem(nvrhi::BindingSetItem::Texture_SRV(1, m_NullSRVTexture))
            .addItem(nvrhi::BindingSetItem::Texture_SRV(2, m_NullSRVTexture))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(3, m_NullSRVBuffer))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(4, m_NullSRVBuffer))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(0, m_Scene.GetAnimStateBuffer()))
            .addItem(nvrhi::BindingSetItem::Texture_UAV(1, m_NullUAVTexture)),
            m_BindingLayout);

        m_BindingSets[(int)ScenePass::AnimateLights] = GetDevice()->createBindingSet(nvrhi::BindingSetDesc()
            .addItem(nvrhi::BindingSetItem::PushConstants(0, sizeof(uint3)))
            .addItem(nvrhi::BindingSetItem::ConstantBuffer(1, m_ConstantBuffer))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(0, m_NullSRVBuffer))
            .addItem(nvrhi::BindingSetItem::Texture_SRV(1, m_NullSRVTexture))
            .addItem(nvrhi::BindingSetItem::Texture_SRV(2, m_NullSRVTexture))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(3, m_NullSRVBuffer))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(4, m_NullSRVBuffer))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(0, m_Scene.GetLightsBuffer()))
            .addItem(nvrhi::BindingSetItem::Texture_UAV(1, m_NullUAVTexture)),
            m_BindingLayout);

        m_BindingSets[(int)ScenePass::GBufferFill] = GetDevice()->createBindingSet(nvrhi::BindingSetDesc()
            .addItem(nvrhi::BindingSetItem::PushConstants(0, sizeof(uint3)))
            .addItem(nvrhi::BindingSetItem::ConstantBuffer(1, m_ConstantBuffer))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(0, m_Scene.GetWorldObjectsBuffer()))
            .addItem(nvrhi::BindingSetItem::Texture_SRV(1, m_NullSRVTexture))
            .addItem(nvrhi::BindingSetItem::Texture_SRV(2, m_NullSRVTexture))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(3, m_Scene.GetMaterialsBuffer()))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(4, m_Scene.GetAnimStateBuffer()))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(0, m_NullUAVBuffer))
            .addItem(nvrhi::BindingSetItem::Texture_UAV(1, m_NullUAVTexture)),
            m_BindingLayout);

        m_BindingSets[(int)ScenePass::LightCulling] = GetDevice()->createBindingSet(nvrhi::BindingSetDesc()
            .addItem(nvrhi::BindingSetItem::PushConstants(0, sizeof(uint3)))
            .addItem(nvrhi::BindingSetItem::ConstantBuffer(1, m_ConstantBuffer))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(0, m_NullSRVBuffer))
            .addItem(nvrhi::BindingSetItem::Texture_SRV(1, m_RenderTargets->m_Depth))
            .addItem(nvrhi::BindingSetItem::Texture_SRV(2, m_NullSRVTexture))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(3, m_NullSRVBuffer))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(4, m_Scene.GetLightsBuffer()))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(0, m_CulledLightsBuffer))
            .addItem(nvrhi::BindingSetItem::Texture_UAV(1, m_NullUAVTexture)),
            m_BindingLayout);

        m_BindingSets[(int)ScenePass::DeferredShading] = GetDevice()->createBindingSet(nvrhi::BindingSetDesc()
            .addItem(nvrhi::BindingSetItem::PushConstants(0, sizeof(uint3)))
            .addItem(nvrhi::BindingSetItem::ConstantBuffer(1, m_ConstantBuffer))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(0, m_Scene.GetMaterialsBuffer()))
            .addItem(nvrhi::BindingSetItem::Texture_SRV(1, m_RenderTargets->m_GBuffer))
            .addItem(nvrhi::BindingSetItem::Texture_SRV(2, m_RenderTargets->m_Depth))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(3, m_CulledLightsBuffer))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(4, m_Scene.GetLightsBuffer()))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(0, m_NullUAVBuffer))
            .addItem(nvrhi::BindingSetItem::Texture_UAV(1, m_RenderTargets->m_LDRBuffer)),
            m_BindingLayout);

         m_BindingSets[(int)ScenePass::WorkGraph] = GetDevice()->createBindingSet(nvrhi::BindingSetDesc()
            .addItem(nvrhi::BindingSetItem::PushConstants(0, sizeof(uint3)))
            .addItem(nvrhi::BindingSetItem::ConstantBuffer(1, m_ConstantBuffer))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(0, m_Scene.GetMaterialsBuffer()))
            .addItem(nvrhi::BindingSetItem::Texture_SRV(1, m_RenderTargets->m_GBuffer))
            .addItem(nvrhi::BindingSetItem::Texture_SRV(2, m_RenderTargets->m_Depth))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(3, m_NullSRVBuffer))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(4, m_Scene.GetLightsBuffer()))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(0, m_NullUAVBuffer))
            .addItem(nvrhi::BindingSetItem::Texture_UAV(1, m_RenderTargets->m_LDRBuffer)),
            m_BindingLayout);

        // Animation state must be reset to good values before being updated every frame.
        m_ForceResetAnimation = true;
        return true;
    }

    bool LoadWorkGraphPipelines(nvrhi::IFramebuffer* framebuffer)
    {
        std::filesystem::path appShaderPath = app::GetDirectoryWithExecutable() / "shaders/work_graphs_d3d12" /  app::GetShaderTypeName(GetDevice()->getGraphicsAPI());
        
        auto nativeFS = std::make_shared<vfs::NativeFileSystem>();
        engine::ShaderFactory shaderFactory(GetDevice(), nativeFS, appShaderPath);

        // Compile the work graph shader library. The library represents a full work graph, and contains all node shaders for that graph.
        nvrhi::ShaderLibraryHandle workGraph_broadcasting_shaderLibrary = shaderFactory.CreateShaderLibrary("work_graph_broadcasting.hlsl", {});

        if (!workGraph_broadcasting_shaderLibrary)
            return false;

        ID3D12Device *device = GetDevice()->getNativeObject(nvrhi::ObjectTypes::D3D12_Device);
        ID3D12RootSignature *rootSignature = m_ShadePSO->getNativeObject(nvrhi::ObjectTypes::D3D12_RootSignature);
        uint2 framebufferSize = uint2(framebuffer->getFramebufferInfo().width, framebuffer->getFramebufferInfo().height);

        ComPtr<ID3D12Device5> deviceD3D12;
        device->QueryInterface(IID_PPV_ARGS(&deviceD3D12));
        if (!deviceD3D12)
        {
            log::fatal("Could not access the D3D12 device interface for work graphs");
            return false;
        }

        // A work graph is expressed in a single ID3D12StateObject. The state object requires several
        // pieces of information (sub-objects) besides the shader itself. It is possible that all the sub-objects
        // needed for creating the state object are already present in the compiled library, in which case
        // CreateStateObject will use those sub-objects automatically.
        // In this sample, the work graph is using a root signature object that is shared with all other shaders in the application.
        // Thus, we manually provide the root signature to the state object descriptor.
        // (The use of D3DX is optional. It simplifies code a lot for this demo).

        // State object descriptor for the work graph.
        CD3DX12_STATE_OBJECT_DESC soWorkGraphDesc(D3D12_STATE_OBJECT_TYPE_EXECUTABLE);

        // Add the first and main sub-object: the shader library.
        auto workGraphSubObj_Library = soWorkGraphDesc.CreateSubobject<CD3DX12_DXIL_LIBRARY_SUBOBJECT>();
        D3D12_SHADER_BYTECODE workGraph_LibCode = getShaderLibD3D12Bytecode(workGraph_broadcasting_shaderLibrary);
        workGraphSubObj_Library->SetDXILLibrary(&workGraph_LibCode);

        // Sub-object describing the work graph (name, and nodes used).
        auto workGraphSubObj_WorkGraph = soWorkGraphDesc.CreateSubobject<CD3DX12_WORK_GRAPH_SUBOBJECT>();
        workGraphSubObj_WorkGraph->SetProgramName(WORKGRAPH_NAME);
        workGraphSubObj_WorkGraph->IncludeAllAvailableNodes(); // Auto populate the graph

        // Provide the root signature.
        auto workGraphSubObj_RootSig = soWorkGraphDesc.CreateSubobject<CD3DX12_GLOBAL_ROOT_SIGNATURE_SUBOBJECT>();
        workGraphSubObj_RootSig->SetRootSignature(rootSignature);

        // The root node's dispatch grid size is hard-coded via a shader attribute. However, that value must
        // change according to the viewport size, which is determined by the application's window size.
        // It is possible to specify the dispatch grid size dynamically at launch time by making the root node
        // use SV_DispatchGrid in its input record. However, since this value only changes when the window is resized,
        // it is better to avoid the performance cost when using SV_DispatchGrid, and rely on overriding
        // the [NodeDispatchGrid()] attribute instead.
        auto rootNodeDispatchGridSizeOverride = workGraphSubObj_WorkGraph->CreateBroadcastingLaunchNodeOverrides(L"LightCull_Node");
        rootNodeDispatchGridSizeOverride->DispatchGrid(GetLightTileCountX(framebufferSize.x), GetLightTileCountY(framebufferSize.y), 1);

        // All sub-objects have been defined. Now create the state object.
        if (!HRSuccess(deviceD3D12->CreateStateObject(soWorkGraphDesc, IID_PPV_ARGS(&m_WorkGraphBroadcastingSO))))
            return false;

        // Readback the program identifier for use in the launch parameters.
        ComPtr<ID3D12StateObjectProperties1> soProperties;
        if (!HRSuccess(m_WorkGraphBroadcastingSO->QueryInterface(IID_PPV_ARGS(&soProperties))))
            return false;
        m_workGraphBroadcastingIdentifier = soProperties->GetProgramIdentifier(WORKGRAPH_NAME);

        // Get the broadcasting launch work graph's memory requirements.
        ComPtr<ID3D12WorkGraphProperties> workGraphProperties;
        if (!HRSuccess(m_WorkGraphBroadcastingSO->QueryInterface(IID_PPV_ARGS(&workGraphProperties))))
            return false;

        D3D12_WORK_GRAPH_MEMORY_REQUIREMENTS workGraphMemoryReqs = {};
        uint32_t workGraphIndex = workGraphProperties->GetWorkGraphIndex(WORKGRAPH_NAME);
        workGraphProperties->GetWorkGraphMemoryRequirements(workGraphIndex, &workGraphMemoryReqs);
        const uint64_t broadcastLaunchMemorySize = workGraphMemoryReqs.MaxSizeInBytes;

        // Create a UAV buffer to hold the work graph backing memory. Use MaxSizeInBytes requested for best performance.
        const uint64_t memorySize = broadcastLaunchMemorySize;

        nvrhi::BufferDesc bufferDesc;
        bufferDesc.byteSize = memorySize;
        bufferDesc.canHaveUAVs = true;
        bufferDesc.debugName = "WorkGraphBackingMem";
        bufferDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
        bufferDesc.keepInitialState = true;
        m_WorkGraphBackingMemory = GetDevice()->createBuffer(bufferDesc);

        return true;
    }

    void UpdateSceneConstants()
    {
        // Camera calculations.
        float3 camPosition, camTarget;
        float4x4 view, proj;
        {
            const float sceneSize = Scene::GetSceneSize();
            const float sceneHeight = Scene::GetSceneHeight();

            camPosition.x = cosf(m_TimeInSeconds * Camera_PositionOrbitSpeed) * sceneSize * Camera_PositionRadiusRatio;
            camPosition.y = sinf(m_TimeInSeconds * Camera_ClimbSpeed - 1.75f) * sceneHeight * Camera_ClimbRatio + sceneHeight * Camera_ClimbRatio + 10.0f;
            camPosition.z = sinf(m_TimeInSeconds * Camera_PositionOrbitSpeed) * sceneSize * Camera_PositionRadiusRatio;

            camTarget.x = cosf(m_TimeInSeconds * Camera_TargetOrbitSpeed) * sceneSize * Camera_TargetRadiusRatio;
            camTarget.y = 0;
            camTarget.z = sinf(m_TimeInSeconds * Camera_TargetOrbitSpeed) * sceneSize * Camera_TargetRadiusRatio;

            float aspectRatio = (float)m_RenderTargets->m_Size.x / (float)m_RenderTargets->m_Size.y;

            const float3 camUp = {0,1,0};
            view = lookToD3DStyle(camPosition, camTarget, camUp);
            proj = perspProjD3DStyle(Camera_VerticalFOV, aspectRatio, Camera_NearClipDistance, sceneSize*1.2f);
        }

        // Write the new values to the constant buffer. Donut internally handles versioning of the buffer.
        SceneConstantBuffer constants = {};
        constants.viewProj = transpose(view*proj);
        constants.viewProjInverse = transpose(inverse(view*proj));
        constants.camPosAndSceneTime.x = camPosition.x;
        constants.camPosAndSceneTime.y = camPosition.y;
        constants.camPosAndSceneTime.z = camPosition.z;
        constants.camPosAndSceneTime.w = m_TimeInSeconds;
        constants.camDir = float4(normalize(camTarget-camPosition),0);
        constants.viewportSizeXY.x = (float)m_RenderTargets->m_Size.x;
        constants.viewportSizeXY.y = (float)m_RenderTargets->m_Size.y;

        m_CommandList->writeBuffer(m_ConstantBuffer, &constants, sizeof(constants));
    }

    void PopulateAnimationPass()
    {
        m_CommandList->beginMarker("Animation");

        bool resetAnim = m_ForceResetAnimation || m_UI.ResetAnim;

        // Object Animation compute shader.
        nvrhi::ComputeState state;
        state.pipeline = m_AnimateObjectsPSO;
        state.bindings = { m_BindingSets[(int)ScenePass::AnimateObjects] };
        m_CommandList->setComputeState(state);

        uint32_t rootConstants[3] = {0, 0, resetAnim ? 1U : 0U};
        ((float*)rootConstants)[0] = m_TimeInSeconds;
        ((float*)rootConstants)[1] = m_TimeDiffThisFrame;
        m_CommandList->setPushConstants(rootConstants, sizeof(rootConstants));

        // Dispatch enough thread groups to cover all scene objects.
        {
            const int threadsX = 32;
            const size_t totalDispatchSize = (m_Scene.GetWorldObjects().size()+(threadsX-1)) / threadsX;
            const size_t dispatchY = max(totalDispatchSize / D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION, size_t(1));
            const size_t dispatchX = max(totalDispatchSize % D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION, size_t(1));
            m_CommandList->dispatch((uint32_t)dispatchX, (uint32_t)dispatchY);
        }

        // Light Animation compute shader.
        state.pipeline = m_AnimateLightsPSO;
        state.bindings = { m_BindingSets[(int)ScenePass::AnimateLights] };
        m_CommandList->setComputeState(state);
        m_CommandList->setPushConstants(rootConstants, sizeof(rootConstants));

        // Dispatch enough thread groups to cover all scene objects.
        {
            const int threadsX = 32;
            const size_t totalDispatchSize = (m_Scene.GetLights().size()+(threadsX-1)) / threadsX;
            const size_t dispatchY = max(totalDispatchSize / D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION, size_t(1));
            const size_t dispatchX = max(totalDispatchSize % D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION, size_t(1));
            m_CommandList->dispatch((uint32_t)dispatchX, (uint32_t)dispatchY);
        }

        m_CommandList->endMarker();

        m_ForceResetAnimation = false; // Animation buffer initialized, no need to redo it again in subsequent frames.
    }

    void PopulateGBufferPass()
    {
        // It is enough to clear the depth-buffer without the g-buffer. Depth buffer values of 1 mean "sky".
        m_CommandList->clearDepthStencilTexture(m_RenderTargets->m_Depth, nvrhi::TextureSubresourceSet(), true, 1.0f, false, 0);

        nvrhi::GraphicsState state;
        state.pipeline = m_GBufferFillPSO;
        state.bindings = { m_BindingSets[(int)ScenePass::GBufferFill] };
        state.framebuffer = m_RenderTargets->m_FrameBufferGB;
        state.viewport.addViewportAndScissorRect(m_RenderTargets->m_FrameBufferGB->getFramebufferInfo().getViewport());
        state.indexBuffer = nvrhi::IndexBufferBinding().setFormat(nvrhi::Format::R16_UINT);
        state.vertexBuffers.push_back(nvrhi::VertexBufferBinding());

        m_CommandList->beginMarker("Draw all meshes");

        size_t objectIndex = 0;
        Scene::MeshType lastMeshType = Scene::MeshType::MT_COUNT;
        uint32_t indexCount = 0;
        for (const Scene::Instance& object : m_Scene.GetWorldObjects())
        {
            if (object.meshType != lastMeshType)
            {
                lastMeshType = object.meshType;

                indexCount = (uint32_t)(m_Scene.GetMeshIndexBuffer(object.meshType)->getDesc().byteSize / sizeof(UINT16));
                state.indexBuffer.buffer = m_Scene.GetMeshIndexBuffer(object.meshType);
                state.vertexBuffers.front().buffer = m_Scene.GetMeshVertexBuffer(object.meshType);
                m_CommandList->setGraphicsState(state);
            }
            
            uint32_t rootConstant[3] = { (uint32_t)objectIndex, 0, 0 };
            m_CommandList->setPushConstants(&rootConstant,sizeof(rootConstant));
            m_CommandList->drawIndexed(nvrhi::DrawArguments().setVertexCount(indexCount));
            objectIndex++;
        }
        m_CommandList->endMarker();
    }

    void PopulateLightCullingPass()
    {
        m_CommandList->beginMarker("Light Culling");

        // Light culling compute shader.
        nvrhi::ComputeState state;
        state.pipeline = m_CullLightsPSO;
        state.bindings = { m_BindingSets[(int)ScenePass::LightCulling] };
        m_CommandList->setComputeState(state);

        const uint32_t tilesX = GetLightTileCountX(m_RenderTargets->m_Size.x);
        const uint32_t tilesY = GetLightTileCountY(m_RenderTargets->m_Size.y);
        const uint32_t rootConstants[3] = {tilesX, tilesY, (uint32_t)m_Scene.GetLights().size()};
        m_CommandList->setPushConstants(rootConstants, sizeof(rootConstants));

        // Dispatch enough thread groups to cover all screen tiles.
        m_CommandList->dispatch(tilesX, tilesY);

        m_CommandList->endMarker();
    }

    void PopulateDeferredShadingPass()
    {
        m_CommandList->beginMarker("Deferred Shading");

        // Deferred shading compute shader.
        nvrhi::ComputeState state;
        state.pipeline = m_ShadePSO;
        state.bindings = { m_BindingSets[(int)ScenePass::DeferredShading] };
        m_CommandList->setComputeState(state);

        const uint32_t tilesX = GetLightTileCountX(m_RenderTargets->m_Size.x);
        const uint32_t tilesY = GetLightTileCountY(m_RenderTargets->m_Size.y);
        const uint32_t rootConstants[3] = {tilesX, tilesY, (uint32_t)m_Scene.GetLights().size()};
        m_CommandList->setPushConstants(rootConstants, sizeof(rootConstants));

        // Dispatch enough thread groups to cover the entire viewport.
        {
            const int threadsX = 8;
            const int threadsY = 4;
            m_CommandList->dispatch((m_RenderTargets->m_Size.x+(threadsX-1))/threadsX, (m_RenderTargets->m_Size.y+(threadsY-1))/threadsY, 1);
        }
        m_CommandList->endMarker();
    }
    
    void PopulateDeferredShadingWorkGraph()
    {
        m_CommandList->beginMarker("Deferred Shading Work Graph");

        // Work graph resource bindings. These are regular bindings applied on the compute state.
        nvrhi::ComputeState state;
        state.pipeline = m_AnimateLightsPSO; // This is ignored. It's just a PSO to allow Donut establish the bindings below.
        state.bindings = { m_BindingSets[(int)ScenePass::WorkGraph] };
        m_CommandList->setComputeState(state);

        const uint32_t rootConstants[3] = {(uint32_t)m_Scene.GetLights().size(), 0, 0};
        m_CommandList->setPushConstants(rootConstants, sizeof(rootConstants));

        // Set the work graph program.
        D3D12_SET_PROGRAM_DESC workGraphSetProgram = {};
        workGraphSetProgram.Type = D3D12_PROGRAM_TYPE_WORK_GRAPH;
        workGraphSetProgram.WorkGraph.ProgramIdentifier = m_workGraphBroadcastingIdentifier;

        ID3D12Resource *workGraphBackingMemoryD3D12 = m_WorkGraphBackingMemory->getNativeObject(nvrhi::ObjectTypes::D3D12_Resource);
        ID3D12GraphicsCommandList *commandListBaseD3D12 = m_CommandList->getNativeObject(nvrhi::ObjectTypes::D3D12_GraphicsCommandList);
        ComPtr<ID3D12GraphicsCommandList10> commandListD3D12;
        commandListBaseD3D12->QueryInterface(IID_PPV_ARGS(&commandListD3D12));

        // Initialize the work graph backing memory only when the backing memory
        // was never used before or if it was used by a different work graph.
        workGraphSetProgram.WorkGraph.Flags = m_InitWorkGraphBackingMemory ? D3D12_SET_WORK_GRAPH_FLAG_INITIALIZE : D3D12_SET_WORK_GRAPH_FLAG_NONE;
        workGraphSetProgram.WorkGraph.BackingMemory.StartAddress = workGraphBackingMemoryD3D12->GetGPUVirtualAddress();
        workGraphSetProgram.WorkGraph.BackingMemory.SizeInBytes = workGraphBackingMemoryD3D12->GetDesc().Width;
        commandListD3D12->SetProgram(&workGraphSetProgram);

        // Spawn work
        D3D12_DISPATCH_GRAPH_DESC dispatchGraphDesc = {};
        dispatchGraphDesc.Mode = D3D12_DISPATCH_MODE_NODE_CPU_INPUT;
        dispatchGraphDesc.NodeCPUInput.EntrypointIndex = 0; // Just one entrypoint in this graph.
        dispatchGraphDesc.NodeCPUInput.NumRecords = 1;
        dispatchGraphDesc.NodeCPUInput.pRecords = nullptr; // Input record has no size, so no need to provide data here.
        dispatchGraphDesc.NodeCPUInput.RecordStrideInBytes = 0;
        commandListD3D12->DispatchGraph(&dispatchGraphDesc);

        m_InitWorkGraphBackingMemory = false; // Memory initialized, no need to redo it again in subsequent frames.

        m_CommandList->endMarker();
    }

    void BackBufferResizing() override
    { 
        m_RenderTargets = nullptr;
    }

    void Animate(float fElapsedTimeSeconds) override
    {
        if (!m_UI.Paused)
        {
            m_TimeDiffThisFrame = fElapsedTimeSeconds;
            m_TimeInSeconds += fElapsedTimeSeconds;
        }
        else m_TimeDiffThisFrame = 0.0f;

        bool resetAnim = m_ForceResetAnimation || m_UI.ResetAnim;
        if (resetAnim)
            m_TimeInSeconds = m_TimeDiffThisFrame = 0.0f;

        if ((int)m_CurrentTechnique != m_UI.CurrentTechnique)
        {
            m_CurrentTechnique = (Techniques)m_UI.CurrentTechnique;
            m_InitWorkGraphBackingMemory = true;
        }

        // Update UI info.
        m_UI.GPUFrameTime = GetLastValidQueryTimer(m_FrameTimers);
        m_UI.GPUShadingTime = GetLastValidQueryTimer(m_ShadingTimers);

        GetDeviceManager()->SetInformativeWindowTitle(g_WindowTitle);
    }
    
    void Render(nvrhi::IFramebuffer* framebuffer) override
    {
        // This is the back buffer. At the end of the frame, the results are copied to it for display.
        const auto& fbinfo = framebuffer->getFramebufferInfo();

        // First frame or window resize. This is where the bulk of the loading occurs.
        if (!m_RenderTargets || m_RenderTargets->IsUpdateRequired(int2(fbinfo.width, fbinfo.height)))
        {
            m_RenderTargets = std::make_unique<RenderTargets>(GetDevice(), int2(fbinfo.width, fbinfo.height));

            LoadScenePipelines(m_RenderTargets->m_FrameBufferGB, framebuffer);
            LoadWorkGraphPipelines(m_RenderTargets->m_FrameBufferGB);
        }

        // Reset GPU timers.
        GetDevice()->resetTimerQuery(m_FrameTimers[m_NextTimerToUse]);
        GetDevice()->resetTimerQuery(m_ShadingTimers[m_NextTimerToUse]);

        // Begin recording the command list for this frame.
        m_CommandList->open();

        m_CommandList->beginTimerQuery(m_FrameTimers[m_NextTimerToUse]);

        // Update scene constants used by all the passes to follow in this frame.
        UpdateSceneConstants();

        // Animation compute passes.
        PopulateAnimationPass();

        // G-buffer fill pass.
        PopulateGBufferPass();

        if (m_CurrentTechnique == Techniques::Dispatch)
        {
            m_CommandList->beginTimerQuery(m_ShadingTimers[m_NextTimerToUse]);

            // Light culling pass.
            PopulateLightCullingPass();

            // Deferred shading pass.
            PopulateDeferredShadingPass();

            m_CommandList->endTimerQuery(m_ShadingTimers[m_NextTimerToUse]);
        }

        if (m_CurrentTechnique == Techniques::WorkGraphBroadcastingLaunch)
        {
            m_CommandList->beginTimerQuery(m_ShadingTimers[m_NextTimerToUse]);

            // Deferred shading work graph pass.
            PopulateDeferredShadingWorkGraph();

            m_CommandList->endTimerQuery(m_ShadingTimers[m_NextTimerToUse]);
        }

        // Copy the final shaded results from the LDR buffer to the back buffer for display.
        m_CommandList->copyTexture(framebuffer->getDesc().colorAttachments[0].texture, nvrhi::TextureSlice(), m_RenderTargets->m_LDRBuffer, nvrhi::TextureSlice());

        m_CommandList->endTimerQuery(m_FrameTimers[m_NextTimerToUse]);

        // Done with this frame.
        m_CommandList->close();
        GetDevice()->executeCommandList(m_CommandList);

        m_NextTimerToUse = (m_NextTimerToUse+1) % QueuedFramesCount;
    }
};

class UIRenderer : public ImGui_Renderer
{
private:
	ImFont* m_FontOpenSans = nullptr;
    std::shared_ptr<donut::vfs::RootFileSystem> m_RootFs;
	std::shared_ptr<ShaderFactory> m_ShaderFactory;

	UIData& m_UI;

public:
    UIRenderer(DeviceManager* deviceManager, UIData& ui) : ImGui_Renderer(deviceManager), m_UI(ui) {}

    bool Init()
    {
        std::filesystem::path mediaPath = app::GetDirectoryWithExecutable().parent_path() / "media";
        std::filesystem::path frameworkShaderPath = app::GetDirectoryWithExecutable() / "shaders/framework" / app::GetShaderTypeName(GetDevice()->getGraphicsAPI());
        m_RootFs = std::make_shared<donut::vfs::RootFileSystem>();
        m_RootFs->mount("/media", mediaPath);
        m_RootFs->mount("/shaders/donut", frameworkShaderPath);

        m_FontOpenSans = LoadFont(*m_RootFs, "/media/fonts/OpenSans/OpenSans-Regular.ttf", 17.f);
        m_ShaderFactory = std::make_shared<ShaderFactory>(GetDevice(), m_RootFs, "/shaders");
        return ImGui_Renderer::Init(m_ShaderFactory);
    }

protected:
    virtual void buildUI(void) override
    {
        if (!m_UI.ShowUI)
            return;

        const char *techniqueNames[] =
        {
            "Work Graph (Broadcast Launch)",
            "Compute Dispatches"
        };

        ImGui::SetNextWindowPos(ImVec2(10.f, 10.f), 0);
        ImGui::Begin("Options/Stats", 0, ImGuiWindowFlags_AlwaysAutoResize);
        ImGui::Combo("Current Technique", &m_UI.CurrentTechnique, techniqueNames, sizeof(techniqueNames)/sizeof(techniqueNames[0]));
        ImGui::Checkbox("Pause Animation", &m_UI.Paused);
        m_UI.ResetAnim = ImGui::Button("Reset Animation");
        ImGui::Text("Frame Time (GPU): %.3f ms", m_UI.GPUFrameTime);
        ImGui::Text("Shading Time (GPU): %.3f ms", m_UI.GPUShadingTime);
        ImGui::End();
    }
};


// AgilitySDK version used with this sample. Incorrect values here will prevent use of experimental features.
extern "C" { __declspec(dllexport) extern const uint32_t D3D12SDKVersion = D3D12_SDK_VERSION; }
extern "C" { __declspec(dllexport) extern const char* D3D12SDKPath = u8".\\D3D12\\"; }

#ifdef WIN32
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
#else
int main(int __argc, const char** __argv)
#endif
{
    nvrhi::GraphicsAPI api = app::GetGraphicsAPIFromCommandLine(__argc, __argv);
    if (api != nvrhi::GraphicsAPI::D3D12)
    {
        log::fatal("The Work Graphs example can only run on D3D12 API.");
        return -1;
    }

    app::DeviceManager* deviceManager = app::DeviceManager::Create(api);

    app::DeviceCreationParameters deviceParams;
#ifdef _DEBUG
    deviceParams.enableDebugRuntime = true; 
    deviceParams.enableNvrhiValidationLayer = true;
#endif
    deviceParams.backBufferWidth = 1920;
    deviceParams.backBufferHeight = 1080;

    if (!deviceManager->CreateWindowDeviceAndSwapChain(deviceParams, g_WindowTitle))
    {
        log::fatal("Cannot initialize a graphics device with the requested parameters");
        return 1;
    }
    
    {
        UIData uiData;
        WorkGraphs example(deviceManager, uiData);
        UIRenderer ui(deviceManager, uiData);
        if (example.Init() && ui.Init())
        {
            deviceManager->AddRenderPassToBack(&example);
            deviceManager->AddRenderPassToBack(&ui);
            deviceManager->RunMessageLoop();
            deviceManager->RemoveRenderPass(&ui);
            deviceManager->RemoveRenderPass(&example);
        }
    }
    
    deviceManager->Shutdown();

    delete deviceManager;

    return 0;
}