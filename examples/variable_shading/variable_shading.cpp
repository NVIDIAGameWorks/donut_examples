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
#include <donut/render/TemporalAntiAliasingPass.h>
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

static const char* g_WindowTitle = "Donut Example: Variable Rate Shading";

// NVIDIA Variable Rate Shading (VRS) sample application
// Relevant sample code is in the Render() function, marked with comments

class RenderTargets
{
public:
    nvrhi::TextureHandle m_Depth;
    nvrhi::TextureHandle m_GBufferDiffuse;
    nvrhi::TextureHandle m_GBufferSpecular;
    nvrhi::TextureHandle m_GBufferNormals;
    nvrhi::TextureHandle m_HdrColor;
    nvrhi::TextureHandle m_ResolvedColor;
    nvrhi::TextureHandle m_TemporalFeedback1;
    nvrhi::TextureHandle m_TemporalFeedback2;
    nvrhi::TextureHandle m_MotionVectors;

    std::shared_ptr<engine::FramebufferFactory> m_HdrFramebuffer;
    std::shared_ptr<engine::FramebufferFactory> m_HdrFramebufferDepth;

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
        desc.debugName = "ResolvedColor";
        m_ResolvedColor = device->createTexture(desc);

        desc.format = nvrhi::Format::RGBA16_SNORM;
        desc.debugName = "TemporalFeedback1";
        m_TemporalFeedback1 = device->createTexture(desc);
        desc.debugName = "TemporalFeedback2";
        m_TemporalFeedback2 = device->createTexture(desc);

        desc.format = nvrhi::Format::RG16_FLOAT;
        desc.debugName = "MotionVectors";
        m_MotionVectors = device->createTexture(desc);

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
    nvrhi::CommandListHandle m_CommandList;
    
    nvrhi::BufferHandle m_ConstantBuffer;

    std::shared_ptr<engine::ShaderFactory> m_ShaderFactory;
    std::unique_ptr<engine::Scene> m_Scene;
    std::unique_ptr<render::ForwardShadingPass> m_ForwardPass;
    std::unique_ptr<render::TemporalAntiAliasingPass> m_temporalPass;
    std::unique_ptr<RenderTargets> m_RenderTargets;
    app::FirstPersonCamera m_Camera;
    engine::PlanarView m_View;
    std::shared_ptr<engine::DirectionalLight>  m_SunLight;
    std::unique_ptr<render::InstancedOpaqueDrawStrategy> m_OpaqueDrawStrategy;
    std::unique_ptr<render::TransparentDrawStrategy> m_TransparentDrawStrategy;
    std::unique_ptr<engine::BindingCache> m_BindingCache;

    nvrhi::ShaderHandle m_shadingRateSurfaceShader;
    nvrhi::ComputePipelineHandle m_Pipeline;
    nvrhi::BindingLayoutHandle m_bindingLayout;
    nvrhi::BindingSetHandle m_bindingSet;
    nvrhi::TextureHandle m_shadingRateSurface;
    uint m_vrsTileSize;

    engine::PlanarView m_ViewPrevious;
    bool m_PreviousViewsValid = false;

    bool m_UseRawD3D12 = false;

public:
    using ApplicationBase::ApplicationBase;

    bool Init(bool useRawD3D12)
    {
        m_UseRawD3D12 = useRawD3D12;

        std::filesystem::path sceneFileName = app::GetDirectoryWithExecutable().parent_path() / "media/glTF-Sample-Models/2.0/Sponza/glTF/Sponza.gltf";
        std::filesystem::path frameworkShaderPath = app::GetDirectoryWithExecutable() / "shaders/framework" / app::GetShaderTypeName(GetDevice()->getGraphicsAPI());
        std::filesystem::path appShaderPath = app::GetDirectoryWithExecutable() / "shaders/variable_shading" / app::GetShaderTypeName(GetDevice()->getGraphicsAPI());
        
        m_RootFS = std::make_shared<vfs::RootFileSystem>();
        m_RootFS->mount("/shaders/donut", frameworkShaderPath);
        m_RootFS->mount("/shaders/app", appShaderPath);

        m_ShaderFactory = std::make_shared<engine::ShaderFactory>(GetDevice(), m_RootFS, "/shaders");
        m_CommonPasses = std::make_shared<engine::CommonRenderPasses>(GetDevice(), m_ShaderFactory);
        m_BindingCache = std::make_unique<engine::BindingCache>(GetDevice());

        m_shadingRateSurfaceShader = m_ShaderFactory->CreateShader("/shaders/app/shaders.hlsl", "main_cs", nullptr, nvrhi::ShaderType::Compute);
        if (!m_shadingRateSurfaceShader)
        {
            return false;
        }

        auto nativeFS = std::make_shared<vfs::NativeFileSystem>();
        m_TextureCache = std::make_shared<engine::TextureCache>(GetDevice(), nativeFS, nullptr);

        SetAsynchronousLoadingEnabled(false);
        BeginLoadingScene(nativeFS, sceneFileName);

        m_OpaqueDrawStrategy = std::make_unique<render::InstancedOpaqueDrawStrategy>();
        m_TransparentDrawStrategy = std::make_unique<render::TransparentDrawStrategy>();

        m_SunLight = std::make_shared<engine::DirectionalLight>();
        m_Scene->GetSceneGraph()->AttachLeafNode(m_Scene->GetSceneGraph()->GetRootNode(), m_SunLight);
        m_SunLight->SetDirection(double3(0.1, -1.0, 0.15));
        m_SunLight->SetName("Sun");
        m_SunLight->angularSize = 0.53f;
        m_SunLight->irradiance = 2.f;

        m_Scene->FinishedLoading(GetFrameIndex());

        m_Camera.LookAt(float3(0.f, 1.8f, 0.f), float3(1.f, 1.8f, 0.f));
        m_Camera.SetMoveSpeed(3.f);

        m_ConstantBuffer = GetDevice()->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(sizeof(LightingConstants), "LightingConstants", engine::c_MaxRenderPassConstantBufferVersions));

        m_CommandList = GetDevice()->createCommandList();
        
#ifdef DONUT_WITH_DX12
        // Query VRS tile size (it can vary depending on hardware)
        if (m_UseRawD3D12)
        {
            D3D12_FEATURE_DATA_D3D12_OPTIONS6 options = {};
            ID3D12Device* device = GetDevice()->getNativeObject(nvrhi::ObjectTypes::D3D12_Device);
            auto hr = device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS6, &options, sizeof(options));
            m_vrsTileSize = options.ShadingRateImageTileSize;
        }
        else
#endif
        {
            nvrhi::VariableRateShadingFeatureInfo info = {};
            GetDevice()->queryFeatureSupport(nvrhi::Feature::VariableRateShading, &info, sizeof(info));
            m_vrsTileSize = info.shadingRateImageTileSize;
        }

        GetDevice()->waitForIdle();

        return true;
    }

    bool LoadScene(std::shared_ptr<vfs::IFileSystem> fs, const std::filesystem::path& sceneFileName) override
    {
        engine::Scene* scene = new engine::Scene(GetDevice(), *m_ShaderFactory, fs, m_TextureCache, nullptr, nullptr);

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

    void BackBufferResizing() override
    {
        m_RenderTargets = nullptr;
        m_BindingCache->Clear();
        m_ForwardPass = nullptr;
        m_shadingRateSurface = nullptr;
        m_temporalPass = nullptr;
        m_Pipeline = nullptr;
    }

    void Render(nvrhi::IFramebuffer* framebuffer) override
    {
        const auto& fbinfo = framebuffer->getFramebufferInfo();

        if (!m_RenderTargets)
        {
            m_RenderTargets = std::make_unique<RenderTargets>(GetDevice(), int2(fbinfo.width, fbinfo.height));
        }

        nvrhi::Viewport windowViewport(float(fbinfo.width), float(fbinfo.height));
        m_View.SetViewport(windowViewport);
        m_View.SetMatrices(m_Camera.GetWorldToViewMatrix(), perspProjD3DStyleReverse(dm::PI_f * 0.25f, windowViewport.width() / windowViewport.height(), 0.1f));
        m_View.UpdateCache();

        // VRS-specific code starts here
        // Use the queried tile size to determine the size of the VRS surface; it will be approximately 1/tileSize in both dimensions (with some rounding)
        uint2 surfaceDimensions((fbinfo.width + m_vrsTileSize - 1) / m_vrsTileSize, (fbinfo.height + m_vrsTileSize - 1) / m_vrsTileSize);
        if (!m_shadingRateSurface)
        {
            nvrhi::TextureDesc desc;
            desc.width = surfaceDimensions.x;
            desc.height = surfaceDimensions.y;
            desc.isRenderTarget = false;
            desc.useClearValue = false;
            desc.sampleCount = 1;
            desc.dimension = nvrhi::TextureDimension::Texture2D;
            desc.keepInitialState = true;
            desc.arraySize = 1;
            desc.isUAV = true;
            desc.isShadingRateSurface = true;
            desc.initialState = nvrhi::ResourceStates::UnorderedAccess;
            // Important!  VRS surface should be R8_UINT format
            desc.format = nvrhi::Format::R8_UINT;

            m_shadingRateSurface = GetDevice()->createTexture(desc);
        }

        if (!m_ForwardPass)
        {
            m_ForwardPass = std::make_unique<render::ForwardShadingPass>(GetDevice(), m_CommonPasses);

            render::ForwardShadingPass::CreateParameters forwardParams;
            if (!m_UseRawD3D12)
            {
                m_RenderTargets->m_HdrFramebufferDepth->ShadingRateSurface = m_shadingRateSurface;
            }
            m_ForwardPass->Init(*m_ShaderFactory, forwardParams);
        }

        if (!m_temporalPass)
        {
            render::TemporalAntiAliasingPass::CreateParameters taaParams;
            taaParams.sourceDepth = m_RenderTargets->m_Depth;
            taaParams.motionVectors = m_RenderTargets->m_MotionVectors;
            taaParams.unresolvedColor = m_RenderTargets->m_HdrColor;
            taaParams.resolvedColor = m_RenderTargets->m_ResolvedColor;
            taaParams.feedback1 = m_RenderTargets->m_TemporalFeedback1;
            taaParams.feedback2 = m_RenderTargets->m_TemporalFeedback2;
            taaParams.motionVectorStencilMask = 0x01;
            taaParams.useCatmullRomFilter = true;

            m_temporalPass = std::make_unique<render::TemporalAntiAliasingPass>(GetDevice(), m_ShaderFactory, m_CommonPasses, m_View, taaParams);
        }

        // A pipeline state for the compute shader which will generate the VRS surface
        if (!m_Pipeline)
        {
            nvrhi::BindingLayoutDesc layoutDesc;
            layoutDesc.visibility = nvrhi::ShaderType::Compute;
            layoutDesc.bindings = {
                nvrhi::BindingLayoutItem::Texture_UAV(0),
                nvrhi::BindingLayoutItem::Texture_SRV(0),
                nvrhi::BindingLayoutItem::Texture_SRV(1)
            };
            m_bindingLayout = GetDevice()->createBindingLayout(layoutDesc);

            nvrhi::BindingSetDesc bindingSetDesc;
            bindingSetDesc.bindings = {
                nvrhi::BindingSetItem::Texture_UAV(0, m_shadingRateSurface, nvrhi::Format::R8_UINT),
                nvrhi::BindingSetItem::Texture_SRV(0, m_RenderTargets->m_MotionVectors, nvrhi::Format::RG16_FLOAT),
                nvrhi::BindingSetItem::Texture_SRV(1, m_RenderTargets->m_HdrColor, nvrhi::Format::RGBA16_FLOAT)
            };
            m_bindingSet = GetDevice()->createBindingSet(bindingSetDesc, m_bindingLayout);

            nvrhi::ComputePipelineDesc psoDesc = {};
            psoDesc.CS = m_shadingRateSurfaceShader;
            psoDesc.bindingLayouts = { m_bindingLayout };

            m_Pipeline = GetDevice()->createComputePipeline(psoDesc);
        }

        m_CommandList->open();

        if (m_PreviousViewsValid)
        {
            m_temporalPass->RenderMotionVectors(m_CommandList, m_View, m_ViewPrevious);
        }

        nvrhi::ComputeState state;
        state.pipeline = m_Pipeline;
        state.bindings = { m_bindingSet };
        m_CommandList->setComputeState(state);

        // Dispatch call to generate the VRS surface
        m_CommandList->dispatch(surfaceDimensions.x, surfaceDimensions.y, 1);

        m_RenderTargets->Clear(m_CommandList);

        LightingConstants constants = {};
        constants.ambientColor = float4(0.2f);
        m_View.FillPlanarViewConstants(constants.view);
        // the PrepareLights() call below will send the constants to the command list, so no need to call it explictly here

#ifdef DONUT_WITH_DX12
        if (m_UseRawD3D12)
        {
            // VRS command list methods require ID3D12GraphicsCommandList5
            ID3D12GraphicsCommandList* d3dcmdlist = m_CommandList->getNativeObject(nvrhi::ObjectTypes::D3D12_GraphicsCommandList);
            ID3D12GraphicsCommandList5* vrscmdlist = nullptr;
            HRESULT hr = d3dcmdlist->QueryInterface(IID_PPV_ARGS(&vrscmdlist));
            ID3D12Resource* vrsResource = m_shadingRateSurface->getNativeObject(nvrhi::ObjectTypes::D3D12_Resource);
            
            D3D12_RESOURCE_BARRIER barrier = {};
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barrier.Transition.pResource = vrsResource;
            barrier.Transition.Subresource = 0;
            barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            // Use the special SHADING_RATE_SOURCE resource state for barriers on the VRS surface
            barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_SHADING_RATE_SOURCE;
            vrscmdlist->ResourceBarrier(1, &barrier);

            // Tell D3D to use the VRS surface for rendering by calling RSSetShadingRateImage()
            vrscmdlist->RSSetShadingRateImage(vrsResource);
            // VRS on D3D12 defines combiners for resolving shading rates from different points in the pipeline (per-drawcall, per-primitive, VRS surface)
            // we want to set the shading rate via the VRS surface only, so just setting all combiners to MAX so that the "coarsest" shading rate always wins, and set all other sources to 1X1 rate
            D3D12_SHADING_RATE_COMBINER combiners[D3D12_RS_SET_SHADING_RATE_COMBINER_COUNT];
            for (int i = 0; i < D3D12_RS_SET_SHADING_RATE_COMBINER_COUNT; i++)
            {
                combiners[i] = D3D12_SHADING_RATE_COMBINER_MAX;
            }
            // In addition to setting the combiners, the RSSetShadingRate() function also defines the per-drawcall shading rate (we set to 1X1 because we don't want to use it)
            vrscmdlist->RSSetShadingRate(D3D12_SHADING_RATE_1X1, combiners);
        }
        else
#endif // DONUT_WITH_DX12
        {
            // enable VRS, with a per-drawcall shading rate of 1X1, and make the shading rate image result always override all others
            m_View.SetVariableRateShadingState(nvrhi::VariableRateShadingState().setEnabled(true).setShadingRate(nvrhi::VariableShadingRate::e1x1).setImageCombiner(nvrhi::ShadingRateCombiner::Override));
        }

        // Forward pass to draw the scene with the VRS surface set above
        render::ForwardShadingPass::Context forwardContext;
        m_ForwardPass->PrepareLights(forwardContext, m_CommandList, m_Scene->GetSceneGraph()->GetLights(), constants.ambientColor, constants.ambientColor, {});
        render::RenderCompositeView(m_CommandList, &m_View, &m_View, *m_RenderTargets->m_HdrFramebufferDepth, m_Scene->GetSceneGraph()->GetRootNode(), *m_OpaqueDrawStrategy, *m_ForwardPass, forwardContext);
        render::RenderCompositeView(m_CommandList, &m_View, &m_View, *m_RenderTargets->m_HdrFramebufferDepth, m_Scene->GetSceneGraph()->GetRootNode(), *m_TransparentDrawStrategy, *m_ForwardPass, forwardContext);

#ifdef DONUT_WITH_DX12
        if (m_UseRawD3D12)
        {
            ID3D12GraphicsCommandList* d3dcmdlist = m_CommandList->getNativeObject(nvrhi::ObjectTypes::D3D12_GraphicsCommandList);
            ID3D12GraphicsCommandList5* vrscmdlist = nullptr;
            HRESULT hr = d3dcmdlist->QueryInterface(IID_PPV_ARGS(&vrscmdlist));
            ID3D12Resource* vrsResource = m_shadingRateSurface->getNativeObject(nvrhi::ObjectTypes::D3D12_Resource);
            D3D12_RESOURCE_BARRIER barrier = {};
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barrier.Transition.pResource = vrsResource;
            barrier.Transition.Subresource = 0;
            barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_SHADING_RATE_SOURCE;
            barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            vrscmdlist->ResourceBarrier(1, &barrier);

            // To disable VRS, set shading rate to 1X1 with no combiners, and null out RSSetShadingRateImage()
            vrscmdlist->RSSetShadingRate(D3D12_SHADING_RATE_1X1, nullptr);
            vrscmdlist->RSSetShadingRateImage(nullptr);
        }
        else
#endif // DONUT_WITH_DX12
        {
            m_View.SetVariableRateShadingState(nvrhi::VariableRateShadingState().setEnabled(false));
        }

        // VRS-specific code ends here

        // TAA pass (runs at full rate)
        {
            render::TemporalAntiAliasingParameters params = {};
            m_temporalPass->TemporalResolve(m_CommandList, params, m_PreviousViewsValid, m_View, m_View);
            m_ViewPrevious = m_View;
            m_PreviousViewsValid = true;
        }

        m_CommonPasses->BlitTexture(m_CommandList, framebuffer, m_RenderTargets->m_ResolvedColor, m_BindingCache.get());

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
    if (api == nvrhi::GraphicsAPI::D3D11)
    {
        log::error("The Variable Rate Shading example does not support D3D11.");
        return 1;
    }

    // if d3d12 is selected and -raw flag is on, use raw d3d12 API path
    bool rawD3D12 = false;
#ifdef DONUT_WITH_DX12
    for (int i = 1; i < __argc; i++)
    {
        if (!strcmp(__argv[i], "-raw"))
        {
            rawD3D12 = (api == nvrhi::GraphicsAPI::D3D12);
        }
    }
#endif

    app::DeviceManager* deviceManager = app::DeviceManager::Create(api);

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

    if (!deviceManager->GetDevice()->queryFeatureSupport(nvrhi::Feature::VariableRateShading))
    {
        log::error("The device does not support Variable Rate Shading");
        return 1;
    }

    {
        VariableRateShading example(deviceManager);
        if (example.Init(rawD3D12))
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
