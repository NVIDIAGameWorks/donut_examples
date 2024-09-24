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
#include <donut/app/imgui_renderer.h>
#include <nvrhi/utils.h>

#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>

using namespace donut;

static const char* g_WindowTitle = "Donut Example: Aftermath";

// The sample currently support two types of crashes: timeout and pagefault
// A timeout happens when a single workload runs over the TDR timeout limit (default 2 seconds on Windows)
// The sample purposely triggers a long workload by causing an infinite loop in the shader
// A page fault can happen in many different ways, but the sample causes one by destroying an in-use resource
// See shaders.hlsl to see where we expect the crashes to trigger
enum class CrashType
{
    NONE = 0,
    TIMEOUT = 1,
    PAGEFAULT = 2
};

class AftermathSample : public app::IRenderPass
{
private:
    nvrhi::ShaderHandle m_VertexShader;
    nvrhi::ShaderHandle m_PixelShader;
    nvrhi::GraphicsPipelineHandle m_Pipeline;
    nvrhi::BindingLayoutHandle m_BindingLayout;
    nvrhi::BindingSetHandle m_BindingSet;
    nvrhi::CommandListHandle m_CommandList;
    std::shared_ptr<engine::ShaderFactory> m_ShaderFactory;
    nvrhi::BufferHandle m_Buffer;
    bool m_WaitingForCrash;
    CrashType m_CrashType;

public:
    using IRenderPass::IRenderPass;

    void SetCrashType(CrashType c)
    {
        m_CrashType = c;
    }

    std::shared_ptr<engine::ShaderFactory> GetShaderFactory()
    {
        return m_ShaderFactory;
    }

    bool Init()
    {
        std::filesystem::path appShaderPath = app::GetDirectoryWithExecutable() / "shaders/aftermath_sample" /  app::GetShaderTypeName(GetDevice()->getGraphicsAPI());
        std::filesystem::path frameworkShaderPath = app::GetDirectoryWithExecutable() / "shaders/framework" / app::GetShaderTypeName(GetDevice()->getGraphicsAPI());


        auto rootFS = std::make_shared<vfs::RootFileSystem>();
        rootFS->mount("/shaders/donut", frameworkShaderPath);
        rootFS->mount("/shaders/app", appShaderPath);
        m_ShaderFactory = std::make_shared<engine::ShaderFactory>(GetDevice(), rootFS, "/shaders");

        m_VertexShader = m_ShaderFactory->CreateShader("app/shaders.hlsl", "main_vs", nullptr, nvrhi::ShaderType::Vertex);
        m_PixelShader = m_ShaderFactory->CreateShader("app/shaders.hlsl", "main_ps", nullptr, nvrhi::ShaderType::Pixel);

        if (!m_VertexShader || !m_PixelShader)
        {
            return false;
        }
        
        m_CommandList = GetDevice()->createCommandList();

        nvrhi::BufferDesc bufDesc = {};
        bufDesc.setByteSize(1024)
            .setCanHaveUAVs(true)
            .setDebugName("Aftermath test buffer")
            .setFormat(nvrhi::Format::R32_FLOAT)
            .setInitialState(nvrhi::ResourceStates::UnorderedAccess)
            .setKeepInitialState(true)
            .setStructStride(sizeof(float));
        m_Buffer = GetDevice()->createBuffer(bufDesc);
        m_WaitingForCrash = false;
        
        return true;
    }

    void BackBufferResizing() override
    { 
        m_Pipeline = nullptr;
    }

    void Animate(float fElapsedTimeSeconds) override
    {
        GetDeviceManager()->SetInformativeWindowTitle(g_WindowTitle);
    }
    
    void Render(nvrhi::IFramebuffer* framebuffer) override
    {
        if (!m_Pipeline)
        {
            nvrhi::BindingLayoutDesc bindingLayoutDesc;
            bindingLayoutDesc.visibility = nvrhi::ShaderType::All;
            bindingLayoutDesc.bindings = {
                nvrhi::BindingLayoutItem::PushConstants(0, sizeof(uint32_t)),
                nvrhi::BindingLayoutItem::StructuredBuffer_UAV(0)
            };
            m_BindingLayout = GetDevice()->createBindingLayout(bindingLayoutDesc);

            nvrhi::BindingSetDesc bindingSetDesc;
            bindingSetDesc.bindings = {
                nvrhi::BindingSetItem::PushConstants(0, sizeof(uint32_t)),
                nvrhi::BindingSetItem::StructuredBuffer_UAV(0, m_Buffer)
            };
            m_BindingSet = GetDevice()->createBindingSet(bindingSetDesc, m_BindingLayout);

            nvrhi::GraphicsPipelineDesc psoDesc;
            psoDesc.VS = m_VertexShader;
            psoDesc.PS = m_PixelShader;
            psoDesc.primType = nvrhi::PrimitiveType::TriangleList;
            psoDesc.renderState.depthStencilState.depthTestEnable = false;
            psoDesc.bindingLayouts = { m_BindingLayout };

            m_Pipeline = GetDevice()->createGraphicsPipeline(psoDesc, framebuffer);
        }


        m_CommandList->open();
        m_CommandList->beginMarker("Frame");

        // One way to cause a page fault is to destroy a resource that is in use
        // If we delete the entire nvrhi resource, the application will crash on the CPU side before the GPU
        // So instead we get the native graphics API objects from NVRHI and destroy them directly
        if (m_CrashType == CrashType::PAGEFAULT && !m_WaitingForCrash)
        {
            auto api = GetDevice()->getGraphicsAPI();
#if DONUT_WITH_DX12
            if (api == nvrhi::GraphicsAPI::D3D12)
            {
                ID3D12Resource* resource = m_Buffer->getNativeObject(nvrhi::ObjectTypes::D3D12_Resource);
                resource->Release();
            }
#endif
#if DONUT_WITH_VULKAN
            if (api == nvrhi::GraphicsAPI::VULKAN)
            {
                vk::DeviceMemory memory = vk::DeviceMemory(m_Buffer->getNativeObject(nvrhi::ObjectTypes::VK_DeviceMemory));
                vk::Device device = vk::Device(GetDevice()->getNativeObject(nvrhi::ObjectTypes::VK_Device));
                device.freeMemory(memory);
            }
#endif
            m_CommandList->setEnableAutomaticBarriers(false);
            m_WaitingForCrash = true;
        }

        m_CommandList->beginMarker("Clear");
        nvrhi::utils::ClearColorAttachment(m_CommandList, framebuffer, 0, nvrhi::Color(0.f));
        m_CommandList->endMarker();

        m_CommandList->beginMarker("Draw Triangle");
        nvrhi::GraphicsState state;
        state.pipeline = m_Pipeline;
        state.framebuffer = framebuffer;
        state.viewport.addViewportAndScissorRect(framebuffer->getFramebufferInfo().getViewport());
        state.addBindingSet(m_BindingSet);

        m_CommandList->setGraphicsState(state);

        m_CommandList->setPushConstants(&m_CrashType, sizeof(uint32_t));
        nvrhi::DrawArguments args;
        args.vertexCount = 3;
        m_CommandList->draw(args);
        m_CommandList->endMarker();

        m_CommandList->endMarker();
        m_CommandList->close();

        GetDevice()->executeCommandList(m_CommandList);
    }

};

class UIRenderer : public app::ImGui_Renderer
{
private:
    AftermathSample& m_app;

public:
    UIRenderer(app::DeviceManager* deviceManager, AftermathSample& app)
        : ImGui_Renderer(deviceManager)
        , m_app(app)
    {
    }

protected:
    virtual void buildUI(void) override
    {
        int width, height;
        GetDeviceManager()->GetWindowDimensions(width, height);

        ImGui::SetNextWindowPos(ImVec2(10.f, 10.f), 0);
        ImGui::Begin("Controls", 0, ImGuiWindowFlags_AlwaysAutoResize);
        if (ImGui::Button("Trigger timeout"))
        {
            m_app.SetCrashType(CrashType::TIMEOUT);
        }
        // d3d11 does not page fault in these conditions, so short circuit showing the button in d3d11
        if (m_app.GetDevice()->getGraphicsAPI() != nvrhi::GraphicsAPI::D3D11 && ImGui::Button("Trigger page fault"))
        {
            m_app.SetCrashType(CrashType::PAGEFAULT);
        }
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
#ifdef _DEBUG
    deviceParams.enableNvrhiValidationLayer = true;
#endif
    // Aftermath is incompatible with D3D debug layer
    deviceParams.enableDebugRuntime = false;
    deviceParams.enableAftermath = true;

    if (!deviceManager->CreateWindowDeviceAndSwapChain(deviceParams, g_WindowTitle))
    {
        log::fatal("Cannot initialize a graphics device with the requested parameters");
        return 1;
    }
    
    {
        AftermathSample example(deviceManager);
        UIRenderer gui(deviceManager, example);
        if (example.Init() && gui.Init(example.GetShaderFactory()))
        {
            deviceManager->AddRenderPassToBack(&example);
            deviceManager->AddRenderPassToBack(&gui);
            deviceManager->RunMessageLoop();
            deviceManager->RemoveRenderPass(&example);
        }
    }
    
    deviceManager->Shutdown();

    delete deviceManager;

    return 0;
}
