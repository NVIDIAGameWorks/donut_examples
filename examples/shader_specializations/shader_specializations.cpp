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
#include <donut/app/DeviceManager.h>
#include <donut/core/log.h>
#include <donut/core/vfs/VFS.h>
#include <nvrhi/utils.h>

using namespace donut;

static const char* g_WindowTitle = "Donut Example: Vulkan Shader Specializations";

class ShaderSpecializations : public app::IRenderPass
{
private:
    nvrhi::ShaderHandle m_VertexShader;
    nvrhi::ShaderHandle m_PixelShader;
    std::vector<nvrhi::GraphicsPipelineHandle> m_Pipelines;
    nvrhi::CommandListHandle m_CommandList;

public:
    using IRenderPass::IRenderPass;

    bool Init()
    {
        auto nativeFS = std::make_shared<vfs::NativeFileSystem>();

        std::filesystem::path appShaderPath = app::GetDirectoryWithExecutable() / "shaders/shader_specializations" / app::GetShaderTypeName(GetDevice()->getGraphicsAPI());

        engine::ShaderFactory shaderFactory(GetDevice(), nativeFS, appShaderPath);
        m_VertexShader = shaderFactory.CreateShader("shaders.hlsl", "main_vs", nullptr, nvrhi::ShaderType::Vertex);
        m_PixelShader = shaderFactory.CreateShader("shaders.hlsl", "main_ps", nullptr, nvrhi::ShaderType::Pixel);

        if (!m_VertexShader || !m_PixelShader)
        {
            return false;
        }
        
        m_CommandList = GetDevice()->createCommandList();

        return true;
    }

    void Animate(float fElapsedTimeSeconds) override
    {
        GetDeviceManager()->SetInformativeWindowTitle(g_WindowTitle);
    }

    void BackBufferResizing() override
    { 
        m_Pipelines.clear();
    }

    void Render(nvrhi::IFramebuffer* framebuffer) override
    {
        if (m_Pipelines.empty())
        {
            nvrhi::IDevice* device = GetDevice();

            // Create pipelines with shader specializations.
            // The specializations could be created ahead of time, but they're cheap and it doesn't really matter.

            for (uint32_t i = 0; i < 4; i++)
            {
                // Vertex shader specialization
                nvrhi::ShaderSpecialization vertexShaderSpecializations[] = {
                    nvrhi::ShaderSpecialization::Float( 0, float(i) * 0.5f - 0.75f)
                };
                nvrhi::ShaderHandle vertexShader = device->createShaderSpecialization(m_VertexShader, 
                    vertexShaderSpecializations, uint32_t(std::size(vertexShaderSpecializations)));
                
                // Pixel shader specialization
                uint32_t colors[4] = { 0x0000ff, 0x00ff00, 0xff0000, 0xff00ff };
                nvrhi::ShaderSpecialization pixelShaderSpecializations[] = {
                    nvrhi::ShaderSpecialization::UInt32(1, colors[i])
                };
                nvrhi::ShaderHandle pixelShader = device->createShaderSpecialization(m_PixelShader, 
                    pixelShaderSpecializations, uint32_t(std::size(pixelShaderSpecializations)));

                // Pipeline
                nvrhi::GraphicsPipelineDesc psoDesc;
                psoDesc.VS = vertexShader;
                psoDesc.PS = pixelShader;
                psoDesc.primType = nvrhi::PrimitiveType::TriangleList;
                psoDesc.renderState.depthStencilState.depthTestEnable = false;

                nvrhi::GraphicsPipelineHandle pipeline = device->createGraphicsPipeline(psoDesc, framebuffer);
                assert(pipeline);

                m_Pipelines.push_back(pipeline);
            }
        }

        m_CommandList->open();

        nvrhi::utils::ClearColorAttachment(m_CommandList, framebuffer, 0, nvrhi::Color(0.f));

        // Render triangles, one with each pipeline.
        // Expected output: 4 triangles side-by-side; red, green, blue, cyan.

        for (const auto& pipeline : m_Pipelines)
        {
            nvrhi::GraphicsState state;
            state.pipeline = pipeline;
            state.framebuffer = framebuffer;
            state.viewport.addViewportAndScissorRect(framebuffer->getFramebufferInfo().getViewport());

            m_CommandList->setGraphicsState(state);

            nvrhi::DrawArguments args;
            args.vertexCount = 3;
            m_CommandList->draw(args);
        }

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
    app::DeviceManager* deviceManager = app::DeviceManager::Create(nvrhi::GraphicsAPI::VULKAN);

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
        ShaderSpecializations example(deviceManager);
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
