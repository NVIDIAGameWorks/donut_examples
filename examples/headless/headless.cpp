/*
* Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

#include <donut/app/DeviceManager.h>
#include <donut/app/ApplicationBase.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/core/log.h>
#include <donut/core/vfs/VFS.h>
#include <nvrhi/utils.h>

using namespace donut;

bool RunTest(nvrhi::IDevice* device)
{
    std::filesystem::path appShaderPath = app::GetDirectoryWithExecutable() / "shaders/headless" /  app::GetShaderTypeName(device->getGraphicsAPI());
    
    auto nativeFS = std::make_shared<vfs::NativeFileSystem>();
    engine::ShaderFactory shaderFactory(device, nativeFS, appShaderPath);

    nvrhi::ShaderHandle computeShader = shaderFactory.CreateShader("shaders.hlsl", "main", nullptr, nvrhi::ShaderType::Compute);

    if (!computeShader)
        return false;

    // The shader is performing a reduction operation within one thread group, adding all uint's in the input buffer.
    // The number of uint's is the same as the thread group size.
    constexpr uint32_t numInputValues = 256;

    // Create the input, output, and readback buffers...

    auto inputBufferDesc = nvrhi::BufferDesc()
        .setByteSize(sizeof(uint32_t) * numInputValues)
        .setCanHaveTypedViews(true)
        .setFormat(nvrhi::Format::R32_UINT)
        .setDebugName("InputBuffer")
        .setInitialState(nvrhi::ResourceStates::CopyDest)
        .setKeepInitialState(true);

    auto outputBufferDesc = nvrhi::BufferDesc()
        .setByteSize(sizeof(uint32_t))
        .setCanHaveTypedViews(true)
        .setCanHaveUAVs(true)
        .setFormat(nvrhi::Format::R32_UINT)
        .setDebugName("OutputBuffer")
        .setInitialState(nvrhi::ResourceStates::UnorderedAccess)
        .setKeepInitialState(true);

    auto readbackBufferDesc = nvrhi::BufferDesc()
        .setByteSize(outputBufferDesc.byteSize)
        .setCpuAccess(nvrhi::CpuAccessMode::Read)
        .setDebugName("ReadbackBuffer")
        .setInitialState(nvrhi::ResourceStates::CopyDest)
        .setKeepInitialState(true);

    auto inputBuffer = device->createBuffer(inputBufferDesc);
    auto outputBuffer = device->createBuffer(outputBufferDesc);
    auto readbackBuffer = device->createBuffer(readbackBufferDesc);

    // Create the binding layout and binding set...

    auto bindingSetDesc = nvrhi::BindingSetDesc()
        .addItem(nvrhi::BindingSetItem::TypedBuffer_SRV(0, inputBuffer))
        .addItem(nvrhi::BindingSetItem::TypedBuffer_UAV(0, outputBuffer));

    nvrhi::BindingSetHandle bindingSet;
    nvrhi::BindingLayoutHandle bindingLayout;
    if (!nvrhi::utils::CreateBindingSetAndLayout(device, nvrhi::ShaderType::Compute, 0, bindingSetDesc, bindingLayout, bindingSet))
        return false;

    // Create the compute pipeline...

    auto computePipelineDesc = nvrhi::ComputePipelineDesc()
        .setComputeShader(computeShader)
        .addBindingLayout(bindingLayout);

    auto computePipeline = device->createComputePipeline(computePipelineDesc);
    
    // Create a command list and begin recording

    nvrhi::CommandListHandle commandList = device->createCommandList();
    commandList->open();

    // Fill the input buffer with some numbers and compute the expected result of shader operation

    uint32_t inputData[numInputValues];
    uint32_t expectedResult = 0;
    for (uint32_t i = 0; i < numInputValues; ++i)
    {
        inputData[i] = i + 1;
        expectedResult += inputData[i];
    }
    commandList->writeBuffer(inputBuffer, inputData, sizeof(inputData));

    // Run the shader

    auto state = nvrhi::ComputeState()
        .setPipeline(computePipeline)
        .addBindingSet(bindingSet);
    commandList->setComputeState(state);
    commandList->dispatch(1, 1, 1);

    // Copy the shader output into the staging buffer

    commandList->copyBuffer(readbackBuffer, 0, outputBuffer, 0, readbackBufferDesc.byteSize);

    // Close and execute the command list, wait on the CPU side for it to be finished

    commandList->close();
    device->executeCommandList(commandList);
    device->waitForIdle();

    // Read the shader output

    uint32_t const* outputData = static_cast<uint32_t const*>(device->mapBuffer(readbackBuffer, nvrhi::CpuAccessMode::Read));
    uint32_t computedResult = *outputData;
    device->unmapBuffer(readbackBuffer);

    // Compre the result to the expected one to see if the test passes

    printf("Expected result: %d, computed result: %d\n", expectedResult, computedResult);
    if (computedResult == expectedResult)
    {
        printf("Test PASSED\n");
        return true;
    }
    else
    {
        printf("Test FAILED!\n");
        return false;
    }
}


int main(int argc, const char** argv)
{
    log::ConsoleApplicationMode();
#ifndef _DEBUG
    log::SetMinSeverity(log::Severity::Warning);
#endif

    nvrhi::GraphicsAPI api = app::GetGraphicsAPIFromCommandLine(argc, argv);
    app::DeviceManager* deviceManager = app::DeviceManager::Create(api);

    app::DeviceCreationParameters deviceParams;
#ifdef _DEBUG
    deviceParams.enableDebugRuntime = true; 
    deviceParams.enableNvrhiValidationLayer = true;
#endif

    if (!deviceManager->CreateHeadlessDevice(deviceParams))
    {
        log::fatal("Cannot initialize a graphics device with the requested parameters");
        return 1;
    }

    printf("Using %s API with %s.\n", nvrhi::utils::GraphicsAPIToString(api), deviceManager->GetRendererString());
    
    if (!RunTest(deviceManager->GetDevice()))
        return 1;

    deviceManager->Shutdown();

    delete deviceManager;

    return 0;
}
