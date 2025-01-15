/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <condition_variable>
#include <filesystem>
#include <initializer_list>
#include <mutex>
#include <nvh/fileoperations.hpp>
#include <nvp/nvpfilesystem.hpp>
#include <nvvk/descriptorsets_vk.hpp>
#include <nvvkhl/glsl_compiler.hpp>
#include <sample_vulkan_objects.hpp>
#include <stdexcept>
#include <vulkan/vulkan_core.h>

#ifndef PROJECT_NAME
#error PROJECT_NAME not defined
#endif

#ifndef NVPRO_CORE_DIR
#error NVPRO_CORE_DIR not defined
#endif

// Returns the ceiling of an integer division. Assumes positive values!
template <std::integral T>
T div_ceil(const T& a, const T& b)
{
  return (a + b - 1) / b;
}

// A shaderc compiler with include dirs and default options, similar to
// nvvk::GlslCompiler but without using nvh::findFile
class SampleGlslCompiler
{
public:
  SampleGlslCompiler(std::initializer_list<std::string_view> includeDirs)
      : m_includeDirs{includeDirs.begin(), includeDirs.end()}
  {
    m_options.SetIncluder(std::make_unique<nvvkhl::GlslIncluder>(m_includeDirs));
    m_options.SetTargetSpirv(shaderc_spirv_version_1_6);
    m_options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_3);
#ifndef NDEBUG
    m_options.SetGenerateDebugInfo();
#endif
  }
  std::filesystem::path find(const std::filesystem::path& filename)
  {
    return nvh::findFile(filename.string(), m_includeDirs);
  }
  vkobj::ShaderModule compile(VkDevice                       device,
                              const std::filesystem::path&   path,
                              shaderc_shader_kind            shaderKind,
                              const shaderc::CompileOptions* options = nullptr)
  {
    std::string source = nvh::loadFile(path.string(), false);
    if(options == nullptr)
      options = &defaultOptions();
    shaderc::SpvCompilationResult binary = m_compiler.CompileGlslToSpv(source, shaderKind, path.string().c_str(), *options);
    shaderc_compilation_status status = binary.GetCompilationStatus();
    if(status != shaderc_compilation_status_success)
    {
      LOGE("Error: failed to compile %s\n%s\n", path.string().c_str(), binary.GetErrorMessage().c_str());
      return {};
    }

#if 0
    auto spvFilePath = std::filesystem::path(".") / path.filename().replace_extension(".spv");
    std::ofstream spvFile(spvFilePath, std::ios::binary);
    if(!spvFile.good())
      throw std::runtime_error("Cannot write spv file " + spvFilePath.string());
    spvFile.write(reinterpret_cast<const char*>(&*binary.begin()), (binary.end() - binary.begin()) * sizeof(*binary.begin()));
#endif

    VkShaderModuleCreateInfo shaderModuleCreateInfo{
        .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext    = nullptr,
        .flags    = 0,
        .codeSize = (binary.end() - binary.begin()) * sizeof(uint32_t),
        .pCode    = reinterpret_cast<const uint32_t*>(binary.begin()),
    };
    return vkobj::ShaderModule(device, shaderModuleCreateInfo);
  }
  const shaderc::CompileOptions& defaultOptions() const { return m_options; }
  shaderc::CompileOptions        m_options;
  shaderc::Compiler              m_compiler;
  std::vector<std::string>       m_includeDirs;
};

// A utility mostly for development. Spins printing error messages and waiting
// for file changes until the shader compiles.
inline vkobj::ShaderModule reloadUntilCompiling(VkDevice                       device,
                                                SampleGlslCompiler&            glslCompiler,
                                                const std::filesystem::path&   path,
                                                shaderc_shader_kind            shaderKind,
                                                const shaderc::CompileOptions* options = nullptr)
{
  std::filesystem::path fullPath = glslCompiler.find(path);
  if(fullPath.empty())
    throw std::runtime_error("file not found: " + path.string());
  vkobj::ShaderModule result = glslCompiler.compile(device, fullPath, shaderKind, options);
  if((VkShaderModule)result == VK_NULL_HANDLE)
  {
    std::mutex                   m;
    std::condition_variable      cv;
    std::string                  parentPath = std::filesystem::absolute(fullPath).parent_path().string();
    std::vector<std::string>     watchPaths{parentPath};
    nvp::ModifiedFilesMonitor    fsm(watchPaths, [&]([[maybe_unused]] nvp::FileSystemMonitor::EventData ev) {
      std::lock_guard<std::mutex> lk(m);
      result = glslCompiler.compile(device, fullPath, shaderKind, options);
      cv.notify_one();
    });
    std::unique_lock<std::mutex> lk(m);
    LOGW("Waiting for changes in %s\n", parentPath.c_str());
    cv.wait(lk, [&] { return (VkShaderModule)result != VK_NULL_HANDLE; });
  }
  return result;
}

namespace vkobj {

// Contains single descriptorset instance for a single list of bindings. Bindings must be allocated at construction, i.e. there is no support for first creating layouts. Far from generic, but useful for quick compute shaders and when there is only one binding instance anyway.
class SingleDescriptorSet
{
public:
  struct Binding
  {
    uint32_t                                                    index;
    VkDescriptorType                                            descriptorType;
    std::variant<VkDescriptorBufferInfo, VkDescriptorImageInfo> descriptorInfo;
  };
  SingleDescriptorSet(VkDevice device, VkShaderStageFlags stageFlags, std::initializer_list<Binding> bindingInfos)
      : m_descriptorSet(std::make_unique<nvvk::DescriptorSetContainer>(device))
  {
    VkDescriptorSetLayoutCreateFlags layoutFlags = 0;  // TODO: needed for e.g. VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR
    for(const Binding& binding : bindingInfos)
      m_descriptorSet->addBinding(binding.index, binding.descriptorType, 1, stageFlags);
    m_descriptorSet->initLayout(layoutFlags);
    m_descriptorSet->initPool(1);
    std::vector<VkWriteDescriptorSet> writes;
    writes.reserve(bindingInfos.size());
    for(const Binding& binding : bindingInfos)
    {
      std::visit([this, &writes,
                  &binding](const auto& info) { writes.push_back(m_descriptorSet->makeWrite(0, binding.index, &info)); },
                 binding.descriptorInfo);
    }
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
  }

  const VkDescriptorSetLayout& layout() const { return m_descriptorSet->getLayout(); }
  const VkDescriptorSet&       get() const { return *m_descriptorSet->getSets(0); }

private:
  // unique_ptr to allow object to be moved
  std::unique_ptr<nvvk::DescriptorSetContainer> m_descriptorSet;
};

// Simpler version of nvvk::PushComputeDispatcher, without integrated binding or
// barrier support. This is quite application-specific and does not support
// sharing the pipeline layout.
template <class PushConstants = void>
struct SimpleComputePipeline
{
  SimpleComputePipeline() = default;

  // Constructor for only push constants
  SimpleComputePipeline(VkDevice                       device,
                        SampleGlslCompiler&            glslCompiler,
                        const std::filesystem::path&   path,
                        const shaderc::CompileOptions* options = nullptr)
    requires(!std::is_void_v<PushConstants>)
  {
    vkobj::ShaderModule module =
        reloadUntilCompiling(device, glslCompiler, path, shaderc_shader_kind::shaderc_glsl_compute_shader, options);

    VkPushConstantRange        pushConstantRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants)};
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {
        .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext                  = nullptr,
        .flags                  = 0,
        .setLayoutCount         = 0u,
        .pSetLayouts            = nullptr,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges    = &pushConstantRange,
    };
    pipelineLayout = vkobj::PipelineLayout(device, pipelineLayoutCreateInfo);

    VkComputePipelineCreateInfo computePipelineCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .stage =
            VkPipelineShaderStageCreateInfo{
                .sType               = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .pNext               = nullptr,
                .flags               = 0,
                .stage               = VK_SHADER_STAGE_COMPUTE_BIT,
                .module              = module,
                .pName               = "main",
                .pSpecializationInfo = nullptr,
            },
        .layout             = pipelineLayout,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex  = 0,
    };

    VkPipeline newPipeline;
    vkCreateComputePipelines(device, nullptr, 1, &computePipelineCreateInfo, nullptr, &newPipeline);
    pipeline = vkobj::Pipeline(device, std::move(newPipeline));
  }

  // Constructor that includes a descriptor set layout, possibly also with push
  // constants
  SimpleComputePipeline(VkDevice                       device,
                        SampleGlslCompiler&            glslCompiler,
                        const std::filesystem::path&   path,
                        VkDescriptorSetLayout          descriptorsetLayout,
                        const shaderc::CompileOptions* options = nullptr)
  {
    vkobj::ShaderModule module =
        reloadUntilCompiling(device, glslCompiler, path, shaderc_shader_kind::shaderc_glsl_compute_shader, options);

    if constexpr(std::is_void_v<PushConstants>)
    {
      VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {
          .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
          .pNext                  = nullptr,
          .flags                  = 0,
          .setLayoutCount         = descriptorsetLayout ? 1u : 0u,
          .pSetLayouts            = &descriptorsetLayout,
          .pushConstantRangeCount = 0,
          .pPushConstantRanges    = nullptr,
      };
      pipelineLayout = vkobj::PipelineLayout(device, pipelineLayoutCreateInfo);
    }
    else
    {
      VkPushConstantRange        pushConstantRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants)};
      VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {
          .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
          .pNext                  = nullptr,
          .flags                  = 0,
          .setLayoutCount         = descriptorsetLayout ? 1u : 0u,
          .pSetLayouts            = &descriptorsetLayout,
          .pushConstantRangeCount = 1,
          .pPushConstantRanges    = &pushConstantRange,
      };
      pipelineLayout = vkobj::PipelineLayout(device, pipelineLayoutCreateInfo);
    }

    VkComputePipelineCreateInfo computePipelineCreateInfo = {

        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .stage =
            VkPipelineShaderStageCreateInfo{
                .sType               = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .pNext               = nullptr,
                .flags               = 0,
                .stage               = VK_SHADER_STAGE_COMPUTE_BIT,
                .module              = module,
                .pName               = "main",
                .pSpecializationInfo = nullptr,
            },
        .layout             = pipelineLayout,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex  = 0,
    };

    VkPipeline newPipeline;
    vkCreateComputePipelines(device, nullptr, 1, &computePipelineCreateInfo, nullptr, &newPipeline);
    pipeline = vkobj::Pipeline(device, std::move(newPipeline));
  }
  operator VkPipeline() const { return pipeline; }
  vkobj::PipelineLayout pipelineLayout;
  vkobj::Pipeline       pipeline;
};

}  // namespace vkobj
