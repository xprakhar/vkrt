#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>

#include <glm/glm.hpp>

#include <string>
#include <vector>

#include <queue>
#include <functional>
#include <iostream>
#include <sstream>
#include <fstream>
#include <stdexcept>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/matrix_clip_space.hpp>

#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

template <vk::QueueFlagBits QueueFlagBit>
struct WorkerPool{

    vk::Device device;
    vk::PhysicalDevice physicalDevice;

    uint32_t queueCount;
    uint32_t queueFamily;
    

};

struct transfer_scheduler
{

    vk::Device device;
    vk::PhysicalDevice physical_device;

    uint32_t queue_count;
    uint32_t queue_family;

    struct per_queue_data
    {
        vk::Queue queue;
        vk::CommandPool command_pool;
    };
    std::vector<per_queue_data> queues;

    struct job
    {
        vk::Fence fence;
        vk::CommandBuffer command_buffer;
        std::function<void(vk::CommandBuffer)> payload;

        job() = default;

        job(vk::Fence fence, vk::CommandBuffer command_buffer, std::function<void(vk::CommandBuffer)> payload)
            : fence(fence), command_buffer(command_buffer), payload(payload)
        {
        }

        void execute() {
            vk::CommandBufferBeginInfo begin_info = {
                .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit};

            command_buffer.begin(&begin_info);

            payload(command_buffer);

            command_buffer.end();
        }
    };
    std::unordered_map<uint64_t, job> jobs;

    std::unordered_map<uint64_t, uint32_t> job_queue_map;

    transfer_scheduler(vk::PhysicalDevice hardware_device, vk::Device logical_device)
        : physical_device(hardware_device), device(logical_device), queue_family(-1), queue_count(0)
    {
        auto queue_families = physical_device.getQueueFamilyProperties();

        for (uint32_t i = 0; i < queue_families.size(); i++)
        {
            if (queue_families[i].queueFlags & vk::QueueFlagBits::eTransfer && queue_families[i].queueCount > queue_count)
            {
                queue_family = i;
                queue_count = queue_families[i].queueCount;
            }
        }

        if (queue_family == -1)
        {
            throw std::runtime_error("Failed to find transfer queue family");
        }

        queues.resize(queue_count);

        for (uint32_t i = 0; i < queue_count; i++)
        {
            queues[i].queue = device.getQueue(queue_family, i);

            vk::CommandPoolCreateInfo pool_info = {
                .flags = vk::CommandPoolCreateFlagBits::eTransient | vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                .queueFamilyIndex = queue_family};

            queues[i].command_pool = device.createCommandPool(pool_info);

        }
    }

    job create_job(uint32_t queue_index, std::function<void(vk::CommandBuffer)> payload)
    {
        vk::FenceCreateInfo fence_info = {
            .sType = vk::StructureType::eFenceCreateInfo,
        };
        vk::Fence fence = device.createFence(fence_info);

        vk::CommandBufferAllocateInfo alloc_info = {
            .sType = vk::StructureType::eCommandBufferAllocateInfo,
            .commandPool = queues[queue_index].command_pool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = 1};

        vk::CommandBuffer command_buffer;
        device.allocateCommandBuffers(&alloc_info, &command_buffer);

        return job(fence, command_buffer, payload);
    }

    uint64_t generate_unique_id() {
        static uint64_t id = 0;
        return id++;
    }

    uint64_t schedule(std::function<void(vk::CommandBuffer)> payload)
    {
        // round robin scheduling

        static uint32_t current_queue = 0;

        auto id = generate_unique_id();
        jobs[id] = create_job(current_queue, payload);

        jobs[id].execute();

        vk::SubmitInfo submit_info = {
            .commandBufferCount = 1,
            .pCommandBuffers = &jobs[id].command_buffer};
        queues[current_queue].queue.submit(1, &submit_info, jobs[id].fence);

        job_queue_map[id] = current_queue;

        current_queue = (current_queue + 1) % queue_count;

        return id;
    }

    void wait(uint32_t job_count, const uint64_t *ids)
    {
        std::vector<vk::Fence> fences;
        for (uint32_t i = 0; i < job_count; i++)
        {
            fences.push_back(jobs[ids[i]].fence);
        }

        device.waitForFences(job_count, fences.data(), VK_TRUE, std::numeric_limits<uint64_t>::max());

        for (uint32_t i = 0; i < job_count; i++)
        {
            device.destroyFence(fences[i]);
            device.freeCommandBuffers(queues[job_queue_map[ids[i]]].command_pool, 1, &jobs[ids[i]].command_buffer);
            jobs.erase(ids[i]);
        }
    }
};

GLFWwindow *g_Window;
vk::Instance g_Instance;
vk::DebugUtilsMessengerEXT g_DebugUtilsMessenger;
vk::SurfaceKHR g_Surface;
vk::PhysicalDevice g_PhysicalDevice;
vk::Device g_Device;
vk::Queue g_GraphicsQueue;
vk::Queue g_PresentQueue;
vk::SwapchainKHR g_Swapchain;
vk::RenderPass g_RenderPass;
vk::PipelineLayout g_PipelineLayout;
vk::Pipeline g_GraphicsPipeline;
vk::CommandPool g_CommandPool;
vk::Buffer g_VertexBuffer;
vk::DeviceMemory g_VertexBufferMemory;
vk::Buffer g_IndexBuffer;
vk::DeviceMemory g_IndexBufferMemory;
vk::DescriptorSetLayout g_DescriptorSetLayout;
vk::DescriptorPool g_DescriptorPool;

uint32_t g_GraphicsQueueFamily = -1;
uint32_t g_PresentQueueFamily = -1;

const uint32_t MAX_FRAMES_IN_FLIGHT = 2u;

std::vector<vk::Image> g_SwapchainImages;
std::vector<vk::ImageView> g_SwapchainImageViews;
std::vector<vk::Framebuffer> g_SwapchainFramebuffers;
std::vector<vk::CommandBuffer> g_CommandBuffers;
std::vector<vk::Semaphore> g_ImageAvailableSemaphores;
std::vector<vk::Semaphore> g_RenderFinishedSemaphores;
std::vector<vk::Fence> g_InFlightFences;
std::vector<vk::Buffer> g_UniformBuffers;
std::vector<vk::DeviceMemory> g_UniformBuffersMemory;
std::vector<void *> g_MappedUniformBuffers;
std::vector<vk::DescriptorSet> g_DescriptorSets;

struct UniformBufferObject
{
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
};

struct Vertex
{
    glm::vec2 pos;
    glm::vec3 color;

    static vk::VertexInputBindingDescription getBindingDescription()
    {
        vk::VertexInputBindingDescription bindingDescription{
            .binding = 0,
            .stride = sizeof(Vertex),
            .inputRate = vk::VertexInputRate::eVertex};

        return bindingDescription;
    }

    static auto getAttributeDescription()
    {
        std::array<vk::VertexInputAttributeDescription, 2> attributeDescriptions{
            vk::VertexInputAttributeDescription{
                .location = 0,
                .binding = 0,
                .format = vk::Format::eR32G32Sfloat,
                .offset = offsetof(Vertex, pos)},
            vk::VertexInputAttributeDescription{
                .location = 1,
                .binding = 0,
                .format = vk::Format::eR32G32B32Sfloat,
                .offset = offsetof(Vertex, color)}};

        return attributeDescriptions;
    }
};

const std::vector<Vertex> vertices = {
    {{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
    {{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
    {{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
    {{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}}};

const std::vector<uint16_t> indices = {0, 1, 2, 2, 3, 0};

struct surface_support_details
{
    vk::SurfaceCapabilitiesKHR capabilities;
    std::vector<vk::SurfaceFormatKHR> formats;
    std::vector<vk::PresentModeKHR> presentModes;
} g_SurfaceSupportDetails;

uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties)
{
    vk::PhysicalDeviceMemoryProperties memProperties = g_PhysicalDevice.getMemoryProperties();

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
    {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
        {
            return i;
        }
    }

    throw std::runtime_error("Failed to find suitable memory type");
}

void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, vk::Buffer &buffer, vk::DeviceMemory &bufferMemory)
{
    vk::BufferCreateInfo bufferInfo = {
        .sType = vk::StructureType::eBufferCreateInfo,
        .size = size,
        .usage = usage,
        .sharingMode = vk::SharingMode::eExclusive};

    buffer = g_Device.createBuffer(bufferInfo);

    vk::MemoryRequirements memRequirements = g_Device.getBufferMemoryRequirements(buffer);

    vk::MemoryAllocateInfo allocInfo = {
        .sType = vk::StructureType::eMemoryAllocateInfo,
        .allocationSize = memRequirements.size,
        .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties)};

    bufferMemory = g_Device.allocateMemory(allocInfo);
    g_Device.bindBufferMemory(buffer, bufferMemory, 0);
}

void copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size)
{
    vk::CommandBufferAllocateInfo allocInfo = {
        .sType = vk::StructureType::eCommandBufferAllocateInfo,
        .commandPool = g_CommandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1};

    vk::CommandBuffer commandBuffer = g_Device.allocateCommandBuffers(allocInfo)[0];

    vk::CommandBufferBeginInfo beginInfo = {
        .sType = vk::StructureType::eCommandBufferBeginInfo,
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit};

    commandBuffer.begin(beginInfo);

    vk::BufferCopy copyRegion = {
        .size = size};

    commandBuffer.copyBuffer(srcBuffer, dstBuffer, 1, &copyRegion);

    commandBuffer.end();

    vk::SubmitInfo submitInfo = {
        .sType = vk::StructureType::eSubmitInfo,
        .commandBufferCount = 1,
        .pCommandBuffers = &commandBuffer};

    g_GraphicsQueue.submit(1, &submitInfo, nullptr);
    g_GraphicsQueue.waitIdle();

    g_Device.freeCommandBuffers(g_CommandPool, commandBuffer);
}

void createVertexBuffer()
{
    vk::DeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

    vk::Buffer stagingBuffer;
    vk::DeviceMemory stagingBufferMemory;
    createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

    void *data = g_Device.mapMemory(stagingBufferMemory, 0, bufferSize);
    memcpy(data, vertices.data(), (size_t)bufferSize);
    g_Device.unmapMemory(stagingBufferMemory);

    createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, g_VertexBuffer, g_VertexBufferMemory);

    copyBuffer(stagingBuffer, g_VertexBuffer, bufferSize);

    g_Device.destroyBuffer(stagingBuffer);
    g_Device.freeMemory(stagingBufferMemory);
}

void createIndexBuffer()
{
    vk::DeviceSize bufferSize = sizeof(indices[0]) * indices.size();

    vk::Buffer stagingBuffer;
    vk::DeviceMemory stagingBufferMemory;
    createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

    void *data = g_Device.mapMemory(stagingBufferMemory, 0, bufferSize);
    memcpy(data, indices.data(), (size_t)bufferSize);
    g_Device.unmapMemory(stagingBufferMemory);

    createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, g_IndexBuffer, g_IndexBufferMemory);

    copyBuffer(stagingBuffer, g_IndexBuffer, bufferSize);

    g_Device.destroyBuffer(stagingBuffer);
    g_Device.freeMemory(stagingBufferMemory);
}

void createUniformBuffer()
{
    vk::DeviceSize bufferSize = sizeof(UniformBufferObject);

    g_UniformBuffers.resize(g_SwapchainImages.size());
    g_UniformBuffersMemory.resize(g_SwapchainImages.size());
    g_MappedUniformBuffers.resize(g_SwapchainImages.size());

    for (size_t i = 0; i < g_SwapchainImages.size(); i++)
    {
        createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, g_UniformBuffers[i], g_UniformBuffersMemory[i]);
        g_MappedUniformBuffers[i] = g_Device.mapMemory(g_UniformBuffersMemory[i], 0, bufferSize);
    }
}

void createDescriptorSetLayout()
{
    vk::DescriptorSetLayoutBinding uboLayoutBinding = {
        .binding = 0,
        .descriptorType = vk::DescriptorType::eUniformBuffer,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eVertex};

    vk::DescriptorSetLayoutCreateInfo layoutInfo = {
        .sType = vk::StructureType::eDescriptorSetLayoutCreateInfo,
        .bindingCount = 1,
        .pBindings = &uboLayoutBinding};

    g_DescriptorSetLayout = g_Device.createDescriptorSetLayout(layoutInfo);
}

void glfwErrorCallback(int error, const char *description)
{
    std::cerr << "GLFW Error: " << description << std::endl;
}

VKAPI_ATTR VkBool32 VKAPI_CALL debugMessengerCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
    void *pUserData)
{
    std::ostringstream message;

    message << vk::to_string(static_cast<vk::DebugUtilsMessageSeverityFlagBitsEXT>(messageSeverity)) << ": "
            << vk::to_string(static_cast<vk::DebugUtilsMessageTypeFlagsEXT>(messageType)) << ":\n";
    message << std::string("\t") << "messageIDName   = <" << pCallbackData->pMessageIdName << ">\n";
    message << std::string("\t") << "messageIdNumber = " << pCallbackData->messageIdNumber << "\n";
    message << std::string("\t") << "message         = <" << pCallbackData->pMessage << ">\n";
    if (0 < pCallbackData->queueLabelCount)
    {
        message << std::string("\t") << "Queue Labels:\n";
        for (uint32_t i = 0; i < pCallbackData->queueLabelCount; i++)
        {
            message << std::string("\t\t") << "labelName = <" << pCallbackData->pQueueLabels[i].pLabelName << ">\n";
        }
    }
    if (0 < pCallbackData->cmdBufLabelCount)
    {
        message << std::string("\t") << "CommandBuffer Labels:\n";
        for (uint32_t i = 0; i < pCallbackData->cmdBufLabelCount; i++)
        {
            message << std::string("\t\t") << "labelName = <" << pCallbackData->pCmdBufLabels[i].pLabelName << ">\n";
        }
    }
    if (0 < pCallbackData->objectCount)
    {
        message << std::string("\t") << "Objects:\n";
        for (uint32_t i = 0; i < pCallbackData->objectCount; i++)
        {
            message << std::string("\t\t") << "Object " << i << "\n";
            message << std::string("\t\t\t") << "objectType   = " << vk::to_string(static_cast<vk::ObjectType>(pCallbackData->pObjects[i].objectType)) << "\n";
            message << std::string("\t\t\t") << "objectHandle = " << pCallbackData->pObjects[i].objectHandle << "\n";
            if (pCallbackData->pObjects[i].pObjectName)
            {
                message << std::string("\t\t\t") << "objectName   = <" << pCallbackData->pObjects[i].pObjectName << ">\n";
            }
        }
    }

    std::cerr << message.str() << std::endl;

    return VK_FALSE;
}

struct swapchain_create_info
{
    vk::SurfaceFormatKHR format;
    vk::PresentModeKHR presentMode;
    vk::Extent2D extent;
    uint32_t imageCount;
    vk::SwapchainKHR oldSwapchain;
};

vk::SurfaceFormatKHR chooseSurfaceFormat(const std::vector<vk::SurfaceFormatKHR> &formats)
{
    if (formats.size() == 1 && formats[0].format == vk::Format::eUndefined)
    {
        return {vk::Format::eB8G8R8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear};
    }

    for (const auto &format : formats)
    {
        if (format.format == vk::Format::eB8G8R8A8Unorm && format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
        {
            return format;
        }
    }

    return formats[0];
}

vk::PresentModeKHR choose_present_mode(const std::vector<vk::PresentModeKHR> &presentModes)
{
    for (const auto &presentMode : presentModes)
    {
        if (presentMode == vk::PresentModeKHR::eMailbox)
        {
            return presentMode;
        }
    }

    return vk::PresentModeKHR::eFifo;
}

vk::Extent2D chooseExtent(const vk::SurfaceCapabilitiesKHR &capabilities, GLFWwindow *window)
{
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
    {
        return capabilities.currentExtent;
    }

    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    vk::Extent2D actualExtent = {
        static_cast<uint32_t>(width),
        static_cast<uint32_t>(height)};

    actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
    actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

    return actualExtent;
}

std::vector<char> readCodeFromFile(const std::string &filename)
{
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();

    return buffer;
}

void createInstance()
{
    vk::DynamicLoader dl;
    VULKAN_HPP_DEFAULT_DISPATCHER.init(dl);
    uint32_t glfwExtensionCount = 0;
    const char **glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    std::vector<const char *> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

    vk::ApplicationInfo appInfo = {
        .sType = vk::StructureType::eApplicationInfo,
        .pApplicationName = "Ray Tracer",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "No Engine",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = VK_API_VERSION_1_3};

    vk::InstanceCreateInfo instanceInfo = {
        .sType = vk::StructureType::eInstanceCreateInfo,
        .pApplicationInfo = &appInfo,
        .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
        .ppEnabledExtensionNames = extensions.data()};

#if defined(_DEBUG)
    std::vector<const char *> layers{
        "VK_LAYER_KHRONOS_validation"};
    instanceInfo.enabledLayerCount = static_cast<uint32_t>(layers.size());
    instanceInfo.ppEnabledLayerNames = layers.data();

    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    instanceInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    instanceInfo.ppEnabledExtensionNames = extensions.data();

    vk::DebugUtilsMessengerCreateInfoEXT messengerInfo = {
        .sType = vk::StructureType::eDebugUtilsMessengerCreateInfoEXT,
        .messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eError | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning,
        .messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
        .pfnUserCallback = debugMessengerCallback};

    instanceInfo.pNext = &messengerInfo;
#endif

    g_Instance = vk::createInstance(instanceInfo);
    VULKAN_HPP_DEFAULT_DISPATCHER.init(g_Instance);
}

void createDebugUtilsMessenger()
{
#if defined(_DEBUG)
    vk::DebugUtilsMessengerCreateInfoEXT messengerInfo = {
        .sType = vk::StructureType::eDebugUtilsMessengerCreateInfoEXT,
        .messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eError | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning,
        .messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
        .pfnUserCallback = debugMessengerCallback};
    g_DebugUtilsMessenger = g_Instance.createDebugUtilsMessengerEXT(messengerInfo);
#endif
}

void initGlfwWindow()
{
    glfwSetErrorCallback(glfwErrorCallback);
    if (!glfwInit())
    {
        throw std::runtime_error("Failed to initialize GLFW");
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    g_Window = glfwCreateWindow(800, 600, "Ray Tracer", nullptr, nullptr);
    if (!g_Window)
    {
        throw std::runtime_error("Failed to create window");
    }
}

void selectPhysicalDevice()
{
    auto gpus = g_Instance.enumeratePhysicalDevices();
    if (gpus.empty())
    {
        throw std::runtime_error("Failed to find GPUs with Vulkan support");
    }

    g_PhysicalDevice = gpus[0];

    g_SurfaceSupportDetails.capabilities = g_PhysicalDevice.getSurfaceCapabilitiesKHR(g_Surface);
    g_SurfaceSupportDetails.formats = g_PhysicalDevice.getSurfaceFormatsKHR(g_Surface);
    g_SurfaceSupportDetails.presentModes = g_PhysicalDevice.getSurfacePresentModesKHR(g_Surface);

    if (g_SurfaceSupportDetails.formats.empty() || g_SurfaceSupportDetails.presentModes.empty())
    {
        throw std::runtime_error("Failed to find surface formats or present modes");
    }
}

void createDevice()
{
    auto queueFamilies = g_PhysicalDevice.getQueueFamilyProperties();
    for (uint32_t i = 0; i < queueFamilies.size(); i++)
    {
        if (queueFamilies[i].queueFlags & vk::QueueFlagBits::eGraphics)
        {
            g_GraphicsQueueFamily = i;
        }

        if (g_PhysicalDevice.getSurfaceSupportKHR(i, g_Surface))
        {
            g_PresentQueueFamily = i;
        }

        if (g_GraphicsQueueFamily != -1 && g_PresentQueueFamily != -1)
        {
            break;
        }
    }

    if (g_GraphicsQueueFamily == -1 || g_PresentQueueFamily == -1)
    {
        throw std::runtime_error("Failed to find queue families");
    }

    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
    std::vector<float> queuePriorities = {1.0f};

    vk::DeviceQueueCreateInfo graphicsQueueCreateInfo = {
        .sType = vk::StructureType::eDeviceQueueCreateInfo,
        .queueFamilyIndex = g_GraphicsQueueFamily,
        .queueCount = 1,
        .pQueuePriorities = queuePriorities.data()};

    queueCreateInfos.push_back(graphicsQueueCreateInfo);

    if (g_GraphicsQueueFamily != g_PresentQueueFamily)
    {
        vk::DeviceQueueCreateInfo presentQueueCreateInfo = {
            .sType = vk::StructureType::eDeviceQueueCreateInfo,
            .queueFamilyIndex = g_PresentQueueFamily,
            .queueCount = 1,
            .pQueuePriorities = queuePriorities.data()};

        queueCreateInfos.push_back(presentQueueCreateInfo);
    }

    std::vector<const char *> deviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    vk::DeviceCreateInfo deviceCreateInfo = {
        .sType = vk::StructureType::eDeviceCreateInfo,
        .queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size()),
        .pQueueCreateInfos = queueCreateInfos.data(),
        .enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()),
        .ppEnabledExtensionNames = deviceExtensions.data()};

    g_Device = g_PhysicalDevice.createDevice(deviceCreateInfo);
    VULKAN_HPP_DEFAULT_DISPATCHER.init(g_Device);

    g_GraphicsQueue = g_Device.getQueue(g_GraphicsQueueFamily, 0);
    g_PresentQueue = g_Device.getQueue(g_PresentQueueFamily, 0);
}

void createSwapchain()
{
    vk::SurfaceFormatKHR surfaceFormat = chooseSurfaceFormat(g_SurfaceSupportDetails.formats);
    vk::PresentModeKHR presentMode = choose_present_mode(g_SurfaceSupportDetails.presentModes);
    vk::Extent2D extent = chooseExtent(g_SurfaceSupportDetails.capabilities, g_Window);

    uint32_t imageCount = g_SurfaceSupportDetails.capabilities.minImageCount + 1;
    if (g_SurfaceSupportDetails.capabilities.maxImageCount > 0 && imageCount > g_SurfaceSupportDetails.capabilities.maxImageCount)
    {
        imageCount = g_SurfaceSupportDetails.capabilities.maxImageCount;
    }

    vk::SwapchainCreateInfoKHR swapchainCreateInfo = {
        .sType = vk::StructureType::eSwapchainCreateInfoKHR,
        .surface = g_Surface,
        .minImageCount = imageCount,
        .imageFormat = surfaceFormat.format,
        .imageColorSpace = surfaceFormat.colorSpace,
        .imageExtent = extent,
        .imageArrayLayers = 1,
        .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
        .imageSharingMode = vk::SharingMode::eExclusive,
        .preTransform = g_SurfaceSupportDetails.capabilities.currentTransform,
        .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
        .presentMode = presentMode,
        .clipped = VK_FALSE,
        .oldSwapchain = VK_NULL_HANDLE};

    g_Swapchain = g_Device.createSwapchainKHR(swapchainCreateInfo);

    g_SwapchainImages = g_Device.getSwapchainImagesKHR(g_Swapchain);

    g_SwapchainImageViews.resize(g_SwapchainImages.size());
    for (size_t i = 0; i < g_SwapchainImages.size(); i++)
    {
        vk::ImageViewCreateInfo createInfo = {
            .sType = vk::StructureType::eImageViewCreateInfo,
            .image = g_SwapchainImages[i],
            .viewType = vk::ImageViewType::e2D,
            .format = surfaceFormat.format,
            .components = {
                .r = vk::ComponentSwizzle::eIdentity,
                .g = vk::ComponentSwizzle::eIdentity,
                .b = vk::ComponentSwizzle::eIdentity,
                .a = vk::ComponentSwizzle::eIdentity},
            .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1}};

        g_SwapchainImageViews[i] = g_Device.createImageView(createInfo);
    }
}

void updateUniformBuffer(uint32_t currentImage)
{
    static auto startTime = std::chrono::high_resolution_clock::now();

    auto currentTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

    UniformBufferObject ubo = {
        .model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
        .view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
        .proj = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 10.0f)};

    ubo.proj[1][1] *= -1;

    memcpy(g_MappedUniformBuffers[currentImage], &ubo, sizeof(ubo));
}

void createRenderpass()
{
    vk::SurfaceFormatKHR surfaceFormat = chooseSurfaceFormat(g_SurfaceSupportDetails.formats);

    vk::AttachmentDescription colorAttachment = {
        .format = surfaceFormat.format,
        .samples = vk::SampleCountFlagBits::e1,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
        .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
        .initialLayout = vk::ImageLayout::eUndefined,
        .finalLayout = vk::ImageLayout::ePresentSrcKHR};

    vk::AttachmentReference colorAttachmentRef = {
        .attachment = 0,
        .layout = vk::ImageLayout::eColorAttachmentOptimal};

    vk::SubpassDescription subpass = {
        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
        .colorAttachmentCount = 1,
        .pColorAttachments = &colorAttachmentRef};

    vk::SubpassDependency dependencies[] = {
        {.srcSubpass = VK_SUBPASS_EXTERNAL,
         .dstSubpass = 0,
         .srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
         .dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
         .srcAccessMask = vk::AccessFlagBits::eNoneKHR,
         .dstAccessMask = vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite,
         .dependencyFlags = vk::DependencyFlagBits::eByRegion},
        {.srcSubpass = 0,
         .dstSubpass = VK_SUBPASS_EXTERNAL,
         .srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
         .dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
         .srcAccessMask = vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite,
         .dstAccessMask = vk::AccessFlagBits::eNoneKHR,
         .dependencyFlags = vk::DependencyFlagBits::eByRegion}};

    vk::RenderPassCreateInfo renderPassInfo = {
        .sType = vk::StructureType::eRenderPassCreateInfo,
        .attachmentCount = 1,
        .pAttachments = &colorAttachment,
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = 2,
        .pDependencies = dependencies};

    g_RenderPass = g_Device.createRenderPass(renderPassInfo);
}

void createPipelineLayout()
{
    vk::PipelineLayoutCreateInfo pipelineLayoutInfo = {
        .sType = vk::StructureType::ePipelineLayoutCreateInfo,
        .setLayoutCount = 1,
        .pSetLayouts = &g_DescriptorSetLayout};

    g_PipelineLayout = g_Device.createPipelineLayout(pipelineLayoutInfo);
}

void createGraphicePipeline()
{

    auto vertShaderCode = readCodeFromFile("C:/Users/prakh/source/repos/dxrt/shaders/vert.spv");
    vk::ShaderModuleCreateInfo vertShaderModuleInfo = {
        .sType = vk::StructureType::eShaderModuleCreateInfo,
        .codeSize = vertShaderCode.size(),
        .pCode = reinterpret_cast<const uint32_t *>(vertShaderCode.data())};

    vk::ShaderModule vertShaderModule = g_Device.createShaderModule(vertShaderModuleInfo);

    auto fragShaderCode = readCodeFromFile("C:/Users/prakh/source/repos/dxrt/shaders/frag.spv");
    vk::ShaderModuleCreateInfo fragShaderModuleInfo = {
        .sType = vk::StructureType::eShaderModuleCreateInfo,
        .codeSize = fragShaderCode.size(),
        .pCode = reinterpret_cast<const uint32_t *>(fragShaderCode.data())};

    vk::ShaderModule fragShaderModule = g_Device.createShaderModule(fragShaderModuleInfo);

    vk::PipelineShaderStageCreateInfo shaderStages[] = {
        {.sType = vk::StructureType::ePipelineShaderStageCreateInfo,
         .stage = vk::ShaderStageFlagBits::eVertex,
         .module = vertShaderModule,
         .pName = "main"},
        {.sType = vk::StructureType::ePipelineShaderStageCreateInfo,
         .stage = vk::ShaderStageFlagBits::eFragment,
         .module = fragShaderModule,
         .pName = "main"}};

    auto bindingDescription = Vertex::getBindingDescription();
    auto attributeDescriptions = Vertex::getAttributeDescription();

    vk::PipelineVertexInputStateCreateInfo vertexInputInfo = {
        .sType = vk::StructureType::ePipelineVertexInputStateCreateInfo,
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &bindingDescription,
        .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
        .pVertexAttributeDescriptions = attributeDescriptions.data()};

    vk::PipelineInputAssemblyStateCreateInfo inputAssembly = {
        .sType = vk::StructureType::ePipelineInputAssemblyStateCreateInfo,
        .topology = vk::PrimitiveTopology::eTriangleList,
        .primitiveRestartEnable = VK_FALSE};

    vk::PipelineViewportStateCreateInfo viewportState = {
        .sType = vk::StructureType::ePipelineViewportStateCreateInfo,
        .viewportCount = 1,
        .scissorCount = 1};

    vk::PipelineRasterizationStateCreateInfo rasterizer = {
        .sType = vk::StructureType::ePipelineRasterizationStateCreateInfo,
        .polygonMode = vk::PolygonMode::eFill,
        .cullMode = vk::CullModeFlagBits::eBack,
        .frontFace = vk::FrontFace::eCounterClockwise,
        .lineWidth = 1.0f};

    vk::PipelineMultisampleStateCreateInfo multisampling = {
        .sType = vk::StructureType::ePipelineMultisampleStateCreateInfo,
        .rasterizationSamples = vk::SampleCountFlagBits::e1,
        .sampleShadingEnable = VK_FALSE};

    vk::PipelineColorBlendAttachmentState colorBlendAttachment = {
        .blendEnable = VK_FALSE,
        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA};

    vk::PipelineColorBlendStateCreateInfo colorBlending = {
        .sType = vk::StructureType::ePipelineColorBlendStateCreateInfo,
        .logicOpEnable = VK_FALSE,
        .attachmentCount = 1,
        .pAttachments = &colorBlendAttachment};

    std::vector<vk::DynamicState> dynamicStates = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};

    vk::PipelineDynamicStateCreateInfo dynamicState = {
        .sType = vk::StructureType::ePipelineDynamicStateCreateInfo,
        .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
        .pDynamicStates = dynamicStates.data()};

    vk::GraphicsPipelineCreateInfo pipelineInfo = {
        .sType = vk::StructureType::eGraphicsPipelineCreateInfo,
        .stageCount = 2,
        .pStages = shaderStages,
        .pVertexInputState = &vertexInputInfo,
        .pInputAssemblyState = &inputAssembly,
        .pViewportState = &viewportState,
        .pRasterizationState = &rasterizer,
        .pMultisampleState = &multisampling,
        .pColorBlendState = &colorBlending,
        .pDynamicState = &dynamicState,
        .layout = g_PipelineLayout,
        .renderPass = g_RenderPass,
        .subpass = 0};

    auto result = g_Device.createGraphicsPipeline(vk::PipelineCache(), pipelineInfo);
    if (result.result != vk::Result::eSuccess)
    {
        throw std::runtime_error("Failed to create graphics pipeline");
    }

    g_GraphicsPipeline = result.value;

    g_Device.destroyShaderModule(vertShaderModule);
    g_Device.destroyShaderModule(fragShaderModule);
}

void createDescriptorPool()
{
    vk::DescriptorPoolSize poolSize = {
        .type = vk::DescriptorType::eUniformBuffer,
        .descriptorCount = static_cast<uint32_t>(g_SwapchainImages.size())};

    vk::DescriptorPoolCreateInfo poolInfo = {
        .sType = vk::StructureType::eDescriptorPoolCreateInfo,
        .maxSets = static_cast<uint32_t>(g_SwapchainImages.size()),
        .poolSizeCount = 1,
        .pPoolSizes = &poolSize};

    g_DescriptorPool = g_Device.createDescriptorPool(poolInfo);
}

void createFrambuffers()
{
    vk::SurfaceCapabilitiesKHR capabilities = g_PhysicalDevice.getSurfaceCapabilitiesKHR(g_Surface);
    vk::Extent2D extent = chooseExtent(capabilities, g_Window);
    g_SwapchainFramebuffers.resize(g_SwapchainImageViews.size());
    for (size_t i = 0; i < g_SwapchainImageViews.size(); i++)
    {
        vk::FramebufferCreateInfo framebufferInfo = {
            .sType = vk::StructureType::eFramebufferCreateInfo,
            .renderPass = g_RenderPass,
            .attachmentCount = 1,
            .pAttachments = &g_SwapchainImageViews[i],
            .width = extent.width,
            .height = extent.height,
            .layers = 1};

        g_SwapchainFramebuffers[i] = g_Device.createFramebuffer(framebufferInfo);
    }
}

void createPerFrameData()
{
    g_CommandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    vk::CommandBufferAllocateInfo allocInfo = {
        .sType = vk::StructureType::eCommandBufferAllocateInfo,
        .commandPool = g_CommandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = static_cast<uint32_t>(g_CommandBuffers.size())};
    auto result2 = g_Device.allocateCommandBuffers(&allocInfo, g_CommandBuffers.data());
    if (result2 != vk::Result::eSuccess)
    {
        throw std::runtime_error("Failed to allocate command buffers");
    }

    g_ImageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    g_RenderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    g_InFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

    vk::SemaphoreCreateInfo semaphoreInfo = {
        .sType = vk::StructureType::eSemaphoreCreateInfo};
    vk::FenceCreateInfo fenceInfo = {
        .sType = vk::StructureType::eFenceCreateInfo,
        .flags = vk::FenceCreateFlagBits::eSignaled};

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        g_ImageAvailableSemaphores[i] = g_Device.createSemaphore(semaphoreInfo);
        g_RenderFinishedSemaphores[i] = g_Device.createSemaphore(semaphoreInfo);
        g_InFlightFences[i] = g_Device.createFence(fenceInfo);
    }
}

void createDescriptorSets()
{
    std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, g_DescriptorSetLayout);
    vk::DescriptorSetAllocateInfo allocInfo = {
        .sType = vk::StructureType::eDescriptorSetAllocateInfo,
        .descriptorPool = g_DescriptorPool,
        .descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT),
        .pSetLayouts = layouts.data()};

    g_DescriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
    auto result = g_Device.allocateDescriptorSets(&allocInfo, g_DescriptorSets.data());
    if (result != vk::Result::eSuccess)
    {
        throw std::runtime_error("Failed to allocate descriptor sets");
    }

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        vk::DescriptorBufferInfo bufferInfo = {
            .buffer = g_UniformBuffers[i],
            .offset = 0,
            .range = sizeof(UniformBufferObject)};

        vk::WriteDescriptorSet descriptorWrite = {
            .sType = vk::StructureType::eWriteDescriptorSet,
            .dstSet = g_DescriptorSets[i],
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eUniformBuffer,
            .pBufferInfo = &bufferInfo};

        g_Device.updateDescriptorSets(1, &descriptorWrite, 0, nullptr);
    }
}

void initProcess()
{

    initGlfwWindow();

    createInstance();

    createDebugUtilsMessenger();

    if (glfwCreateWindowSurface(static_cast<VkInstance>(g_Instance), g_Window, nullptr, reinterpret_cast<VkSurfaceKHR *>(&g_Surface)) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create window surface");
    }

    selectPhysicalDevice();

    createDevice();

    createSwapchain();

    vk::CommandPoolCreateInfo commandPoolInfo = {
        .sType = vk::StructureType::eCommandPoolCreateInfo,
        .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        .queueFamilyIndex = g_GraphicsQueueFamily};

    g_CommandPool = g_Device.createCommandPool(commandPoolInfo);

    createRenderpass();
    createDescriptorSetLayout();
    createUniformBuffer();
    createDescriptorPool();
    createDescriptorSets();
    createPipelineLayout();

    createGraphicePipeline();

    createFrambuffers();
    createPerFrameData();
    createVertexBuffer();
    createIndexBuffer();
}

void finalizeProcess()
{

    if (g_Device)
    {
        g_Device.waitIdle();
        for (size_t i = 0; i < g_SwapchainImages.size(); i++)
        {
            g_Device.unmapMemory(g_UniformBuffersMemory[i]);
            g_Device.destroyBuffer(g_UniformBuffers[i]);
            g_Device.freeMemory(g_UniformBuffersMemory[i]);
        }
        g_Device.destroyBuffer(g_IndexBuffer);
        g_Device.freeMemory(g_IndexBufferMemory);
        g_Device.destroyBuffer(g_VertexBuffer);
        g_Device.freeMemory(g_VertexBufferMemory);
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            g_Device.destroySemaphore(g_ImageAvailableSemaphores[i]);
            g_Device.destroySemaphore(g_RenderFinishedSemaphores[i]);
            g_Device.destroyFence(g_InFlightFences[i]);
        }
        g_Device.freeCommandBuffers(g_CommandPool, g_CommandBuffers);
        for (auto framebuffer : g_SwapchainFramebuffers)
        {
            g_Device.destroyFramebuffer(framebuffer);
        }
        for (auto imageView : g_SwapchainImageViews)
        {
            g_Device.destroyImageView(imageView);
        }
        g_Device.destroyCommandPool(g_CommandPool);
        g_Device.destroyPipeline(g_GraphicsPipeline);
        g_Device.destroyPipelineLayout(g_PipelineLayout);
        g_Device.destroyRenderPass(g_RenderPass);
        g_Device.destroySwapchainKHR(g_Swapchain);
        g_Device.destroy();
        g_Device = nullptr;
    }

    if (g_Instance)
    {
#if defined(_DEBUG)
        g_Instance.destroyDebugUtilsMessengerEXT(g_DebugUtilsMessenger);
#endif
        g_Instance.destroySurfaceKHR(g_Surface);
        g_Instance.destroy();
        g_Instance = nullptr;
    }

    if (g_Window)
    {
        glfwDestroyWindow(g_Window);
        g_Window = nullptr;
    }

    glfwTerminate();
}

void recordCommandBuffer(vk::CommandBuffer commandBuffer, uint32_t imageIndex, uint32_t currentFrame)
{
    vk::CommandBufferBeginInfo beginInfo = {
        .sType = vk::StructureType::eCommandBufferBeginInfo,
        .flags = vk::CommandBufferUsageFlagBits::eSimultaneousUse};

    commandBuffer.begin(beginInfo);

    vk::ClearValue clearColor = vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f});
    vk::RenderPassBeginInfo renderPassInfo = {
        .sType = vk::StructureType::eRenderPassBeginInfo,
        .renderPass = g_RenderPass,
        .framebuffer = g_SwapchainFramebuffers[imageIndex],
        .renderArea = {
            .offset = {0, 0},
            .extent = g_SurfaceSupportDetails.capabilities.currentExtent},
        .clearValueCount = 1,
        .pClearValues = &clearColor};

    commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, g_GraphicsPipeline);
    vk::Viewport viewport = {
        .x = 0.0f,
        .y = 0.0f,
        .width = static_cast<float>(g_SurfaceSupportDetails.capabilities.currentExtent.width),
        .height = static_cast<float>(g_SurfaceSupportDetails.capabilities.currentExtent.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f};
    commandBuffer.setViewport(0, viewport);

    vk::Rect2D scissor = {
        .offset = {0, 0},
        .extent = g_SurfaceSupportDetails.capabilities.currentExtent};
    commandBuffer.setScissor(0, scissor);

    vk::DeviceSize offsets[] = {0};

    commandBuffer.bindVertexBuffers(0, 1, &g_VertexBuffer, offsets);

    commandBuffer.bindIndexBuffer(g_IndexBuffer, 0, vk::IndexType::eUint16);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, g_PipelineLayout, 0, 1, &g_DescriptorSets[currentFrame], 0, nullptr);
    commandBuffer.drawIndexed(static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
    commandBuffer.endRenderPass();
    commandBuffer.end();
}

void recreateSwapchain()
{
    g_Device.waitIdle();

    for (auto framebuffer : g_SwapchainFramebuffers)
    {
        g_Device.destroyFramebuffer(framebuffer);
    }

    for (auto imageView : g_SwapchainImageViews)
    {
        g_Device.destroyImageView(imageView);
    }

    g_Device.destroySwapchainKHR(g_Swapchain);

    createSwapchain();

    createFrambuffers();
}

void drawFrame()
{
    static size_t currentFrame = 0;
    auto result = g_Device.waitForFences(1, &g_InFlightFences[currentFrame], VK_TRUE, std::numeric_limits<uint64_t>::max());

    uint32_t imageIndex;
    result = g_Device.acquireNextImageKHR(g_Swapchain, std::numeric_limits<uint64_t>::max(), g_ImageAvailableSemaphores[currentFrame], nullptr, &imageIndex);
    if (result == vk::Result::eErrorOutOfDateKHR)
    {
        recreateSwapchain();
        return;
    }
    else if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR)
    {
        g_GraphicsQueue.waitIdle();
    }

    result = g_Device.resetFences(1, &g_InFlightFences[currentFrame]);
    g_CommandBuffers[currentFrame].reset();
    recordCommandBuffer(g_CommandBuffers[currentFrame], imageIndex, currentFrame);

    vk::PipelineStageFlags waitStages[] = {vk::PipelineStageFlagBits::eColorAttachmentOutput};

    updateUniformBuffer(imageIndex);

    vk::SubmitInfo submitInfo = {
        .sType = vk::StructureType::eSubmitInfo,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &g_ImageAvailableSemaphores[currentFrame],
        .pWaitDstStageMask = &waitStages[0],
        .commandBufferCount = 1,
        .pCommandBuffers = &g_CommandBuffers[currentFrame],
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &g_RenderFinishedSemaphores[currentFrame]};

    result = g_GraphicsQueue.submit(1, &submitInfo, g_InFlightFences[currentFrame]);

    vk::PresentInfoKHR presentInfo = {
        .sType = vk::StructureType::ePresentInfoKHR,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &g_RenderFinishedSemaphores[currentFrame],
        .swapchainCount = 1,
        .pSwapchains = &g_Swapchain,
        .pImageIndices = &imageIndex};

    try
    {
        result = g_PresentQueue.presentKHR(presentInfo);
    }
    catch (const vk::OutOfDateKHRError &e)
    {
        std::cerr << "Out of date error" << std::endl;
        recreateSwapchain();
    }
    if (result == vk::Result::eSuboptimalKHR)
    {
        recreateSwapchain();
    }
    else if (result != vk::Result::eSuccess)
    {
        throw std::runtime_error("Failed to present swapchain image");
    }

    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

int main(int argc, char **argv)
{
    try
    {
        initProcess();

        while (!glfwWindowShouldClose(g_Window))
        {
            glfwPollEvents();
            drawFrame();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    finalizeProcess();
    return 0;
}