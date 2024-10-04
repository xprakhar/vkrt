# Vulkan Graphics Application

This project demonstrates a Vulkan-based graphics rendering engine capable of handling vertex and index buffers, custom job scheduling for asynchronous tasks, and managing GPU-CPU synchronization through transfer queues and command buffers. The application showcases basic rendering techniques with efficient memory management and synchronization, making it a strong foundation for advanced graphics programming.

## Features

- **Vulkan Graphics Pipeline**: Implements a basic graphics pipeline in Vulkan, rendering a colored rectangle using vertex and index buffers.
- **Transfer Queue Scheduler**: Includes a custom scheduler for job handling on Vulkan's transfer queues, using a round-robin assignment system to balance the load across multiple queues.
- **Vertex & Index Buffer Management**: Efficient handling of vertex and index buffers with optimal memory allocation, buffer creation, and data transfer using command buffers.
- **Asynchronous Task Execution**: Utilizes Vulkan's command buffers and fences to schedule tasks asynchronously and ensure proper synchronization between GPU and CPU operations.


## Requirements

To run this project, you'll need the following dependencies:

- **Vulkan SDK**: Ensure that the Vulkan SDK is installed on your machine and properly set up in your environment.
- **GLFW**: Used for window management and input handling.
- **GLM**: OpenGL Mathematics library for matrix and vector operations.
- **STB Image**: For texture loading (already integrated in the code).
- **CMake**: For building the project.

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/vulkan-graphics-app.git
    cd vulkan-graphics-app
    ```

2. **Install Dependencies**: Ensure the required libraries and dependencies are installed (Vulkan SDK, GLFW, GLM).

3. **Build the project**:
    ```bash
    mkdir build
    cd build
    cmake ..
    make
    ```

4. **Run the application**:
    After successful build, run the generated executable:
    ```bash
    ./VulkanGraphicsApp
    ```

## Usage

The application initializes a Vulkan instance, creates vertex and index buffers, and renders a simple colored rectangle to the window using the graphics pipeline. The following Vulkan features are showcased:

1. **Buffer Management**: Manages vertex and index buffers with efficient memory usage and buffer copying.
2. **Transfer Queue Scheduling**: Schedules tasks on Vulkan transfer queues, leveraging a custom round-robin scheduler to balance load and optimize performance.
3. **Asynchronous Command Execution**: Uses command buffers to execute rendering and transfer operations asynchronously, with fences to ensure proper synchronization.

## Custom Scheduler Design

The project introduces a **transfer_scheduler** class to handle the round-robin scheduling of command buffer execution on multiple Vulkan transfer queues. It abstracts the creation of jobs, submission of command buffers, and synchronization using fences. The scheduler can:

- Create and allocate command buffers.
- Submit jobs asynchronously to different queues.
- Synchronize tasks using fences and wait for job completion.

This design improves performance by distributing the workload across multiple transfer queues and allowing tasks to be executed asynchronously.

## Future Improvements

- **Depth Buffering**: Adding depth buffering support for 3D rendering.
- **Texturing**: Introducing texture mapping to enhance rendered objects.
- **Multithreading**: Optimizing further by introducing multithreading to handle command buffer recording and submission.
- **Lighting**: Implementing basic lighting to add realism to rendered objects.





