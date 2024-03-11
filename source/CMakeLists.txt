cmake_minimum_required(VERSION 3.11)

file(GLOB_RECURSE SOURCES "*.c" "*.cpp")
file(GLOB_RECURSE HEADERS "../include/*.h" "../include/*.hpp")

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

target_compile_definitions(${PROJECT_NAME} PRIVATE VK_NO_PROTOTYPES)

include(FetchContent)

FetchContent_Declare(
  glfw
  GIT_REPOSITORY https://github.com/glfw/glfw.git
)

FetchContent_GetProperties(glfw)
if (NOT glfw_POPULATED)
  FetchContent_Populate(glfw)
  set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
  add_subdirectory(${glfw_SOURCE_DIR} ${glfw_BINARY_DIR})
endif()

target_link_libraries(${PROJECT_NAME} PRIVATE glfw)

FetchContent_Declare(
    Vulkan-Headers
    GIT_REPOSITORY https://github.com//KhronosGroup/Vulkan-Headers.git
    GIT_TAG vulkan-sdk-1.3.275.0
)

FetchContent_MakeAvailable(Vulkan-Headers)

target_link_libraries(${PROJECT_NAME} PRIVATE Vulkan-Headers)

FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
)

FetchContent_MakeAvailable(glm)

target_link_libraries(${PROJECT_NAME} PRIVATE glm::glm)

FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
)

FetchContent_MakeAvailable(stb)