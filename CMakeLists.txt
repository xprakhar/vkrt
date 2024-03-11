cmake_minimum_required(VERSION 3.11)

project(vkrt)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

add_subdirectory(source)