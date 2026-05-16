cmake_minimum_required(VERSION 3.10)
project(exp1_image_processing)

# 强制使用 g++ 编译（即使文件是 .c 也用 C++ 编译）
set(CMAKE_C_COMPILER g++)
set(CMAKE_CXX_COMPILER g++)

# 设置 C++11 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 OpenCV 4
find_package(OpenCV 4 REQUIRED)

# 源文件使用 .c 后缀
add_executable(exp1_image_processing exp1_image_processing.cpp)

# 链接 OpenCV
target_link_libraries(exp1_image_processing ${OpenCV_LIBS})

# 启用调试信息
set(CMAKE_BUILD_TYPE Debug)

# 或者通过命令行指定: cmake -DCMAKE_BUILD_TYPE=Debug ..

# 添加调试编译选项
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # GCC/Clang
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_compile_options(-g -O0)           # -g: 调试信息, -O0: 禁用优化
        add_compile_options(-Wall -Wextra)     # 显示所有警告
        add_compile_options(-fsanitize=address) # 地址 sanitizer（可选）
    endif()
    

endif()

# 输出信息
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "Compiler used: ${CMAKE_C_COMPILER}")