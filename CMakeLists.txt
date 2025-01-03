cmake_minimum_required(VERSION 3.20)

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()

#-------------------------------------------------------------------------------
# Project setup and globals
#-------------------------------------------------------------------------------

project(FeGen LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

#-------------------------------------------------------------------------------
# MLIR/LLVM Configuration
#-------------------------------------------------------------------------------
option(LLVM_INCLUDE_TOOLS "Generate build targets for the LLVM tools." ON)
option(LLVM_BUILD_TOOLS "Build the LLVM tools. If OFF, just generate build targets." ON)

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

if(NOT DEFINED LLVM_MAIN_SRC_DIR)
  get_filename_component(LLVM_MAIN_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/llvm/llvm ABSOLUTE)
endif()
set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

if(NOT DEFINED LLVM_MAIN_SRC_DIR)
  get_filename_component(LLVM_MAIN_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/llvm/llvm ABSOLUTE)
endif()
set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")


set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(MLIR_INCLUDE_DIR ${MLIR_MAIN_SRC_DIR}/include)

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)


# Add MLIR and LLVM headers to the include path
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})


#-------------------------------------------------------------------------------
# Initialize Python packages
#-------------------------------------------------------------------------------

find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Using Python3Config.cmake in: ${Python3_DIR}")
message(STATUS "Python3_EXECUTABLE: ${Python3_EXECUTABLE}")
message(STATUS "Python3_INCLUDE_DIRS: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python3_LIBRARIES: ${Python3_LIBRARIES}")
set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
add_subdirectory(thirdparty/pybind11)

# # 设置 Python 模块路径
# set(PYTHON_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/python)

include_directories(${Python3_INCLUDE_DIRS})


#-------------------------------------------------------------------------------
# FeGen configuration
#-------------------------------------------------------------------------------
# Add FeGen files to the include path
include_directories(${PROJECT_SOURCE_DIR}/runtime/include)

set(FeGen_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(FeGen_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(FeGen_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(FeGen_LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(FeGen_EXAMPLES_DIR ${FeGen_SOURCE_DIR}/examples)

set(EXECUTABLE_OUTPUT_PATH ${FeGen_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${FeGen_LIBRARY_DIR})

add_subdirectory(runtime)