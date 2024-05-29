cmake_minimum_required(VERSION 3.11)
project(seamat)
include(FetchContent)

if (CMAKE_VERSION VERSION_LESS "3.1")
  if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++17")
  endif ()
else ()
  set (CMAKE_CXX_STANDARD 17)
  set (CMAKE_CXX_STANDARD_REQUIRED ON)
endif ()

## Determine build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release) ## Build Release by default
elseif(CMAKE_BUILD_TYPE MATCHES Debug)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -Wextra -Wpedantic")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -g -Wall -Wextra -Wpedantic")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -g -Wall -Wextra -Wpedantic")
endif()

## Set compiler flags for Release
if(CMAKE_BUILD_TYPE MATCHES Release)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -xHost -fp-model fast=2 -no-prec-div -fimf-use-svml=true -ipo -m64")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -O3 -xHost -fp-model fast=2 -no-prec-div -fimf-use-svml=true -ipo -m64")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -O3 -xHost -fp-model fast=2 -no-prec-div -fimf-use-svml=true -ipo -m64")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -ffast-math -funroll-loops")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Ofast -ffast-math -funroll-loops")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Ofast -ffast-math -funroll-loops")
  endif()
endif()

## Check dependencies
find_package(OpenMP)
if (OPENMP_FOUND)
  set(SEAMAT_OPENMP_SUPPORT 1)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_PARALLEL")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -D_GLIBCXX_PARALLEL")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -D_GLIBCXX_PARALLEL")
else()
  set(SEAMAT_OPENMP_SUPPORT 0)
endif()
### Configure openmp support
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/seamat_openmp_config.hpp.in ${CMAKE_CURRENT_BINARY_DIR}/include/seamat_openmp_config.hpp @ONLY)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

## BitMagic - supplied with the project
## BitMagic
if (DEFINED CMAKE_BITMAGIC_HEADERS)
  message(STATUS "BitMagic headers provided in: ${CMAKE_BITMAGIC_HEADERS}")
else()
  set(CMAKE_BITMAGIC_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/external/BitMagic-7.12.3/src)
endif()
include_directories(${CMAKE_BITMAGIC_HEADERS})

## Download googletest if building tests
if(CMAKE_BUILD_TESTS)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
  if (DEFINED CMAKE_GOOGLETEST_HEADERS)
    message(STATUS "googletest headers provided in: ${CMAKE_GOOGLETEST_HEADERS}")
  else()
    FetchContent_Declare(googletest
      GIT_REPOSITORY    https://github.com/google/googletest.git
      GIT_TAG           release-1.11.0
      PREFIX            "external"
      SOURCE_DIR        "${CMAKE_CURRENT_SOURCE_DIR}/external/googletest"
      BINARY_DIR        "${CMAKE_CURRENT_BINARY_DIR}/external/googletest"
      BUILD_IN_SOURCE   0
      CMAKE_ARGS      -D CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
		      -D "CMAKE_C_FLAGS=${CMAKE_C_FLAGS}"
		      -D "CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}"
		      -D "CMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
		      -D "CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
    INSTALL_COMMAND   ""
    )
    FetchContent_MakeAvailable(googletest)
    set(CMAKE_GOOGLETEST_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/external/googletest")
   endif()
   include_directories(${CMAKE_GOOGLETEST_HEADERS})
   include_directories(${CMAKE_CURRENT_SOURCE_DIR}/test/include ${CMAKE_CURRENT_SOURCE_DIR}/include)

   enable_testing()
   add_executable(runTests
     ${CMAKE_CURRENT_SOURCE_DIR}/test/src/DenseMatrix_unittest.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/test/src/SparseMatrix_unittest.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/test/src/SparseIntegerTypeMatrix_unittest.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/test/src/IndexMatrix_unittest.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/test/src/Matrix_unittest.cpp)
   add_test(runTests runTests)

   ## Link tests
   target_link_libraries(runTests gtest gtest_main)
   if (OPENMP_FOUND)
     target_link_libraries(runTests OpenMP::OpenMP_CXX)
   endif()
endif()
