project(octree)
cmake_minimum_required(VERSION 3.10)

if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

# Release by default
# Turn on Debug with "-DCMAKE_BUILD_TYPE=Debug"
if( NOT CMAKE_BUILD_TYPE )
   SET( CMAKE_BUILD_TYPE Release )
endif()

# Set relative install directories
set(INSTALL_INCLUDE_DIR "include")
set(INSTALL_CMAKE_DIR "lib${LIB_SUFFIX}/cmake/octree")

# Define octree lib (header-only)
add_library(octree INTERFACE)
target_include_directories(octree
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${INSTALL_INCLUDE_DIR}>)

install(TARGETS octree
        EXPORT octreeTargets)

add_library(octree::octree ALIAS octree)
 
 
 # Generate the cmake config and cmake config-version file:
include(CMakePackageConfigHelpers)

configure_package_config_file(
    "${octree_SOURCE_DIR}/octreeConfig.cmake.in"
    "${octree_BINARY_DIR}/octreeConfig.cmake"
    INSTALL_DESTINATION ${INSTALL_CMAKE_DIR}
    PATH_VARS INSTALL_INCLUDE_DIR)
    
find_package(Boost)

if(Boost_FOUND)
  ADD_EXECUTABLE(example1 examples/example1.cpp)
  target_link_libraries(example1
    octree::octree
    Boost::boost
    )
    
  ADD_EXECUTABLE(example2 examples/example2.cpp)
  target_link_libraries(example2
    octree::octree
    Boost::boost
    )
    
  ADD_EXECUTABLE(example3 examples/example3.cpp)
  target_link_libraries(example3
    octree::octree
    Boost::boost
    )
endif()

# find gtest ...
IF(IS_DIRECTORY "/usr/src/gtest/")
  MESSAGE("Found google test sources in /usr/src/gtest/")
  
  ADD_SUBDIRECTORY(/usr/src/gtest/ gtest)
  
  # mimick the behaviour of find_package(GTest)
  SET(GTEST_FOUND TRUE)
  SET(GTEST_BOTH_LIBRARIES gtest gtest_main)
  SET(GTEST_LIBRARIES gtest)
  SET(GTEST_MAIN_LIBRARIES gtest_main)
ELSE()
  FIND_PACKAGE(GTest)
ENDIF()

# build gtests if possible
IF(GTEST_FOUND)
  MESSAGE("Building tests with google test framework")
	ENABLE_TESTING()
	INCLUDE_DIRECTORIES(${GTEST_INCLUDE_DIRS})
	ADD_EXECUTABLE(octree-test test/octree-test.cpp)
	TARGET_LINK_LIBRARIES(octree-test 
      ${GTEST_LIBRARIES}
      ${GTEST_MAIN_LIBRARIES}
      octree
      Boost::boost
      )
	ADD_TEST(octree-test octree-test)
ENDIF()


install(EXPORT octreeTargets
        NAMESPACE octree::
        DESTINATION "${INSTALL_CMAKE_DIR}")

install(
    FILES "${octree_BINARY_DIR}/octreeConfig.cmake"
    DESTINATION "${INSTALL_CMAKE_DIR}" )
    
install(
	FILES "${octree_SOURCE_DIR}/include/Octree.hpp"
	DESTINATION "${INSTALL_INCLUDE_DIR}" )
        