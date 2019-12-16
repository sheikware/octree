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
	ADD_EXECUTABLE(octree-test octree-test.cpp)
	TARGET_LINK_LIBRARIES(octree-test 
      ${GTEST_LIBRARIES}
      ${GTEST_MAIN_LIBRARIES}
      octree
      Boost::boost
      )
	ADD_TEST(octree-test octree-test)
ENDIF()