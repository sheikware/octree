# find gtest ...
FIND_PACKAGE(GTest REQUIRED)

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
      pthread
      )
	ADD_TEST(octree-test octree-test)
ENDIF()
