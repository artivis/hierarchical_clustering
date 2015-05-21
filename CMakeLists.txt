CMAKE_MINIMUM_REQUIRED(VERSION 2.8.3)
project(hierarchical_clustering)

#find_package(Eigen REQUIRED)
#find_package(OpenCV REQUIRED)

include_directories(
  include
#  ${Eigen_INCLUDE_DIRS}
#  ${OpenCV_INCLUDE_DIRS}
)

if(NOT CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)

add_definitions("-msse3 -mfpmath=sse")

SET(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -fopenmp)

add_executable(test_accumulator
        sample/test_accumulator.cpp
)
target_link_libraries(test_accumulator)

add_executable(test_distance
        sample/test_distance.cpp
)
target_link_libraries(test_distance)

add_executable(test_initializer
        sample/test_initializer.cpp
)
target_link_libraries(test_initializer)

add_executable(test_kmeans
        sample/test_kmeans.cpp
)
target_link_libraries(test_kmeans)

add_executable(test_kmedoids
        sample/test_kmedoids.cpp
)
target_link_libraries(test_kmedoids)

add_executable(test_hierarchical_clustering
        sample/test_hierarchical_clustering.cpp
)
target_link_libraries(test_hierarchical_clustering)

set(INSTALL_INCLUDE_DIR "/usr/include")

install(
  DIRECTORY include/
  DESTINATION ${INSTALL_INCLUDE_DIR}
  FILES_MATCHING PATTERN "*.h"
)

message(${CMAKE_CURRENT_BINARY_DIR})

add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/script/cmake_uninstall.cmake
)
