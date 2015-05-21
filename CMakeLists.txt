CMAKE_MINIMUM_REQUIRED(VERSION 2.8.3)
project(hierarchical_clustering)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
  include
  ${Eigen_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

if(NOT CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)

add_definitions("-msse3 -mfpmath=sse")

SET(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -fopenmp)

#add_library(hierarchical_clustering
#	src/Vocabulary.cpp
#)
#target_link_libraries(hierarchical_clustering ${OpenCV_LIBRARIES} ${Eigen_LIBRARIES})

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

add_executable(test_hierarchical_clustering
        sample/test_hierarchical_clustering.cpp
)
target_link_libraries(test_hierarchical_clustering)

#install(TARGETS vocabulary
#    RUNTIME DESTINATION bin
#    LIBRARY DESTINATION lib/${PROJECT_NAME}
#    ARCHIVE DESTINATION lib/${PROJECT_NAME})

#install(DIRECTORY include/${PROJECT_NAME}/
#DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#FILES_MATCHING PATTERN "*.h")
