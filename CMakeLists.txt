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

add_library(hierarchical_clustering
	src/Vocabulary.cpp
)
target_link_libraries(hierarchical_clustering ${OpenCV_LIBRARIES} ${Eigen_LIBRARIES})


#install(TARGETS vocabulary
#    RUNTIME DESTINATION bin
#    LIBRARY DESTINATION lib/${PROJECT_NAME}
#    ARCHIVE DESTINATION lib/${PROJECT_NAME})

#install(DIRECTORY include/${PROJECT_NAME}/
#DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#FILES_MATCHING PATTERN "*.h")
