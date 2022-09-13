# CMakeList.txt : CMake project for trustid_image_processing_win, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("trustid_image_processing_win" VERSION 0.1.0)
set(CMAKE_CXX_STANDARD 17)

# TODO: Add tests and install targets if needed.
set(OpenCV_DIR depends/opencv/build)
#set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
#set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
find_package( OpenCV REQUIRED )

add_subdirectory(depends/dlib)

file(GLOB base_src CONFIGURE_DEPENDS "src/trustid_image_processing/*.cpp")
file(GLOB dlib_src CONFIGURE_DEPENDS "src/trustid_image_processing/dlib_impl/*.cpp")
file(GLOB ocv_src CONFIGURE_DEPENDS "src/trustid_image_processing/opencv_impl/*.cpp")
file(GLOB client_src CONFIGURE_DEPENDS "src/trustid_image_processing/client/*.cpp")
file(GLOB server_src CONFIGURE_DEPENDS "src/trustid_image_processing/server/*.cpp")

file(GLOB_RECURSE source_list "src/*.cc" "includes/*.h")

# Resource files we need to add to the library
set(RESOURCE_FILES
  resources/ERT68.dat
  resources/dlib_face_recognition_resnet_model_v1.dat)

 #add_library(trustid-image-processing-lib STATIC ${base_src} ${dlib_src} ${PROTO_SRCS} ${PROTO_HDRS} ${RESOURCE_FILES})
add_library(trustid-image-processing-lib STATIC ${source_list} ${RESOURCE_FILES})
target_include_directories(trustid-image-processing-lib PUBLIC "includes" ${OpenCV_INCLUDE_DIRS})
#target_link_libraries(trustid-image-processing-lib PUBLIC dlib ${OpenCV_LIBS} ${Protobuf_LIBRARIES})
target_link_libraries(trustid-image-processing-lib PUBLIC dlib ${OpenCV_LIBS})

add_executable(trustid-image-processing-ex-buildverifier "examples/build_simple_face_verifier.cc")
add_executable(trustid-image-processing-ex-verifyface "examples/detect_and_verify_faces.cc")

target_include_directories(trustid-image-processing-ex-verifyface PUBLIC "includes" ${OpenCV_INCLUDE_DIRS})
target_link_libraries(trustid-image-processing-ex-verifyface PUBLIC dlib trustid-image-processing-lib ${OpenCV_LIBS})
target_include_directories(trustid-image-processing-ex-buildverifier PUBLIC "includes" ${OpenCV_INCLUDE_DIRS})
target_link_libraries(trustid-image-processing-ex-buildverifier PUBLIC dlib trustid-image-processing-lib ${OpenCV_LIBS})


## Only needed for MacOSX
#set_target_properties(ExecutableTarget PROPERTIES
#  MACOSX_BUNDLE TRUE
#  MACOSX_FRAMEWORK_IDENTIFIER org.cmake.ExecutableTarget
#  RESOURCE "${RESOURCE_FILES}"))


#set(CPACK_PROJECT_NAME ${PROJECT_NAME})
#set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
#include(CPack)
