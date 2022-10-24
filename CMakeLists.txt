﻿# CMakeList.txt : CMake project for trustid_image_processing_win, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("trustid_image_processing_win" VERSION 0.1.0)
set(CMAKE_CXX_STANDARD 17)

# import needed packages
find_package( OpenCV REQUIRED)
add_subdirectory(depends/dlib)

# list source files to compile
file(GLOB_RECURSE source_list "src/*.cc" "includes/*.h" "includes/*.cc")

# Resource files we need to add to the library
set(RESOURCE_FILES
  resources/ERT68.dat
  resources/dlib_face_recognition_resnet_model_v1.dat)

add_library(trustid-image-processing-lib STATIC ${source_list} ${RESOURCE_FILES})
target_include_directories(trustid-image-processing-lib PUBLIC "includes" ${OpenCV_INCLUDE_DIRS})
target_link_libraries(trustid-image-processing-lib PUBLIC dlib ${OpenCV_LIBS})

add_executable(trustid-image-processing-ex-buildverifier "examples/build_simple_face_verifier.cc")
add_executable(trustid-image-processing-ex-verifyface "examples/detect_and_verify_faces.cc")

target_include_directories(trustid-image-processing-ex-verifyface PUBLIC "includes" ${OpenCV_INCLUDE_DIRS})
target_link_libraries(trustid-image-processing-ex-verifyface PUBLIC dlib trustid-image-processing-lib ${OpenCV_LIBS})

target_include_directories(trustid-image-processing-ex-buildverifier PUBLIC "includes" ${OpenCV_INCLUDE_DIRS})
target_link_libraries(trustid-image-processing-ex-buildverifier PUBLIC dlib trustid-image-processing-lib ${OpenCV_LIBS})

add_executable(trustid-image-processing-ex-serialization-test "examples/serialization_test.cc")
target_include_directories(trustid-image-processing-ex-serialization-test PUBLIC "includes" ${OpenCV_INCLUDE_DIRS})
target_link_libraries(trustid-image-processing-ex-serialization-test PUBLIC dlib trustid-image-processing-lib ${OpenCV_LIBS})

add_executable(trustid-image-processing-ex-webcam-builder "examples/build_simple_face_verifier_webcam.cc")
target_include_directories(trustid-image-processing-ex-webcam-builder PUBLIC "includes" ${OpenCV_INCLUDE_DIRS})
target_link_libraries(trustid-image-processing-ex-webcam-builder PUBLIC dlib trustid-image-processing-lib ${OpenCV_LIBS})

IF (WIN32)
  # set stuff for windows

get_target_property(__dll_dbg opencv_world IMPORTED_LOCATION_DEBUG)
get_target_property(__dll_release opencv_world  IMPORTED_LOCATION_RELEASE)

add_custom_command(TARGET trustid-image-processing-ex-buildverifier POST_BUILD
COMMAND ${CMAKE_COMMAND} -E copy_directory
${PROJECT_SOURCE_DIR}/resources $<TARGET_FILE_DIR:trustid-image-processing-ex-buildverifier>/resources
)
add_custom_command(TARGET  trustid-image-processing-ex-buildverifier POST_BUILD
COMMAND ${CMAKE_COMMAND} -E copy_if_different           # which executes "cmake - E copy_if_different..."
"$<$<CONFIG:debug>:${__dll_dbg}>$<$<CONFIG:release>:${__dll_release}>"      # <--this is in-file
$<TARGET_FILE_DIR:trustid-image-processing-ex-buildverifier>                        # <--this is out-file path
)

add_custom_command(TARGET trustid-image-processing-ex-verifyface POST_BUILD
COMMAND ${CMAKE_COMMAND} -E copy_directory
${PROJECT_SOURCE_DIR}/resources $<TARGET_FILE_DIR:trustid-image-processing-ex-verifyface>/resources
)
add_custom_command(TARGET  trustid-image-processing-ex-verifyface POST_BUILD
COMMAND ${CMAKE_COMMAND} -E copy_if_different           # which executes "cmake - E copy_if_different..."
"$<$<CONFIG:debug>:${__dll_dbg}>$<$<CONFIG:release>:${__dll_release}>"      # <--this is in-file
$<TARGET_FILE_DIR:trustid-image-processing-ex-verifyface>                        # <--this is out-file path
)

add_custom_command(TARGET trustid-image-processing-ex-serialization-test POST_BUILD
COMMAND ${CMAKE_COMMAND} -E copy_directory
${PROJECT_SOURCE_DIR}/resources $<TARGET_FILE_DIR:trustid-image-processing-ex-serialization-test>/resources
)
add_custom_command(TARGET trustid-image-processing-ex-serialization-test POST_BUILD
COMMAND ${CMAKE_COMMAND} -E copy_if_different           # which executes "cmake - E copy_if_different..."
"$<$<CONFIG:debug>:${__dll_dbg}>$<$<CONFIG:release>:${__dll_release}>"      # <--this is in-file
$<TARGET_FILE_DIR:trustid-image-processing-ex-serialization-test>                        # <--this is out-file path
)

add_custom_command(TARGET trustid-image-processing-ex-webcam-builder POST_BUILD
COMMAND ${CMAKE_COMMAND} -E copy_directory
${PROJECT_SOURCE_DIR}/resources $<TARGET_FILE_DIR:trustid-image-processing-ex-webcam-builder>/resources
)
add_custom_command(TARGET trustid-image-processing-ex-webcam-builder POST_BUILD
COMMAND ${CMAKE_COMMAND} -E copy_if_different           # which executes "cmake - E copy_if_different..."
"$<$<CONFIG:debug>:${__dll_dbg}>$<$<CONFIG:release>:${__dll_release}>"      # <--this is in-file
$<TARGET_FILE_DIR:trustid-image-processing-ex-webcam-builder>                        # <--this is out-file path
)
ENDIF()

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
