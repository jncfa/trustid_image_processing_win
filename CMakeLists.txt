# CMakeList.txt : CMake project for trustid_image_processing_win, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("trustid_image_processing_win" VERSION 0.1.0)
set(CMAKE_CXX_STANDARD 17)

# TODO: Add tests and install targets if needed.
set(OpenCV_DIR depends/opencv/build)

# import needed packages
find_package( OpenCV REQUIRED )
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



add_executable(trustid-image-processing-ex-webcam-builder "examples/build_simple_face_verifier_webcam.cc")
target_include_directories(trustid-image-processing-ex-webcam-builder PUBLIC "includes" ${OpenCV_INCLUDE_DIRS})
target_link_libraries(trustid-image-processing-ex-webcam-builder PUBLIC dlib trustid-image-processing-lib ${OpenCV_LIBS})

add_custom_command(TARGET trustid-image-processing-ex-webcam-builder POST_BUILD
COMMAND ${CMAKE_COMMAND} -E copy_directory
${PROJECT_SOURCE_DIR}/resources $<TARGET_FILE_DIR:trustid-image-processing-ex-webcam-builder>/resources
)
add_custom_command(TARGET trustid-image-processing-ex-webcam-builder POST_BUILD
COMMAND ${CMAKE_COMMAND} -E copy_if_different           # which executes "cmake - E copy_if_different..."
"$<$<CONFIG:debug>:${__dll_dbg}>$<$<CONFIG:release>:${__dll_release}>"      # <--this is in-file
$<TARGET_FILE_DIR:trustid-image-processing-ex-webcam-builder>                        # <--this is out-file path
)

add_executable(trustid-image-processing-ex-serialization-test "examples/serialization_test.cc")
target_include_directories(trustid-image-processing-ex-serialization-test PUBLIC "includes" ${OpenCV_INCLUDE_DIRS})
target_link_libraries(trustid-image-processing-ex-serialization-test PUBLIC dlib trustid-image-processing-lib ${OpenCV_LIBS})

add_custom_command(TARGET trustid-image-processing-ex-serialization-test POST_BUILD
COMMAND ${CMAKE_COMMAND} -E copy_directory
${PROJECT_SOURCE_DIR}/resources $<TARGET_FILE_DIR:trustid-image-processing-ex-serialization-test>/resources
)
add_custom_command(TARGET trustid-image-processing-ex-serialization-test POST_BUILD
COMMAND ${CMAKE_COMMAND} -E copy_if_different           # which executes "cmake - E copy_if_different..."
"$<$<CONFIG:debug>:${__dll_dbg}>$<$<CONFIG:release>:${__dll_release}>"      # <--this is in-file
$<TARGET_FILE_DIR:trustid-image-processing-ex-serialization-test>                        # <--this is out-file path
)

#set(CPACK_PROJECT_NAME ${PROJECT_NAME})
#set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
#include(CPack)

#find_package(Protobuf CONFIG REQUIRED )
#find_package(absl CONFIG REQUIRED)
#find_package(gRPC CONFIG REQUIRED)
#
#set(PROTO_FILES
#    examples/grpc/helloworld.proto
#)
#
##get_filename_component(hw_proto "examples/grpc/helloworld.proto" ABSOLUTE)
##get_filename_component(hw_proto_path "${hw_proto}" PATH)
#
## Generated sources
##set(hw_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/helloworld.pb.cc")
##set(hw_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/helloworld.pb.h")
##set(hw_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/helloworld.grpc.pb.cc")
##set(hw_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/helloworld.grpc.pb.h")
##include_directories("${CMAKE_CURRENT_BINARY_DIR}")
#
#add_executable(trustid-image-processing-grpc-simpleserver "examples/grpc/grpc-simpleserver.cc" "examples/grpc/helloworld.grpc.pb.cc" "examples/grpc/helloworld.pb.cc")
#target_include_directories(trustid-image-processing-grpc-simpleserver PUBLIC "includes" ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR})
#
#get_target_property(grpc_cpp_plugin_location gRPC::grpc_cpp_plugin LOCATION)
##protobuf_generate(LANGUAGE cpp TARGET trustid-image-processing-grpc-simpleserver)
##protobuf_generate( 
##    TARGET
##      trustid-image-processing-grpc-simpleserver 
##    LANGUAGE
##        grpc 
##    PLUGIN
##        "protoc-gen-grpc=${grpc_cpp_plugin_location}"
##)
#
#add_executable(trustid-image-processing-grpc-simpleclient "examples/grpc/grpc-simpleclient.cc" "examples/grpc/helloworld.grpc.pb.cc" "examples/grpc/helloworld.pb.cc")
#target_include_directories(trustid-image-processing-grpc-simpleserver PUBLIC "includes" ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR})
#
#
#target_link_libraries(trustid-image-processing-grpc-simpleserver )
#target_link_libraries( trustid-image-processing-grpc-simpleserver PUBLIC dlib trustid-image-processing-lib ${OpenCV_LIBS} gRPC::grpc++ gRPC::grpc++_reflection gRPC::gpr gRPC::grpc gRPC::grpc++ protobuf::libprotoc protobuf::libprotobuf )
#
#add_custom_command(TARGET trustid-image-processing-grpc-simpleserver POST_BUILD
#COMMAND ${CMAKE_COMMAND} -E copy_directory
#${CMAKE_SOURCE_DIR}/resources $<TARGET_FILE_DIR:trustid-image-processing-grpc-simpleserver>/resources
#)
#add_custom_command(TARGET trustid-image-processing-grpc-simpleserver POST_BUILD
#COMMAND ${CMAKE_COMMAND} -E copy_if_different           # which executes "cmake - E copy_if_different..."
#"$<$<CONFIG:debug>:${__dll_dbg}>$<$<CONFIG:release>:${__dll_release}>"      # <--this is in-file
#$<TARGET_FILE_DIR:trustid-image-processing-grpc-simpleserver>                        # <--this is out-file path
#)