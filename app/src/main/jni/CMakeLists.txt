project(yoloncnn)

cmake_minimum_required(VERSION 3.10)

set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv-mobile-4.9.0-android/sdk/native/jni)
find_package(OpenCV REQUIRED core imgproc)

set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20240102-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

add_library(yoloncnn SHARED yoloncnn.cpp yolo.cpp ndkcamera.cpp hepler.cpp)

target_link_libraries(yoloncnn ncnn ${OpenCV_LIBS} camera2ndk mediandk)
