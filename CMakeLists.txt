cmake_minimum_required(VERSION 3.14)
project(bird_cbir_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message(FATAL_ERROR
        "Detected GNU g++ on macOS. This project depends on Homebrew OpenCV, which is typically built with AppleClang/libc++. "
        "Using g++ causes C++ ABI mismatch (e.g. undefined cv::imread/cv::error with std::__cxx11). "
        "Please configure with clang++ instead.")
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(SQLite3 REQUIRED)

add_executable(cbir_app
    src/main.cpp
    src/image_preprocess.cpp
    src/feature_extractor.cpp
    src/sqlite_repo.cpp
    src/indexer.cpp
    src/searcher.cpp
)

target_include_directories(cbir_app PRIVATE
    include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(cbir_app PRIVATE
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
    SQLite::SQLite3
)

