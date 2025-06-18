cmake_minimum_required(VERSION 3.10)
project(ExtractDataset)

set(CMAKE_CXX_STANDARD 17)

# === Aria SDK Headers ===
include_directories(
  ${CMAKE_SOURCE_DIR}/projectaria_tools/core
  ${CMAKE_SOURCE_DIR}/projectaria_tools/core/data_provider
)

# === Aria SDK Libs (optional) ===
link_directories(${CMAKE_SOURCE_DIR}/projectaria_tools/install/lib)

# === OpenCV ===
include_directories(/opt/homebrew/opt/opencv/include)
link_directories(/opt/homebrew/opt/opencv/lib)

# === Sophus ===
include_directories(/opt/homebrew/include)

# === nlohmann_json ===
include_directories(/opt/homebrew/opt/nlohmann-json/include)

# === Executable ===
add_executable(extract_from_vrs tools/extract_from_vrs.cpp)

# === Link Everything ===
target_link_libraries(extract_from_vrs
  PRIVATE
    opencv_core
    opencv_imgcodecs
    opencv_imgproc
    opencv_highgui
    projectaria_tools_core
)
