cmake_minimum_required(VERSION 3.16)
project(RocketDerby VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(SDL2 REQUIRED)

add_executable(rocket main.cpp)

target_include_directories(rocket PRIVATE ${SDL2_INCLUDE_DIRS})
target_link_libraries(rocket ${SDL2_LIBRARIES})

if(WIN32)
  target_link_libraries(rocket opengl32 glu32)
else()
  find_package(OpenGL REQUIRED)
  target_link_libraries(rocket ${OPENGL_LIBRARIES} ${OPENGL_GLU_LIBRARIES})
endif()
