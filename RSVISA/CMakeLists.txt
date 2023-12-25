cmake_minimum_required(VERSION 3.20)
project(RSVISA)

set(CMAKE_CXX_STANDARD 23)

set(VISA_FOUND TRUE)

include_directories(/usr/include/rsvisa)

add_executable(RSVISA main.cpp)
