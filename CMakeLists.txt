cmake_minimum_required(VERSION 3.10)
project(network)

# Specify the C standard
set(CMAKE_C_STANDARD 99)

# Add your source files
add_executable(exec lib.c mongoose.c)

# Link the math library (equivalent of -lm)
target_link_libraries(exec m)

