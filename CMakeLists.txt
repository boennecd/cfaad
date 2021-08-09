cmake_minimum_required (VERSION 2.8) 
project (cfaad) 

set(CMAKE_CXX_FLAGS "-std=gnu++17 -Wall -Wclobbered -Wdeprecated -Wignored-qualifiers -Wmissing-field-initializers -Wsign-compare -Wtype-limits -Wuninitialized -Wshift-negative-value -Wunused-but-set-parameter -Wlto-type-mismatch -Wno-ignored-attributes -Wno-deprecated-declarations -Wno-parentheses -Wodr -Wconversion -Wsign-conversion")

include_directories("include")
file(GLOB lib_headers "include/*.h")
file(GLOB test_files "tests/*.cpp")

find_package(Catch2 REQUIRED)

add_executable(cfaad-test ${test_files} ${lib_headers})
target_link_libraries(cfaad-test Catch2::Catch2WithMain)

include(CTest)
include(Catch)

