cmake -S . -B build/
cmake --build build --verbose -j4
./build/cfaad-test
