## cfaad

This code is a re-written version of the
algorithmic adjoint differentiation implementation at
https://github.com/asavine/CompFinance. A few tests are provided in the
[test](tests/) directory along with some benchmarks using
[Catch2](https://github.com/catchorg/Catch2) library.

The code has been extended with a few vector, matrix, and vector-matrix
functions which I need in my research. Some have required that the library is linked
with BLAS and LAPACK.

I have not spent much time on CMake but a very simple `CMakeLists.txt` file is
provided that works locally for me on Ubuntu with g++ 9.3.0. The file can be used
to run the tests and produce the benchmarks shown in [test.log](test.log). This
is done by calling:

```bash
cmake -S . -B build/
cmake --build build --verbose -j4
./build/cfaad-test > test.log
```

### License
The author of the original library (Antoine Savine)
has allowed this software to distributed under the MIT license as long as the
original comments in the beginning of the files remain.