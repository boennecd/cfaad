## CFAAD

This code is a re-written version of the
algorithmic adjoint differentiation implementation at
[github.com/asavine/CompFinance](https://github.com/asavine/CompFinance). A few tests are provided in the
[tests](tests/) directory along with some benchmarks using 
the [Catch2](https://github.com/catchorg/Catch2) library.

The code has been extended with a few vector, matrix, and vector-matrix
functions which I need in my research. Some have required that the library is linked
with BLAS and LAPACK.

I have not spent much time on CMake but a very simple `CMakeLists.txt` file is
provided that works locally for me on Ubuntu with g++ 9.3.0. The file can be used
to run the tests and produce the benchmarks shown in [test.log](test.log). 
All the pure double versions in the benchmarks do not produce the gradient.
The tests are run and the benchmark are performed by running:

```bash
cmake -S . -B build/
cmake --build build --verbose -j4
./build/cfaad-test > test.log
```

### License
The original library is written by Antoine Savine and
is distributed under the following conditions

> The code is freely available to anyone. Any person who purchased a copy of the book is authorized to use, modify and distribute the code for any application, as long as the credits remain on the top of the files.

The book is

> Modern Computational Finance: AAD and Parallel Simulations
Antoine Savine
Wiley, 2018
