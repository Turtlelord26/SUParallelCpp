This is an implementation of the K-Means clustering algorithm parallelized using MPI.
More detailed information in the header comment of `KMeansMPI.h`.

`KMeansMPI.h` is my work, and includes implementation. All other files were provided for the assignment under requirement of non-modification. Those files' descriptions from the original assignment are as follows:
> - `Color.h/Color.cpp`  	A simple color class used by both the example program and your hw5 program.
- `KMeans.h`    The k-means clustering general abstract class. This is used by the example program and is the basis for your work in KMeansMPI.h.
- `ColorKMeans.h`   This is the concrete subclass template of KMeans used in the example program.
- `kmean_color_test.cpp`   This is the example program. It doesn't use MPI or any parallelization. It is a basis for hw5.cpp.
- `hw5.cpp`  	This is the example program designed to be run as part of an MPI-parallelized version of the example program. The KMeansMPI class you write must work with this program without modifications to the provided hw5.cpp, Color.cpp/h, and ColorKMeansMPI.h.
- `ColorKMeansMPI.h`    This is the subclass used by the provided hw5.cpp. It inherits from the abstract class you provide in your KMeansMPI.h submitted file.

This is the summary of the make commands given in the assignment:
> - `make`  # builds both the example program and your hw5 program
- `make run_sequential`  # builds and runs the example program
- `make hw5`  # builds your homework (won't work until you create KMeansMPI.h)
- `make run_hw5` #builds and runs your homework with mpirun -n 2
- `make bigger_test`  #builds and runs your homework with mpirun -n 10
- `make valgrind`  #builds and runs your homework using memchecker valgrind
- `make clean` #deletes all the non-source files
