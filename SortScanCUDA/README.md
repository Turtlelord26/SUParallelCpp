The goal of this assignment was to both sort and scan (cumulative sum) a large list of values in parallel using GPU acceleration.

The data are (x, y) tuples, which are to be sorted by x value and then scanned by y value. The output format is `original row number, x, y, cumulative sum of y`.

The sorting algorithm is Bitonic Sort

The scan algorithm is Parallel Dissemination Prefix Scan, based on this (paper)[https://dl.acm.org/doi/pdf/10.1145/7902.7903].

`build.bat` builds the program.
`run.bat` executes it.
`profile.bat` runs the nvprof activity profiler. Sample output on a university machine in `csp239_profile.txt`.

the `cuda_test.cu` file can be run to confirm the functionality of the GPU.
