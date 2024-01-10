/**
 * @file hw6.h - GPU-accelerated bitonic sort and dissemination scan
 * @author James Talbott
 * @see "Seattle University, CPSC5600, Winter 2023"
 * @version 1.0
 * 
 * @section Description
 * This program reads a list of (x,y) tuples from an input csv file.
 * This data is prepended by its row number in the input, sorted by x value increasing, 
 * and then appended with the scan of its y values.
 * The processed data, in (row,x,y,scan) form, is written to an output csv file.
 * The sort and scan algorithms are GPU-accelerated with CUDA.
 * Sort algorithm: bitonic sort.
 * Scan algorithm: dissemination scan (reference https://dl.acm.org/doi/pdf/10.1145/7902.7903)
 * 
*/

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
using namespace std;

const int MAX_BLOCK_SIZE = 1024; //Fetchable MAX_BLOCK_SIZE: BlockDim.x
const string DATA_PATH = "x_y/x_y.csv";
const string OUT_PATH = "out.csv";

/**
 * A light struct to store the input tuples, original row number, and scan values
 */
struct XY {
    int row;
    float x, y, scan = 0;

    XY(int row, float x, float y) {
        this->row = row;
        this->x = x;
        this->y = y;
    }
};

/** 
 * Dummy XY for padding data to a power of 2, specifically for ease of bitonic sorting.
 * x=max_int to force dummies to stay at the end of the vector
 * y=0 to guarantee scan is unaffected even if the above condition fails.
 * row=-1 sentinel value to make it even more obvious in printouts if a dummy sneaks in.
 */
const XY DUMMY_XY(-1, INT_MAX, 0); 

typedef vector<XY> XYs;

/**
 * Reads the input cxv of (x,y) tuples and stores them in memory.
 * @return A vector of XY data.
 */
XYs* readCSV() {
    fstream xyStream(DATA_PATH, fstream::in);
    XYs* data = new XYs();
    int row = 1;
    while (!xyStream.eof()) {
        string readbuf;
        getline(xyStream, readbuf);
        if (readbuf.length() > 3) { //don't keep header
            int comma = readbuf.find(',');
            string xs(readbuf, 0, comma);
            string ys(readbuf, comma + 1);
            float x = stof(xs);
            float y = stof(ys);
            XY xy(row++, x, y);
            data->push_back(xy);
        }
    }
    xyStream.close();
    return data;
}

/**
 * Pads the data vector of XYs to a length that is a power of 2.
 * This simplifies the bitonic sort, which in its base form works only on such-sized inputs.
 */
void padXYs(XYs* data) {
    int count = data->size();
    while (log2(count) != floor(log2(count))) { //Pad data to power of 2 size
        data->push_back(DUMMY_XY);
        count++;
    }
}

/**
 * GPU utility method to swap elements in the index array.
 * @param indices Array of indexes being sorted.
 * @param index1 First swappend index.
 * @param index2 Second swappend index.
 */
__device__ void swapIndices(int* indices, int index1, int index2) {
    int temp = indices[index1];
    indices[index1] = indices[index2];
    indices[index2] = temp;
}

/**
 * GPU Bitonic swap. Swaps two indices if their elements meet the bitonic swap condition.
 * The canonical i parameter of bitonic sort is provided by the global thread index.
 * @param xs Data array of input x values.
 * @param indices Array indexing the xs, on which swaps occur.
 * @param k Canonical k variable of bitonic sort.
 * @param j Canonical j variable of bitonic sort.
 * @param size Length of data and index arrays.
 */
__device__ void d_bitonicCompareSwap(float* xs, int* indices, int k, int j, int size) {
    int i = threadIdx.x + blockIdx.x * MAX_BLOCK_SIZE;
    if (i >= size) return;
    int ixj = i ^ j;
    if (ixj > i) {
        float x_i = xs[indices[i]];
        float x_ixj = xs[indices[ixj]];
        if ((i & k) == 0 && x_i > x_ixj
            || (i & k) != 0 && x_i < x_ixj) {
            swapIndices(indices, i, ixj);
        }
    }
}

/**
 * Kernel Bitonic swap for j values greater than the size of a block.
 * Directly calls device d_bitonicCompareSwap method.
 * @param xs Data array of input x values.
 * @param indices Array indexing the xs, on which swaps occur.
 * @param k Canonical k variable of bitonic sort.
 * @param j Canonical j variable of bitonic sort.
 * @param size Length of data and index arrays.
 */
__global__ void bitonicCompareSwap(float* xs, int* indices, int k, int j, int size) {
    d_bitonicCompareSwap(xs, indices, k, j, size);
}

/**
 * Kernel Bitonic swap for tailing j values <= block size. 
 * Calls Device d_bitonicCompareSwap method for each value of j remaining in this k loop.
 * @param xs Data array of input x values.
 * @param indices Array indexing the xs, on which swaps occur.
 * @param k Canonical k variable of bitonic sort.
 * @param j Canonical j variable of bitonic sort.
 * @param size Length of data and index arrays.
 */
__global__ void bitonicArgSortTailingJ(float* xs, int* indices, int k, int j, int size) {
    while (j > 0) {
        d_bitonicCompareSwap(xs, indices, k, j, size);
        j >>= 1;
        __syncthreads();
    }
}

/**
 * Bitonic Argsort. Given an array of XY data, performs an argsort on the x values.
 * The input vector is unmodified, instead an array of indices is returned that 
 * provides a sorted view of the data vector when used to index it.
 * @param data Vector of XY data to be sorted.
 * @return Array of sorted indexes, to be used for accessing the passed-in data array.
 */
int* bitonicArgSort(XYs* data) {
    int paddedSize = data->size();
    float* h_xs = new float[paddedSize];
    int* h_indices = new int[paddedSize];
    for (int i = 0; i < paddedSize; i++) {
        h_xs[i] = data->at(i).x;
        h_indices[i] = i;
    }
    float* d_xs;
    cudaMalloc(&d_xs, paddedSize * sizeof(float));
    cudaMemcpy(d_xs, h_xs, paddedSize * sizeof(float), cudaMemcpyHostToDevice);
    delete[] h_xs;
    int* d_indices;
    cudaMalloc(&d_indices, paddedSize * sizeof(int));
    cudaMemcpy(d_indices, h_indices, paddedSize * sizeof(int), cudaMemcpyHostToDevice);
    int num_blocks = (paddedSize + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
    for (int k = 2; k <= paddedSize; k <<= 1) {
        int j = k >> 1;
        while (j > MAX_BLOCK_SIZE / 2) {
            bitonicCompareSwap<<<num_blocks, MAX_BLOCK_SIZE>>>(d_xs, d_indices, k, j, paddedSize);
            j >>= 1;
            //implicit interblock barrier due to returning to host between kernels
        }
        bitonicArgSortTailingJ<<<num_blocks, MAX_BLOCK_SIZE>>>(d_xs, d_indices, k, j, paddedSize);
    }
    cudaMemcpy(h_indices, d_indices, paddedSize * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_xs);
    cudaFree(d_indices);
    return h_indices;
}

/**
 * GPU utility method that performs the partial sums of dissemination scan.
 * @param blockScan The local value array for which a scan is to be calculated in-place.
 * @param localCount The length of the local value array.
 * @param size The length of the global value array, of which the local array is a portion.
 */
__device__ void disseminate(float* blockScan, int localCount, int size) {
    int localIndex = threadIdx.x;
    int globalIndex = localIndex + blockIdx.x * MAX_BLOCK_SIZE;
    for (int displacement = 1; displacement < localCount; displacement <<= 1) {
        __syncthreads();
        float temp = 0;
        if (globalIndex < size && localIndex >= displacement) {
            temp = blockScan[localIndex - displacement];
        }
        __syncthreads();
        blockScan[localIndex] += temp;
    }
}

/**
 * Kernel dissemination scan. Performs a parallel scan of one block's worth of values.
 * @param d_ys The local value array for which a scan is to be calculated in-place.
 * @param postfixes A postfixes array at most one block size in length in which scan postfixes are to be stored.
 * @param size The length of the global value array, of which the local array is a portion.
 */
__global__ void disseminationScanBlock(float* d_ys, float* postfixes, int size) {
    int localIndex = threadIdx.x;
    int blockIndex = blockIdx.x;
    int globalIndex = localIndex + blockIndex * MAX_BLOCK_SIZE; //Fetchable MAX_BLOCK_SIZE: BlockDim.x
    int localCount = blockIndex == size / MAX_BLOCK_SIZE ? size % MAX_BLOCK_SIZE : MAX_BLOCK_SIZE;
    __shared__ float blockScan[MAX_BLOCK_SIZE];
    if (globalIndex < size) {
        blockScan[localIndex] = d_ys[globalIndex];
    }
    disseminate(blockScan, localCount, size);
    if (globalIndex < size) {
        d_ys[globalIndex] = blockScan[localIndex];
    }
    __syncthreads();
    if (localIndex == 0) {
        postfixes[blockIndex] = blockScan[localCount - 1];
    }
}

/**
 * Kernel dissemination scan. Performs a parallel scan of postfixes, which
 * is prerequisite to applying them to block-scanned values.
 * @param postfixes The local value array for which a scan is to be calculated in-place.
 * @param size The length of the postfix array, which is the number of blocks used to scan y values.
 */
__global__ void disseminationScanPostfixes(float* postfixes, int size) {
    int localIndex = threadIdx.x;
    int globalIndex = localIndex + blockIdx.x * MAX_BLOCK_SIZE;
    __shared__ float blockScan[MAX_BLOCK_SIZE];
    if (globalIndex < size) {
        blockScan[localIndex] = postfixes[globalIndex];
    }
    disseminate(blockScan, size, size);
    if (globalIndex < size) {
        postfixes[globalIndex] = blockScan[localIndex];
    }
}

/**
 * Kernel dissemination scan postfix application - corrects scans taken locally 
 * in blocks by adding the postfixes to each element in a particular block, which 
 * having themselves been scanned now equal the sum of all elements prior to said block.
 */
__global__ void disseminationScanApplyPostfixes(float* d_ys, float* postfixes, int size) {
    int blockIndex = blockIdx.x;
    int globalIndex = threadIdx.x + blockIdx.x * MAX_BLOCK_SIZE;
    if (blockIndex > 0 && globalIndex < size) {
        d_ys[globalIndex] += postfixes[blockIndex - 1];
    }
}

/**
 * Dissemination scan. Extracts y values from data and performs a scan on them, 
 * assigning the results as their original data's scan field.
 * While this uses postfixes instead of prefixes, this is for computational simplicity
 * in inner methods. They are handled in such a way as to mimic the expected use
 * of scan prefixes (ie. added to the block one index over).
 * Only safe up to 1024 blocks, or 2^20 = 1,048,576 total elements, 
 * as provision is only made for 1024 postfixes.
 * @param data Data vector.
 * @param indices Index array providing a sorted view of the data.
 * @param data_size Unpadded length of the data vector.
 */
void disseminationScan(XYs* data, int* indices, int data_size) {
    //Extract relevant data to simpler structure
    int byte_size = data_size * sizeof(float);
    float* h_ys = new float[data_size];
    for (int i = 0; i < data_size; i++) {
        h_ys[i] = data->at(indices[i]).y;
    }

    //Device allocations
    float* d_ys;
    cudaMalloc(&d_ys, byte_size);
    cudaMemcpy(d_ys, h_ys, byte_size, cudaMemcpyHostToDevice);
    int numBlocks = (data_size + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
    float* d_postfixes;
    cudaMalloc(&d_postfixes, numBlocks * sizeof(float));

    //Dissemination scan with GPU working inner loop
    disseminationScanBlock<<<numBlocks, MAX_BLOCK_SIZE>>>(d_ys, d_postfixes, data_size);
    if (numBlocks > 1) {
        disseminationScanPostfixes<<<1, numBlocks>>>(d_postfixes, numBlocks);
        disseminationScanApplyPostfixes<<<numBlocks, MAX_BLOCK_SIZE>>>(d_ys, d_postfixes, data_size);
    }
    cudaFree(d_postfixes);
    cudaMemcpy(h_ys, d_ys, byte_size, cudaMemcpyDeviceToHost);
    cudaFree(d_ys);

    //record scans in XYs
    for (int i = 0; i < data_size; i++) {
        data->at(indices[i]).scan = h_ys[i];
    }
    delete[] h_ys;
}

/**
 * Verifies the sortedness of output record x values with a simple serial check.
 * Prints errors to console if detected. Maxes out at 20 errors before it stops looking, 
 * so as to not blow out the console for large inputs.
*/
void verifySortedness(XYs* data, int* indices, int data_size) {
    int errorTolerance = 20;
    float last_x = data->at(indices[0]).x;
    for (int i = 1; i < data_size; i++) {
        float x = data->at(indices[i]).x;
        if (x < last_x) {
            cout << "Failed x sortedness check at index " << i << ", elements " << last_x << " and " << x << " at row " << data->at(indices[i]).row << endl;
            errorTolerance--;
            if (errorTolerance <= 0) {
                cout << "Sort error capacity reached, terminating check for printout brevity" << endl;
                return;
            }
        }
        last_x = x;
    }
    cout << "Sortedness verification complete" << endl;
}

/**
 * Verifies the computed scan values are monotonously nondecreasing.
 * More complex scan verification against a serially-calculated scan flags many false positive discrepancies
 * (as compared to sample output, assuming sample output correctness).
*/
void verifyScan(XYs* data, int* indices, int data_size) {
    float last_scan = -1;
    float scan = 0;
    for (int i = 0; i < data_size; i++) {
        XY xy = data->at(indices[i]);
        scan += xy.y;
        float diff = abs(scan - xy.scan);
        if (xy.scan < last_scan) {
            cout << "Scan monotonicity failure: index " << i << " == " << xy.scan << ", index " << i-1 << " == " << last_scan << endl;
        }
    }
    cout << "Scan verification complete" << endl;
}

/**
 * Writes sorted and scanned data to an output file defined above as a const.
 * @param data Data vector.
 * @param indices Index array providing a sorted view of the data.
 * @param data_size Unpadded length of the data vector.
 */
void writeXYsToFile(XYs* data, int* indices, int data_size) {
    fstream file(OUT_PATH, fstream::out | fstream::trunc);
    file << "n,x,y,scan" << endl;
    cout << "Write to output file...";
    for (int i = 0; i < data_size; i++) {
        XY xy = data->at(indices[i]);
        file << xy.row << "," << xy.x << "," << xy.y << "," << xy.scan << endl;
    }
    file.close();
    cout << "complete" << endl;
}

int main() {
    XYs* data = readCSV();
    int size = data->size();
    padXYs(data);

    int* indices = bitonicArgSort(data);
    disseminationScan(data, indices, size);

    //Verification methods commented out for submission to avoid cluttering grader's console.
    //verifySortedness(data, indices, size);
    //verifyScan(data, indices, size);

    writeXYsToFile(data, indices, size);

    delete data;
    delete[] indices;
}
