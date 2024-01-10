/**
 * @file 
 * @author James Talbott
 * @version 1.0
 * @see "Seattle University, CPSC5600, Winter 2023"
 * 
 * @section Description
 * Fixed-threadcount implementation of Ladner-Fischer Parallel Prefix Sum
 * Homework 2 of Seattle University CPSC 5600-01 WQ2023
 * 
 * @section Extra Credit
 * I have implemented the extra credit requirement that the input data is allowed to not be an integer power of 2.
 * By carefully following N=6 and N=7 scenarios with pen and paper and subsequently tuning print statements for debugging,
 * I have determined that the algorithm's problem with non-2^x data size is that the data is worked on out of order. 
 * The algorithm seems to require that the leaves maintain their left-to-right ordering regardless of depth.
 * While this is solvable by packing the tree, that seems inelegant and inefficient.
 * Instead, I apply an offset to value calls on leaf nodes equal to the number of leaves on the shallower level, 
 * to the effect that the data will be treated as if it is in order left-to-right across the tree.
 * We also have to apply the offset to the indices in the output vector when writing results, to compensate.
 * For cleanliness' sake, the offset and modulo calculations have been given their own methods.
 */

#include <chrono>
#include <future>
#include <iostream>
#include <vector>
using namespace std;

using IntVector = vector<int>;

/**
 * @brief Heaper records and manages access to binary tree data using heap-style vector storage.
 * @class Heaper records and manages access to binary tree data using heap-style vector storage. 
 * The input data is used for leaf access and is assumed to be unmodified outside this class while this class uses it.
 * A second vector is initialized for interior nodes, but the separateness of the vectors is invisible to users.
 * This class provides basic tree navigation operations and is subclassable for more specific operations.
 * Created with reference to heap.h & heap.cpp, by Professor Lundeen, provided by assignment.
 */
class Heaper {
public:

    /**
     * @brief Constructor that requires an int vector input. The Heaper does not modify this vector and assumes no external modifications during use.
     * @param data int vector of data to use as leaves.
     */
    Heaper(IntVector* data) {
        leaves = data;
        interiorSize = leaves->size() - 1;
        interior = new IntVector(interiorSize, 0);
    }

    /**
     * @brief Simple destructor.
     */
    ~Heaper() {
        delete interior;
    }

    /**
     * @brief Formulaic copy constructor.
     */
    Heaper (const Heaper &other) {
        leaves = other.leaves;
        interiorSize = other.interiorSize;
        interior = new IntVector(interiorSize);
        for (int i = 0; i < interiorSize; i++) {
            interior->at(i) = other.interior->at(i);
        }
    }

    /**
     * @brief Formulaic assignment operator.
     */
    Heaper &operator=(const Heaper &rhs) {
        if (&rhs != this) {
            delete interior;
            leaves = rhs.leaves;
            interiorSize = rhs.interiorSize;
            interior = new IntVector(interiorSize);
            for (int i = 0; i < interiorSize; i++) {
                interior->at(i) = rhs.interior->at(i);
            }
        }
        return *this;
    }

protected:

    IntVector* interior;
    IntVector* leaves;
    int interiorSize;

    /**
     * @brief Get the parent node.
     * @param index The index of the child node.
     * @return The index of the parent node.
     */
    int parent(int index) {
        return (index - 1) / 2;
    }

    /**
     * @brief Get the right child.
     * @param index The index of the parent node.
     * @return The index of the right child node.
     */
    int right(int index) {
        return 2 * index + 2;
    }

    /** 
     * @brief Get the left child.
     * @param index The index of the parent node.
     * @return The index of the left child node.
     */
    int left(int index) {
        return 2 * index + 1;
    }

    /**
     * @brief Check if a node is a leaf.
     * @param index The index of a node.
     * @return True if the node is a leaf.
     */
    bool isLeaf(int index) {
        return index >= interiorSize;
    }

    /**
     * @brief Get the value stored in a node.
     * @param index The logical index of the node to access.
     * @return The value of the node at the argument index.
     */
    virtual int value(int index) {
        if (isLeaf(index)) {
            return leaves->at(index - interiorSize);
        } else {
            return interior->at(index);
        }
    }
};

/**
 * @brief A Heaper subclass that implements Ladner-Fischer Parallel Prefix Sum.
 * @class A Heaper subclass that implements Ladner-Fischer Parallel Prefix Sum.
 * Uses fixed parallelism with a maximum of 16 threads during each of the intemediate sum- and prefix sum phases.
*/
class SumHeap : public Heaper {
public:

    /**
     * @brief Constructor overload that performs initializations and intermediate sum calculations after object initialization.
     * @param data The int vector upon which to calculate prefix sums.
    */
    SumHeap(IntVector* data) : Heaper(data) {
        leavesSize = leaves->size();
        nodeOffset = calculateNodeOffset();
        calcSums();
    }

    /**
     * @brief Calculate the prefix sums of the input data, placing the results in the argument vector.
     * @param scan A vector in which to place the prefix sum results. It must match or exceed the input data vector's length.
     */
    void calcPrefixes(IntVector* scan) {
        calcPrefix(scan, 0, 0);
    }

private:

    int nodeOffset;
    int leavesSize;

    /**
     * @brief Calculates the offset at which results are placed into the output vector when calculating prefix sums.
     * Necessary to properly handle data that is not an integer power of two in size.
     * @return The offset with which to handle the leaves vector.
     */
    int calculateNodeOffset() {
        int totalSize = interiorSize + leavesSize;
        int nodesIfFullTree = 1;
        while (nodesIfFullTree < totalSize) {
            nodesIfFullTree *= 2;
        }
        nodesIfFullTree--; //Recalling that a full tree has 2^x - 1 nodes.
        int numLeavesAtShallowerDepth = (nodesIfFullTree - totalSize) / 2;
        return numLeavesAtShallowerDepth;
    }

    /**
     * @brief Calculate the intermediate sums of the input data, placing the results in the inherited interior member.
     */
    void calcSums() {
        calcSum(0);
    }

    /**
     * @brief Recursively calculates the intermediate sums of the data.
     * Creates up to 16 total threads (including main), forking based on tree depth.
     * @param i The index of the node currently being operated on.
    */
    void calcSum(int i) {
        if (isLeaf(i)) {
            return;
        }
        if (i < 16) {
            auto handle = async(launch::async, &SumHeap::calcSum, this, left(i));
            calcSum(right(i));
            handle.wait();
        }
        else {
            calcSum(left(i));
            calcSum(right(i));
        }
        interior->at(i) = value(left(i)) + value(right(i));
    }

    /**
     * @brief Recursively calculates the prefix sums of the data, using intermediate sums calculated previously.
     * Creates up to 16 total threads (including main), forking based on tree depth.
     * @param scan The output vector in which to place the prefix sums.
     * @param i The index of the node currently being operated on.
     * @param sumPrior The sum of all elements prior to the leftmost descendent of this node.
    */
    void calcPrefix(IntVector* scan, int i, int sumPrior) {
        if (isLeaf(i)) {
            recordResult(scan, i, sumPrior + value(i));
            return;
        }
        if (i < 16) {
            auto handle = async(launch::async, &SumHeap::calcPrefix, this, scan, left(i), sumPrior);
            calcPrefix(scan, right(i), sumPrior + value(left(i)));
            handle.wait();
        } else {
            calcPrefix(scan, left(i), sumPrior);
            calcPrefix(scan, right(i), sumPrior + value(left(i)));
        }
    }

    /**
     * @brief value override accesses leaves with an offset to preserve correctness if data is not an integer power of 2 in size.
     * @param index The logical index of the node to access.
     * @return The value of the node at the argument index.
     */
    int value(int index) override {
        if (isLeaf(index)) {
            return leaves->at(correctModulusOperation(index - interiorSize - nodeOffset, leavesSize));
        } else {
            return interior->at(index);
        }
    }

    /**
     * @brief Record the prefix sum of a leaf node in the correct index of the output scan vector.
     * @param scan The output vector for prefix sums.
     * @param i The index of the leaf node whose prefix sum is calculated.
     * @param prefixSum The value of the calculated prefix sum.
     */
    void recordResult(IntVector* scan, int i, int prefixSum) {
        int offsetIndex = correctModulusOperation(i - interiorSize - nodeOffset, leavesSize);
        scan->at(offsetIndex) = prefixSum;
    }

    /**
     * @brief Correctly computes index modulo modulus. 
     * The C++ % operator does not handle negative numbers correctly by itself, being a remainder operator rather than a true modulo operator.
     * @param index An index to be bounds-checked and possibly corrected.
     * @param modulus The value which the index must be smaller than.
     * @return The index modulo the modulus, which will be in the range [0, modulus).
     */
    int correctModulusOperation(int index, int modulus) {
        index %= modulus;
        if (index < 0)
            index += modulus;
        return index;
    }
};

const int N = 10000 * 10000;

/**
 * Copied nearly-verbatim with permission from assignment text.
 * @brief Runs the parallel prefix sum algorithm in a timer block.
 * Variable initialization and correctness check are excluded from timing.
 */
int main() {
    IntVector data(N, 1);  // put a 1 in each element of the data array
    data.at(0) = 10;
    IntVector prefix(N, 1);

    // start timer
    auto start = chrono::steady_clock::now();

    SumHeap heap(&data);
    heap.calcPrefixes(&prefix);

    // stop timer
    auto end = chrono::steady_clock::now();
    auto elapsed = chrono::duration<double,milli>(end-start).count();

    int check = 10;
    for (int elem: prefix) {
        if (elem != check++) {
            cout << "FAILED RESULT at " << check-1 << " ";
            break;
        }
    }
    cout << "in " << elapsed << "ms" << endl;
    return 0;
}