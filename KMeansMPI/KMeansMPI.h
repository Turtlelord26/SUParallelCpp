/**
 * @file KMeansMPI.h - implementation of parallelized k-means clustering algorithm using MPI
 * @author James Talbott
 * @see "Seattle University, CPSC5600, Winter 2023"
 * @version 1.0
 * 
 * @section Description
 * Implements a parallel Kmeans clustering algorithm using OpenMPI message passing.
 * The root node calls fit, which sets up root-specific values.
 * Other nodes proceed directly to fitwork, which performs the parallel KMeans process:
 * 1. update the distance from local elements to each centroid
 * 2. update local cluster assignments based on new distances
 * 3. collect updated local cluster centroids to the root process, which weighted-averages them.
 * 4. broadcast averaged centroids to all processes
 * 5. if the centroids are now different than at step 1, return to step 1
 * 6. otherwise, gather the final membership assignments from each process to the root.
 * 
 * Implementation notes:
 *     I gather cluster centroids as ints, and divide them back to u_chars at the last possible moment
 * in order to avoid non-associative floating point error accumulation. Thus my results are completely identical
 * regardless of the number of processes running, down to the exact centroid coordinates.
 *     I also never transmit any individual Elements after they are first scattered. Because scatterv and gatherv 
 * guarantee ordering, I can match indices of the gatherv recvbuffer to indices in the original elements array and 
 * thus only need to transmit an int indicating a cluster in the post-convergence gather. As for combining centroids
 * during iterations, I only transmit the sums of the component elements' attributes and a size value.
 * 
 * @section Attribution
 * Modified from serial KMeans.h file included in assignment, authored by Prof. Lundeen.
 */

#pragma once  // only process the first time it is included; ignore otherwise
#include "mpi.h"
#include <vector>
#include <random>
#include <algorithm>
#include <iostream>
#include <set>
#include <array>

/**
 * @brief MPI-based parallel implementation of the KMeans clustering algorithm.
 * 
 * @param k The number of clusters to fit the data to.
 * @param d The number of attributes per datum.
*/
template <int k, int d>
class KMeansMPI {
public:
    // some type definitions to make things easier
    typedef std::array<u_char,d> Element;
    class Cluster;
    typedef std::array<Cluster,k> Clusters;
    const int MAX_FIT_STEPS = 300; //TODO FIXME restore line from test state.
    const int ROOT = 0;

    // debugging
    const bool VERBOSE = false;  // set to true for debugging output
    const bool VERY_VERBOSE = false;
    const bool VERBOSE_BARRIERS = false;
#define V(stuff) if (VERBOSE) {using namespace std; stuff}
#define VV(stuff) if (VERY_VERBOSE) {using namespace std; stuff}
#define VB(stuff) if (VERBOSE_BARRIERS) {using namespace std; stuff}

    /**
     * Expose the clusters to the client readonly.
     * @return clusters from latest call to fit()
     */
    virtual const Clusters& getClusters() {
        return clusters;
    }

    /**
     * Called by the root process only. Sets up root-specific data preparatory to sharing with other processes, then calls fitwork automatically.
     * 
     * @param data An array of Elements, aka length-d unsigned char arrays.
     * @param data_n the number of data.
     */
    virtual void fit(Element* data, int data_n) {
        MPI_Comm_size(MPI_COMM_WORLD, &world);
        this->data_n = data_n;
        V(cout<<("World size detected at " + to_string(world) + ".\nThere are " + to_string(data_n) + " total data and " + to_string(data_n * d) + " total elements.\n"););
        sendCounts = new int[world];
        sendOffsets = new int[world];
        for (int rank = 0; rank < world; rank++) {
            int offset = d * (data_n * rank / world);
            sendOffsets[rank] = offset;
            sendCounts[rank] = d * (data_n * (rank + 1) / world) - offset; //the double-calculation is preferable to an additional O(world) memory.
            elements = data;
        }
        VV(string s = ""; for(int i = 0; i < world; i++) {s += ("[" + to_string(sendOffsets[i]) + "] ");}cout<<("Root calculates sendOffsets at: " + s + "\n"););
        VV(string s = ""; for(int i = 0; i < world; i++) {s += ("[" + to_string(sendCounts[i]) + "] ");}cout<<("Root calculates sendCounts at: " + s + "\n"););
        fitWork(0);
    }

    /**
     * Performs KMeans clustering.
     * Does some MPI administation, like sharing relevant variables from the root process and scattering the elements among the processes.
     * Then performs KMeans iterations until convergence or iteration limit, and gathers final cluster assignments to the root.
     * 
     * @param rank The MPI rank of the calling process.
     */
    virtual void fitWork(int rank) {
        MPI_Comm_size(MPI_COMM_WORLD, &world);
        
        //bcast data_n so processes can calculate their elemCounts
        //This repeats some displs calculations from fit(), but is better than adding communication overhead.
        broadcastN(rank);
        elemCount = (data_n * (rank + 1) / world) - (data_n * rank / world);
        VV(cout<<("Process " + to_string(rank) + " calculated " + to_string(elemCount) + " elemCount from data_n/world Bcast.\n"););

        scatterElements(rank);

        //initialize kmeans variables
        dist.resize(elemCount);
        if (rank == ROOT) {
            reseedClusters();
            V(cout << ("Process " + to_string(rank) + " reseeded initial clusters: " + printClusters() + "\n");)
        } else {
            for (int c = 0; c < k; c++) {
                clusters[c].centroid = Element{};
            }
        }
        broadcastCentroids(rank);
        Clusters prior = clusters;
        prior[0].centroid[0]++;  // just to make it different the first time

        //iterate kmeans algorithm
        for (int generation = 0; generation < MAX_FIT_STEPS && prior != clusters; generation++) {
            V(cout<<("Process " + to_string(rank) + " beginning KMeans iteration " + to_string(generation) + "\n");)
            updateDistances(rank);
            prior = clusters;
            VV(cout<<("Process " + to_string(rank) + " prior clusters: " + printClusters() + "\n");)
            updateClusters(rank);
            broadcastCentroids(rank);
            VV(cout<<("Process " + to_string(rank) + " updated clusters: " + printClusters() + "\n");)
            VB(cout<<("Process " + to_string(rank) + " waiting at KMeans iteration end barrier.\n");)
            MPI_Barrier(MPI_COMM_WORLD);
        }
        V(cout<<"Process " + to_string(rank) + " detected convergence or iteration limit.\n";)
        gatherClusterMemberships(rank);
        V(cout<<"Process " + to_string(rank) + " finished working.\n";)
    }

    /**
     * The algorithm constructs k clusters and attempts to populate them with like neighbors.
     * This inner class, Cluster, holds each cluster's centroid (mean) and the index of the objects
     * belonging to this cluster.
     */
    struct Cluster {
        Element centroid;  // the current center (mean) of the elements in the cluster
        std::vector<int> elements;

        // equality is just the centroids, regarless of elements
        friend bool operator==(const Cluster& left, const Cluster& right) {
            return left.centroid == right.centroid;  // equality means the same centroid, regardless of elements
        }
    };

protected:
    Element* elements = nullptr;             // set of elements to classify into k categories (supplied to latest call to fit())
    Element* myElements = nullptr;           // set of elements local to each MPI process
    int elemCount = 0;                       // number of elements in this->elements
    Clusters clusters;                       // k clusters resulting from latest call to fit()
    std::vector<std::array<double,k>> dist;  // dist[i][j] is the distance from elements[i] to clusters[j].centroid
    int* sendOffsets = nullptr;              // starting indices for MPI_Scatterv
    int* sendCounts = nullptr;               // send lengths for MPI_Scatterv
    int world;                               // MPI process count
    int data_n;                              // Total number of data

    ~KMeansMPI() {
        //elements, which is passed externally into fit, is deleted by external code.
        delete[] myElements;
        delete[] sendOffsets;
        delete[] sendCounts;
    }

    /**
     * Get the initial cluster centroids.
     * Default implementation here is to just pick k elements at random from the element
     * set
     * @return list of clusters made by using k random elements as the initial centroids
     */
    virtual void reseedClusters() {
        std::vector<int> seeds;
        std::vector<int> candidates(data_n);
        std::iota(candidates.begin(), candidates.end(), 0);
        auto random = std::mt19937{std::random_device{}()};
        // Note that we need C++20 for std::sample
        std::sample(candidates.begin(), candidates.end(), back_inserter(seeds), k, random);
        V(string s = "";for(int i=0;i<(int)seeds.size();i++){s+=to_string(seeds.at(i)) + ", ";}cout<<("Process 0 seeded " + to_string(k) + " clusters as: " + s + "\n");)
        for (int i = 0; i < k; i++) {
            clusters[i].centroid = elements[seeds[i]];
            clusters[i].elements.clear();
        }
    }

    /**
     * Calculate the distance from each element to each centroid.
     * Place into this->dist which is a k-vector of distances from each element to the kth centroid.
     */
    virtual void updateDistances(int rank) {
        for (int i = 0; i < elemCount; i++) {
            VV(cout<<("Process " + to_string(rank) + ": distances for " + to_string(i) + " " + printElement(myElements[i]) + ": "););
            for (int j = 0; j < k; j++) {
                dist[i][j] = distance(clusters[j].centroid, myElements[i]);// As a note, we do not need the sqrt in Color's euclidean distance.
                                                                           // It doesn't change the result of min and is relatively expensive to calculate.
                VV(cout<<(to_string(dist[i][j]) + " "););
            }
            VV(cout<<endl;);
        }
    }

    /**
     * Recalculate the current clusters based on the new distances shown in this->dist.
     */
    virtual void updateClusters(int rank) {
        // clear old cluster memberships
        for (int i = 0; i < k; i++) {
            clusters[i].elements.clear();
        }

        // reassign cluster memberships and marshal sized centroids
        int sendCount = k * (d + 1);
        int* marshaledSizedCentroids = new int[sendCount];
        for (int i = 0; i < sendCount; i++) {
            marshaledSizedCentroids[i] = 0; // Not initializing the values that encode size can lead to junk data for empty clusters.
        }
        for (int i = 0; i < elemCount; i++) {
            int closestCluster = 0;
            for (int j = 1; j < k; j++) {
                if (dist[i][j] < dist[i][closestCluster]) {
                    closestCluster = j;
                }
            }
            clusters[closestCluster].elements.push_back(i);
            int clusterOffset = closestCluster * (d + 1);
            for (int j = 0; j < d; j++) {
                marshaledSizedCentroids[clusterOffset + j] += myElements[i][j];
            }
            marshaledSizedCentroids[clusterOffset + d]++;
        }
        VV(string s = "";for(int i=0; i<sendCount;i++){s+=(to_string(marshaledSizedCentroids[i])+", ");}cout<<("Process " + to_string(rank) + " marshaled sized clusters as: " + s + ".\n");)
        
        // Gather
        int* allSizedClusters = nullptr;
        if (rank == ROOT) {
            allSizedClusters = new int[world * sendCount];
        }
        V(cout<<("Process " + to_string(rank) + " entering centroids gather.\n");)
        MPI_Gather(marshaledSizedCentroids, sendCount, MPI_INT,
                   allSizedClusters, sendCount, MPI_INT,
                   0, MPI_COMM_WORLD);
        VB(cout<<("Process " + to_string(rank) + " waiting at centroids gather barrier.\n");)
        MPI_Barrier(MPI_COMM_WORLD);
        delete[] marshaledSizedCentroids;
        V(cout<<("Process " + to_string(rank) + " exiting cluster gather.\n"););

        // Unmarshal and combine clusters
        if (rank == ROOT) {
            unmarshalAndCombineSizedCentroids(allSizedClusters);
        }
    }

    /**
     * Subclass-supplied method to calculate the distance between two elements
     * @param a one element
     * @param b another element
     * @return distance from a to b (or more abstract metric); distance(a,b) >= 0.0 always
     */
    virtual double distance(const Element& a, const Element& b) const = 0;

    // Shares the value of data_n (passed in at fit) to all processes in the group.
    void broadcastN(int rank) {
        VB(cout<<("Process " + to_string(rank) + " waiting at data_n bcast barrier\n");)
        MPI_Bcast(&data_n, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Apportions the elements near-equally among all processes.
    void scatterElements(int rank) {
        //marshal elements
        u_char* marshaledElements = nullptr;
        if (rank == ROOT) {
            marshaledElements = marshalElements();
        }

        //scatterv the data
        V(cout<<("Process " + to_string(rank) + " entering elements scatterv.\n"););
        u_char* myMarshaledElements = new u_char[elemCount * d];
        MPI_Scatterv(marshaledElements, sendCounts, sendOffsets, 
                     MPI_UNSIGNED_CHAR, myMarshaledElements, elemCount * d,
                     MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
        VB(cout<<("Process " + to_string(rank) + " waiting at elements scatterv barrier\n");)
        MPI_Barrier(MPI_COMM_WORLD);
        delete[] marshaledElements;
        
        //unmarshal elements
        VV(cout<<("Process " + to_string(rank) + " unmarshaling elements\n");)
        myElements = unmarshalElements(myMarshaledElements);
        delete[] myMarshaledElements;
    }

    // Shares the root's centroids with the rest of the processes. Used for initial centers and weighted-averaged centers after each KMeans iteration.
    void broadcastCentroids(int rank) {
        u_char* marshaledCentroids = rank == ROOT ? marshalCentroids() : new u_char[d * k];
        V(cout<<("Process " + to_string(rank) + " bcasting marshaled centroids.\n");)
        MPI_Bcast(marshaledCentroids, d * k, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
        VB(cout<<("Process " + to_string(rank) + " waiting at initial centroids Bcast barrier.\n"););
        MPI_Barrier(MPI_COMM_WORLD);
        unmarshalCentroids(marshaledCentroids);
        delete[] marshaledCentroids;
        V(cout<<("Process " + to_string(rank) + " unmarshaled " + to_string(k) + " Bcasted centers: " + printClusters() + "\n"););
    }

    // At the end of KMeans, gathers all local memberships to the root process.
    void gatherClusterMemberships(int rank) {
        //Marshal memberships
        V(cout<< ("Process " + to_string(rank) + " beginning membership marshal.\n");)
        int* myMemberships = marshalClusterMemberships(rank);
        int* allMemberships = rank == ROOT ? new int[data_n] : nullptr;
        if (rank == ROOT) {
            for (int i = 0; i < world; i++) {
                sendCounts[i] /= d;  // sendCounts/Offsets vars formerly used to scatter elements of d values, 
                sendOffsets[i] /= d; // but now we only need to scatter a single membership int per element.
            }
        }
        V(string s = "Process " + to_string(rank) + " memberships: ";for (int i = 0; i < elemCount; i++) {s += to_string(myMemberships[i]) + ", ";}cout << s + "\n";)
        //Gatherv
        MPI_Gatherv(myMemberships, elemCount, MPI_INT,
                    allMemberships, sendCounts, sendOffsets, MPI_INT,
                    0, MPI_COMM_WORLD);
        VB(cout<<("Process " + to_string(rank) + " waiting at membership gatherv barrier\n");)
        MPI_Barrier(MPI_COMM_WORLD);
        V(cout<< ("Process " + to_string(rank) + " exiting membership gatherv\n");)
        //unmarshal memberships and assign to clusters
        if (rank == ROOT) {
            unmarshalClusterMemberships(allMemberships);
            V(string s = "Gathered memberships: ";for (int i = 0; i < data_n; i++) {s += to_string(allMemberships[i]) + ", ";}cout << s + "\n";)
        }
        delete[] myMemberships;
        delete[] allMemberships;
    }

    // Marshals all elements at the root into a 1d u_char array.
    u_char* marshalElements() {
        u_char* marshaledElements = new u_char[data_n * d];
        for (int i = 0; i < data_n; i++) {
            for (int j = 0; j < d; j++) {
                marshaledElements[i * d + j] = elements[i][j];
            }
        }
        return marshaledElements;
    }

    // Unmarshals a 1d u_char array into its component Elements.
    Element* unmarshalElements(u_char* marshaledElements) {
        myElements = new Element[elemCount];
        for (int i = 0; i < elemCount; i++) {
            for (int j = 0; j < d; j++) {
                myElements[i][j] = marshaledElements[i * d + j];
            }
        }
        return myElements;
    }

    // Marshals the centroids of all k local clusters into a 1d u_char array.
    u_char* marshalCentroids() {
        u_char* marshaledCentroids = new u_char[d * k];
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < d; j++) {
                marshaledCentroids[i * d + j] = clusters[i].centroid[j];
            }
        }
        return marshaledCentroids;
    }

    // Marshals the size of all k local clusters' element arrays into a 1d int array.
    int* marshalClusterSizes() {
        int* marshaledSizes = new int[k];
        for (int i = 0; i < k; i++) {
            marshaledSizes[i] = clusters[i].elements.size();
        }
        return marshaledSizes;
    }

    // Unmarshals a u_char array into its component centroids.
    void unmarshalCentroids(u_char* marshaledCentroids) {
        for (int i = 0; i < k; i++) {
            //clusters[i].centroid = Element{};
            for (int j = 0; j < d; j++) {
                clusters[i].centroid[j] = marshaledCentroids[i * d + j];
            }
        }
    }

    // Unmarshals sized centroids (aggregate centroid components and a size value), 
    // combines all like clusters from different processes, and reduces them to their final values.
    void unmarshalAndCombineSizedCentroids(int* allSizedClusters) {
        int sendCount = (d + 1) * k;
        VV(string s = "";for(int i=0; i<world*sendCount;i++){s+=(to_string(allSizedClusters[i])+", ");}cout<<("Process 0 received all sized clusters as: " + s + ".\n");)
        int* combinedClusters = new int[k * d];
        int* runningSizes = new int[k];
        for (int i = 0; i < k * d; i++) {
            combinedClusters[i] = 0;
        }
        for (int i = 0; i < k; i++) {
            runningSizes[i] = 0;
        }
        for (int p = 0; p < world; p++) {
            int processOffset = p * sendCount;
            for (int i = 0; i < k; i++) {
                int clusterOffset = i * (d + 1);
                for (int j = 0; j < d; j++) {
                    combinedClusters[i * d + j] += allSizedClusters[processOffset + clusterOffset + j];
                }
                runningSizes[i] += allSizedClusters[processOffset + clusterOffset + d];
            }
        }
        V(cout<<("Process 0 Finished unmarshaling summed clusters.\n");)
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < d; j++) {
                clusters[i].centroid[j] = (u_char)(combinedClusters[i * d + j] / runningSizes[i]);
            }
        }
        VV(cout<<("Process 0 combined clusters: " + printClusters() + "\n");)
        delete[] allSizedClusters;
        delete[] combinedClusters;
        delete[] runningSizes;
    }

    // Creates an int array containing the membership assignments of each local element.
    int* marshalClusterMemberships(int rank) {
        int* myMemberships = new int[elemCount];
        for (int i = 0; i < k; i++) {
            std::vector<int> elems = clusters[i].elements;
            for (int j = 0; j < (int)elems.size(); j++) {
                VV(cout<<("Process " + to_string(rank) + " assigning element " + printElement(myElements[elems[j]]) + " to cluster " + to_string(i) + ".\n");)
                myMemberships[elems[j]] = i;
            }
        }
        return myMemberships;
    }

    // Consumes marshaled memberships and assigns computed total sizes to each cluster.
    void unmarshalClusterMemberships(int* allMemberships) {
        V(cout<<("Process 0 entering membership unmarshal, unpacking " + to_string(data_n) + " memberships.\n");)
        VV(string s = "";for(int i=0;i<data_n;i++){s+=to_string(allMemberships[i]) + ", ";}cout<<("Process 0 received all memberships as: " + s + ".\n");)
        for (int i = 0; i < k; i++) {
            clusters[i].elements.clear();
        }
        for (int i = 0; i < data_n; i++) {
            VV(cout<<to_string(i);)
            VV(cout<<to_string(allMemberships[i]);)
            VV(cout<<("Process 0 received element " + to_string(i) + " assignment to cluster " + to_string(allMemberships[i]) + ".\n");)
            clusters[allMemberships[i]].elements.push_back(i);
        }
    }

    // Helper function for debug macro to print all local cluster centroids.
    std::string printClusters() {
        std::string s = "";
        for (int i = 0; i < k; i++) {
            s += "center " + std::to_string(i) + printElement(clusters[i].centroid) + ", ";
        }
        return s;
    }

    // Helper function for debug macro to print all Elements in a cluster.
    std::string printClusterMembership(Cluster cluster) {
        std::string s = "\tCenter: " + printElement(cluster.centroid) + "\n\tMembers: ";
        for (int i = 0; i < (int)cluster.elements.size(); i++) {
            s += printElement(myElements[cluster.elements[i]]) + ", ";
        }
        s += "\n";
        return s;
    }

    // Helper function for debug macro to print all components of an Element.
    std::string printElement(Element& element) const {
        std::string s = "(";
        for (int j = 0; j < d; j++) {
            s += (std::to_string(element[j]) + ", ");
        }
        return s + ")";
    }
};
