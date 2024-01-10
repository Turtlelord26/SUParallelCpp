#include <iostream>
using namespace std;

__global__ void hello() {
	printf("Hello, world! This is thread %d\n", threadIdx.x);
}
int main(void) {
	hello<<<1,10>>>();
	cudaDeviceSynchronize();
	cout << endl;
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	for (int device = 0; device < deviceCount; ++device) {
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, device);
		cout << "Device " << device << " has compute capability "
			<< deviceProp.major << "." << deviceProp.minor << endl;
	}
	return 0;
}