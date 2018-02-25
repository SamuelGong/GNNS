#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define THREADS 8
#define BLOCKS 4

__global__ void mergeBlocks(int *a, int *temp, int sortedsize)
{
	int id = blockIdx.x;

	int index1 = id * 2 * sortedsize;
	int endIndex1 = index1 + sortedsize;
	int index2 = endIndex1;
	int endIndex2 = index2 + sortedsize;
	int targetIndex = id * 2 * sortedsize;
	int done = 0;
	while (!done)
	{
		if ((index1 == endIndex1) && (index2 < endIndex2))
			temp[targetIndex++] = a[index2++];
		else if ((index2 == endIndex2) && (index1 < endIndex1))
			temp[targetIndex++] = a[index1++];
		else if (a[index1] < a[index2])
			temp[targetIndex++] = a[index1++];
		else
			temp[targetIndex++] = a[index2++];
		if ((index1 == endIndex1) && (index2 == endIndex2))
			done = 1;
	}
}

int main() {
	int blocks = BLOCKS / 2;
	int sortedsize = THREADS;
	while (blocks > 0)
	{
		mergeBlocks <<<blocks, 1 >>>(dev_a, dev_temp, sortedsize);
		cudaMemcpy(dev_a, dev_temp, N * sizeof(int), cudaMemcpyDeviceToDevice);
		blocks /= 2;
		sortedsize *= 2;
	}
	cudaMemcpy(a, dev_a, N * sizeof(int), cudaMemcpyDeviceToHost);
}
