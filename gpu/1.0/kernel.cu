#include <io.h>
#include <process.h>
#include <vectors.h>
#include <read_write.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <curand.h>
#include <curand_kernel.h>
#include <string>
#include <ctime>

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
			getchar();\
            exit(1); \
        } \
    } while (0)

namespace GNNS {

	__device__ float Euclidean(float * a, float * b, unsigned size) {
		float result = 0;
		for (unsigned i = 0; i < size; i++)
			result = result + (a[i] - b[i]) * (a[i] - b[i]);
		return result;
	}

	struct Range {
		int start, end;
		__device__ Range(int s = 0, int e = 0) {
			start = s, end = e;
		}
	};

	__device__ void swap(float & a, float & b) {
		float temp(a);
		a = b;
		b = temp;
	}

	__device__ void swap(int & a, int & b) {
		int temp(a);
		a = b;
		b = temp;
	}

	__device__ void table_sort(float * a, int * b, unsigned size) {
		if (size == 0)
			return;

		Range * r = new Range[size];
		int p = 0;
		float mid;
		r[p++] = Range(0, size - 1);
		while (p) {
			Range range = r[--p];
			if (range.start >= range.end)
				continue;

			mid = a[range.end];
			int left = range.start, right = range.end - 1;
			while (left < right) {
				while (a[left] < mid && left < right) left++;
				while (a[right] >= mid && left < right) right--;
				swap(a[left], a[right]);
				swap(b[left], b[right]);
			}

			if (a[left] >= a[range.end]) {
				swap(a[left], a[range.end]);
				swap(b[left], b[range.end]);
			}
			else
				left++;

			r[p++] = Range(range.start, left - 1);
			r[p++] = Range(left + 1, range.end);
		}
		delete[] r;
	}

	__global__ void
	kNN(int * k, float * data, float * dist, int * index, int * result, int * ri) {
		int vertex_1 = *ri * 100 + gridDim.x * blockIdx.x + blockIdx.y;
		int vertex_2;
		int i;

		for (i = 0; i < 100;  i++) {
			vertex_2 = (threadIdx.x * blockDim.x + threadIdx.y) * 100 + i;
			index[vertex_1 * 10000 + vertex_2] = vertex_2;
			dist[vertex_1 * 10000 + vertex_2] =
				Euclidean(&data[128 * vertex_1], &data[128 * vertex_2], 128);
		}

		__syncthreads();
		if (threadIdx.x == 0 && threadIdx.y == 0) {
			table_sort(&dist[vertex_1 * 10000], &index[vertex_1 * 10000], 10000);
			for (i = 1; i <= *k; i++) {
				result[vertex_1*(*k) + i - 1] = index[vertex_1 * 10000 + i];
			}
		}
	}

	Vectors<int> & build_kNN_Graph(string path, Vectors<float> & base, int k) {

		if (_access(path.c_str(), 0) == 0) {
			return read_file<int>(path);
		}

		time_t start, end;
		Vectors<int> * result = new Vectors<int>();
		result->dim = k;
		result->num = base.num;
		result->data = new int[result->dim * result->num];

		start = clock();
		float * dev_base;
		int * dev_result;
		int * dev_k;
		float * dev_dist;
		int * dev_index;
		int * dev_i;

		cudaMalloc((void**)&dev_base, base.num * base.dim * sizeof(float));
		cudaMalloc((void**)&dev_result, k * base.num * sizeof(int));
		cudaMalloc((void**)&dev_k, sizeof(int));
		cudaMalloc((void**)&dev_dist, sizeof(float) * base.num * base.num);
		cudaMalloc((void**)&dev_index, sizeof(int) * base.num * base.num);
		cudaMalloc((void**)&dev_i, sizeof(int));
		cudaMemcpy(dev_base, base.data, base.num * base.dim * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_k, &k, sizeof(unsigned), cudaMemcpyHostToDevice);

		dim3 blockDim(10, 10);
		dim3 threadDim(10, 10);
		for (int i = 0; i < 100; i++) {
			cudaMemcpy(dev_i, &i, sizeof(int), cudaMemcpyHostToDevice);
			kNN<<<blockDim,threadDim>>>(dev_k, dev_base, dev_dist, dev_index, dev_result, dev_i);
			cudaDeviceSynchronize();
		}

		cudaMemcpy(result->data, dev_result, k * base.num * sizeof(int), cudaMemcpyDeviceToHost);
		/*cudaCheckErrors("cudamemcpy or cuda kernel fail");*/
		cudaFree(dev_base);
		cudaFree(dev_result);
		cudaFree(dev_k);
		cudaFree(dev_dist);
		cudaFree(dev_index);

		end = clock();
		
		std::cout << std::endl << "Time for building the graph: "
			<< (double)(end - start) / CLOCKS_PER_SEC << " second" << std::endl;
		write_file<int>(path, *result);

		return *result;
	}

	__global__ void 
	serve_kernel(float * base, float * query, int * graph, int * result,
		int * ri, int * e, int * s, int * k, int * r, float * dist, int * index) {

		//int vertex = *ri * 16 + blockIdx.x;
		int start = threadIdx.x;
		
		curandState_t state;
		curand_init(*ri * (start+1), 0, 0, &state);
		
		int i, j;
		int no = curand(&state) % 10000;
		int temp_no;
		int min_no;
		float temp_dist;
		float min_dist;

		int count = 0;
		int iter = 0;
		int temp;

		for (i = 0; i < *s; i++) {
			for (j = 0; j < *e; j++) {
				temp_no = graph[1000 * no + j];
				temp_dist = Euclidean(query, &base[temp_no * 128], 128);
				index[*s * *e * start + i * *e + j] = temp_no;
				dist[*s * *e * start + i * *e + j] = temp_dist;
				if (j == 0) {
					min_no = temp_no;
					min_dist = temp_dist;
					continue;
				}
				if (temp_dist < min_dist) {
					min_dist = temp_dist;
					min_no = temp_no;
				}
			}
			no = temp_no;
		}

		__syncthreads();
		if (threadIdx.x == 0) {
			table_sort(dist, index, *e * *r * *s);
			while (iter != *e * *r * *s) {
				if (count == 0) {
					temp = index[iter];
					result[count] = temp;
					++count;
					++iter;
					continue;
				}
				else if (count == *k)
					break;
				else if (index[iter] == temp) {
					++iter;
				}
				else {
					temp = index[iter];
					result[count] = temp;
					++count;
					++iter;
				}
			}
		}
	}

	Vectors<int> & serve(Vectors<float> base, Vectors<float> query, Vectors<int> graph,
		int k, int r, int s, int e) {

		Vectors<int> * result = new Vectors<int>();
		result->dim = k;
		result->num = query.num;
		result->data = new int[result->dim * result->num];

		float * dev_base;
		float * dev_query;
		float * dev_dist;
		int * dev_result;
		int * dev_graph;
		int * dev_index;
		int * dev_i;
		int * dev_e;
		int * dev_s;
		int * dev_k;
		int * dev_r;

		cudaMalloc((void**)&dev_base, base.num * base.dim * sizeof(float));
		cudaCheckErrors("Fail to malloc dev_base!");
		cudaMalloc((void**)&dev_query, query.dim * sizeof(float));
		cudaCheckErrors("Fail to malloc dev_query!");
		cudaMalloc((void**)&dev_dist, e * s * r * sizeof(float));
		cudaCheckErrors("Fail to malloc dev_dist!");
		cudaMalloc((void**)&dev_graph, graph.num * graph.dim * sizeof(int));
		cudaCheckErrors("Fail to malloc dev_graph!");
		cudaMalloc((void**)&dev_result, result->dim * sizeof(int));
		cudaCheckErrors("Fail to malloc dev_result!");
		cudaMalloc((void**)&dev_index, e * s * r * sizeof(int));
		cudaCheckErrors("Fail to malloc dev_index!");
		cudaMalloc((void**)&dev_i, sizeof(int));
		cudaMalloc((void**)&dev_e, sizeof(int));
		cudaMalloc((void**)&dev_s, sizeof(int));
		cudaMalloc((void**)&dev_k, sizeof(int));
		cudaMalloc((void**)&dev_r, sizeof(int));

		cudaMemcpy(dev_base, base.data, base.num * base.dim * sizeof(float), cudaMemcpyHostToDevice);
		cudaCheckErrors("Fail to memcpy dev_base!");
		cudaMemcpy(dev_graph, graph.data, graph.num * graph.dim * sizeof(int), cudaMemcpyHostToDevice);
		cudaCheckErrors("Fail to memcpy dev_graph!");
		cudaMemcpy(dev_e, &e, sizeof(int), cudaMemcpyHostToDevice);
		cudaCheckErrors("Fail to memcpy dev_e!");
		cudaMemcpy(dev_s, &s, sizeof(int), cudaMemcpyHostToDevice);
		cudaCheckErrors("Fail to memcpy dev_s!");
		cudaMemcpy(dev_k, &k, sizeof(int), cudaMemcpyHostToDevice);
		cudaCheckErrors("Fail to memcpy dev_k!");
		cudaMemcpy(dev_r, &r, sizeof(int), cudaMemcpyHostToDevice);
		cudaCheckErrors("Fail to memcpy dev_r!");

		for (int i = 0; i < query.num; i++) {
			cudaMemcpy(dev_i, &i, sizeof(int), cudaMemcpyHostToDevice);
			cudaCheckErrors("Fail to malloc dev_i!");
			cudaMemcpy(dev_query, &query.data[i*query.dim], query.dim * sizeof(float), cudaMemcpyHostToDevice);
			cudaCheckErrors("Fail to memcpy dev_query!");
			serve_kernel<<<1,r>>>(dev_base, dev_query, dev_graph, dev_result,
				dev_i, dev_e, dev_s, dev_k, dev_r, dev_dist, dev_index);
			cudaCheckErrors("Fail to launch!");
			cudaDeviceSynchronize();
			cudaMemcpy(&(result->data[i*result->dim]), dev_result, result->dim * sizeof(int), cudaMemcpyDeviceToHost);
			cudaCheckErrors("Fail to memcpy dev_result!");
		}

		cudaFree(dev_base);
		cudaFree(dev_query);
		cudaFree(dev_graph);
		cudaFree(dev_result);
		cudaFree(dev_dist);
		cudaFree(dev_index);
		cudaFree(dev_i);
		cudaFree(dev_e);
		cudaFree(dev_s);
		cudaFree(dev_k);
		cudaFree(dev_r);

		//for (int i = 0; i < 3; i++) {
		//	for (int j = 0; j < 10; j++)
		//		std::printf("%d ", result->data[10*i+j]);
		//	std::printf("\n");
		//}
			
		return *result;
	}
}