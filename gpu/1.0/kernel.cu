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

	typedef long long sort_t;
	typedef long long base_t;

	__device__ float Euclidean(float * a, float * b, int size) {
		float result = 0;
		for (int i = 0; i < size; i++)
			result = result + (a[i] - b[i]) * (a[i] - b[i]);
		return result;
	}

	struct Range {
		sort_t start, end;
		__device__ Range(sort_t s = 0, sort_t e = 0) {
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

	__device__ void table_sort(float * a, int * b, sort_t size) {
		if (size == 0)
			return;

		Range * r = new Range[size];
		sort_t p = 0;
		sort_t left, right;
		float mid;
		r[p++] = Range(0, size - 1);
		while (p) {
			Range range = r[--p];
			if (range.start >= range.end)
				continue;

			mid = a[range.end];
			left = range.start, right = range.end - 1;
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
	kNN(int * k, float * data, float * dist, int * index, int * result, int * ri,
		int * vertices_per_block, int * blocks_per_iteration, int * pairs_per_threads, 
		int * data_num, int * data_dim) {

		int vertex_1 = *ri * *blocks_per_iteration * *vertices_per_block + blockIdx.x;
		int vertex_2;
		int i;

		for (i = 0; i < *pairs_per_threads;  i++) {
			vertex_2 = threadIdx.x * *pairs_per_threads + i;
			index[vertex_1 * *data_num + vertex_2] = vertex_2;
			dist[vertex_1 * *data_num + vertex_2] =
				Euclidean(&data[*data_dim * vertex_1], &data[*data_dim * vertex_2], *data_dim);
		}

		__syncthreads();
		if (threadIdx.x == 0) {
			table_sort(&dist[vertex_1 * *data_num], &index[vertex_1 * *data_num], *data_num);
			for (i = 1; i <= *k; i++) {
				result[vertex_1*(*k) + i - 1] = index[vertex_1 * *data_num + i];
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
		

		const int pairs_per_thread = 100;
		int threads_per_vertex = base.num / pairs_per_thread;
		const int vertices_per_block = 1;
		int blocks = base.num / vertices_per_block;
		const int parallel = 100;				// blocks per iteration

		float * dev_base;
		int * dev_result;
		int * dev_k;
		float * dev_dist;
		int * dev_index;
		int * dev_i;
		int * dev_v_p_b;
		int * dev_b_p_i;
		int * dev_p_p_t;
		int * dev_data_num;
		int * dev_data_dim;

		cudaMalloc((void**)&dev_base, base.num * base.dim * sizeof(float));
		cudaMalloc((void**)&dev_result, k * base.num * sizeof(int));
		cudaMalloc((void**)&dev_k, sizeof(int));
		cudaMalloc((void**)&dev_dist, sizeof(float) * base.num * base.num);
		cudaMalloc((void**)&dev_index, sizeof(int) * base.num * base.num);
		cudaMalloc((void**)&dev_i, sizeof(int));
		cudaMalloc((void**)&dev_v_p_b, sizeof(int));
		cudaMalloc((void**)&dev_b_p_i, sizeof(int));
		cudaMalloc((void**)&dev_p_p_t, sizeof(int));
		cudaMalloc((void**)&dev_data_num, sizeof(int));
		cudaMalloc((void**)&dev_data_dim, sizeof(int));

		cudaMemcpy(dev_base, base.data, base.num * base.dim * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_k, &k, sizeof(unsigned), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_v_p_b, &vertices_per_block, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_b_p_i, &parallel, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_p_p_t, &pairs_per_thread, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_data_dim, &base.dim, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_data_num, &base.num, sizeof(int), cudaMemcpyHostToDevice);
		
		for (int i = 0; i < blocks / parallel; i++) {
			cudaMemcpy(dev_i, &i, sizeof(int), cudaMemcpyHostToDevice);
			kNN<<<parallel, threads_per_vertex>>>(dev_k, dev_base, dev_dist, dev_index, 
				dev_result, dev_i, dev_v_p_b, dev_b_p_i, dev_p_p_t, dev_data_num, dev_data_dim);
			cudaDeviceSynchronize();
		}
		cudaMemcpy(result->data, dev_result, k * base.num * sizeof(int), cudaMemcpyDeviceToHost);

		cudaFree(dev_base);
		cudaFree(dev_result);
		cudaFree(dev_k);
		cudaFree(dev_dist);
		cudaFree(dev_index);
		cudaFree(dev_v_p_b);
		cudaFree(dev_b_p_i);
		cudaFree(dev_p_p_t);
		cudaFree(dev_data_num);
		cudaFree(dev_data_dim);

		end = clock();
		std::cout << std::endl << "Time for building the graph: "
			<< (double)(end - start) / CLOCKS_PER_SEC << " second" << std::endl;
		write_file<int>(path, *result);
		return *result;
	}

	__global__ void 
	serve_kernel(float * base, float * query, int * graph, int * result,
		int * ri, int * e, int * s, int * k, int * k_, int * r, float * dist, int * index,
		int * base_num, int * base_dim) {

		int parallel = blockIdx.x;
		int start = threadIdx.x;
		
		int temp_no;
		int min_no;
		float temp_dist;
		float min_dist;

		int count;
		sort_t iter;
		int temp;

		int query_offset = *base_dim * parallel;
		sort_t dist_offset = sort_t(*e) * *r * *s * parallel;
		int result_offset = *k * parallel;

		curandState_t state;
		curand_init(start, parallel, *ri, &state);
		int no = curand(&state) % *base_num;

		int i, j;
		for (i = 0; i < *s; i++) {
			for (j = 0; j < *e; j++) {
				temp_no = graph[*k_ * no + j];
				temp_dist = Euclidean(query + query_offset,
					&base[base_t(temp_no) * *base_dim], *base_dim);
				index[dist_offset + sort_t(*s) * *e * start + i * *e + j] = temp_no;
				dist[dist_offset + sort_t(*s) * *e * start + i * *e + j] = temp_dist;

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
			no = min_no;
		}

		__syncthreads();

		if (threadIdx.x == 0) {
			iter = dist_offset;
			count = result_offset;

			table_sort(dist + dist_offset, index + dist_offset, sort_t(*e) * *r * *s);
			while (iter != dist_offset + sort_t(*e) * *r * *s) {
				if (count == result_offset) {
					temp = index[iter];
					result[count] = temp;
					++count;
					++iter;
					continue;
				}
				else if (count == result_offset + *k)
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
		int k, int k_, int r, int s, int e) {

		const int parallel = 100;
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
		int * dev_k_;
		int * dev_r;
		int * dev_base_num;
		int * dev_base_dim;

		cudaMalloc((void**)&dev_base, base.num * base.dim * sizeof(float));
		cudaMalloc((void**)&dev_query, parallel * query.dim * sizeof(float));
		cudaMalloc((void**)&dev_dist, parallel * e * s * r * sizeof(float));
		cudaMalloc((void**)&dev_graph, graph.num * graph.dim * sizeof(int));
		cudaMalloc((void**)&dev_result, parallel * result->dim * sizeof(int));
		cudaMalloc((void**)&dev_index, parallel * e * s * r * sizeof(int));
		cudaMalloc((void**)&dev_i, sizeof(int));
		cudaMalloc((void**)&dev_e, sizeof(int));
		cudaMalloc((void**)&dev_s, sizeof(int));
		cudaMalloc((void**)&dev_k, sizeof(int));
		cudaMalloc((void**)&dev_k_, sizeof(int));
		cudaMalloc((void**)&dev_r, sizeof(int));
		cudaMalloc((void**)&dev_base_num, sizeof(int));
		cudaMalloc((void**)&dev_base_dim, sizeof(int));

		cudaMemcpy(dev_base, base.data, base.num * base.dim * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_graph, graph.data, graph.num * graph.dim * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_e, &e, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_s, &s, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_k, &k, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_k_, &k_, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_r, &r, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_base_num, &(base.num), sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_base_dim, &(base.dim), sizeof(int), cudaMemcpyHostToDevice);

		for (int i = 0; i < query.num / parallel; i++) {
			cudaMemcpy(dev_i, &i, sizeof(int), cudaMemcpyHostToDevice);
			cudaCheckErrors("Fail to malloc dev_i!");
			cudaMemcpy(dev_query, &query.data[i*parallel*query.dim], parallel * query.dim * sizeof(float), cudaMemcpyHostToDevice);
			cudaCheckErrors("Fail to memcpy dev_query!");
			serve_kernel<<<parallel,r>>>(dev_base, dev_query, dev_graph, dev_result,
				dev_i, dev_e, dev_s, dev_k, dev_k_, dev_r, dev_dist, dev_index,
				dev_base_num, dev_base_dim);
			cudaCheckErrors("Fail to launch!");
			cudaDeviceSynchronize();
			cudaCheckErrors("Fail to sychronize!");
			cudaMemcpy(&(result->data[i*parallel*result->dim]), dev_result, parallel * result->dim * sizeof(int), cudaMemcpyDeviceToHost);
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
		cudaFree(dev_k_);
		cudaFree(dev_r);
		cudaFree(dev_base_num);
		cudaFree(dev_base_dim);
			
		return *result;
	}
}