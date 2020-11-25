//
// Created by Shun-Cheng Wu on 2020-01-12.
//
#include <cuda_runtime.h>
#include <cstdio>
//#include "meshing_occupancy_shared.h"

#define CUDA_1D_LOOP(i, n)                       \
for (int i = blockIdx.x * blockDim.x + threadIdx.x;     \
i < (n);                                                \
i += blockDim.x * gridDim.x)

int GET_1D_BLOCKS(int n, int num_thread =512) {
    return (n + num_thread - 1) / num_thread;
}
int threadPerBlock = 32;

__device__ __host__ inline void calcGridPos(int i, int X,int Y, int Z, int &x,int&y,int&z)
{
//    z = (int) floor(double(i) / double(X * Y));
//    y = (int) floor(double(i - z * X * Y) / double(X));
//    x = i - (z * X * Y) - (y * X);
    x = i / (X*Y);
    y = (i % (X*Y)) / X;
    z = (i % (X*Y)) % X;
}

static const __constant__ int32_t index_offset[12][3]={
        {0,1,3},{0,3,2},
        {4,0,2},{4,2,6},
        {5,4,6},{5,6,7},
        {1,5,7},{1,7,3},
        {2,3,7},{2,7,6},
        {1,0,4},{1,4,5}
};

static const __constant__ float vertice_offsets[8][3] {
        {-0.5,-0.5,-0.5},
        {-0.5,-0.5,0.5},
        {-0.5,0.5,-0.5},
        {-0.5,0.5,0.5},
        {0.5,-0.5,-0.5},
        {0.5,-0.5,0.5},
        {0.5,0.5,-0.5},
        {0.5,0.5,0.5}
};

__global__ void process_device(int size, int X, int Y, int Z, float *volume_data, int *in_label_data,
        float *vertices_data, int *faces_data, int *label_data, float threshold, float voxel_scale, int *counter, int *face_counter){
    CUDA_1D_LOOP(i, size) {
        if (volume_data[i] > threshold) {
            int x,y,z;
            calcGridPos(i,X,Y,Z,x,y,z);
            int triangleId = atomicAdd(face_counter, 12);
            int vertexId = atomicAdd(counter, 8);
            for (auto off : index_offset) {
                faces_data[triangleId*3+0] = off[0] + vertexId;
                faces_data[triangleId*3+1] = off[1] + vertexId;
                faces_data[triangleId*3+2] = off[2] + vertexId;
                triangleId++;
            }
            for (auto vertice_offset : vertice_offsets) {
                vertices_data[vertexId*3 + 0] = x + vertice_offset[0]*voxel_scale;
                vertices_data[vertexId*3 + 1] = y + vertice_offset[1]*voxel_scale;
                vertices_data[vertexId*3 + 2] = z + vertice_offset[2]*voxel_scale;
                if (in_label_data)
                    label_data[vertexId] = in_label_data[i];
                vertexId++;
            }
        }
    }
}

int process (int size, int X, int Y, int Z, float *volume_data, int *in_label_data,
             float *vertices_data, int *faces_data, int *label_data, float threshold, float voxel_scale) {
    int *face_counter, *vertice_counter;
    cudaMalloc((void**)&face_counter, 1 * sizeof(int));
    cudaMalloc((void**)&vertice_counter, 1 * sizeof(int));
    cudaMemset(face_counter,0,1);
    cudaMemset(vertice_counter,0,1);

    process_device<<<GET_1D_BLOCKS(size, threadPerBlock),threadPerBlock>>>
    (size,X,Y,Z,volume_data,in_label_data,vertices_data,faces_data,label_data,threshold,voxel_scale, vertice_counter, face_counter);

    cudaFree(vertice_counter);
    cudaFree(face_counter);
    return 0;
}

__global__ void classifyVoxel_device(int size, float *volume, float threshold, int *counter)
{
    CUDA_1D_LOOP(i, size)
    {
        if(volume[i] > threshold)
            atomicAdd(counter, 1);
    }
}

int classifyVoxel(int size, float* volume, float threshold){
    int *counter_cuda, counter_cpu=0;
    cudaMalloc((void**)&counter_cuda, 1 * sizeof(int));
    cudaMemset(counter_cuda,0,1);

    cudaDeviceSynchronize();
//    printf("Counter_cpu: %d\n", counter_cpu);
    classifyVoxel_device<<<GET_1D_BLOCKS(size, threadPerBlock),threadPerBlock>>>
    (size,volume,threshold,counter_cuda);
    cudaDeviceSynchronize();
    cudaMemcpy(&counter_cpu, counter_cuda, 1 * sizeof(int), cudaMemcpyDeviceToHost);
//    printf("Counter_cpu: %d\n", counter_cpu);
    cudaFree(counter_cuda);
    return counter_cpu;
}

__global__ void label2color_device(int size, int *label, long *color, long *output) {
    CUDA_1D_LOOP(i,size) {
        output[i * 3 + 0] = color[label[i] * 3 + 0];
        output[i * 3 + 1] = color[label[i] * 3 + 1];
        output[i * 3 + 2] = color[label[i] * 3 + 2];
    }
}

int label2color(int size, int *label, long *color, long *output){
    cudaDeviceSynchronize();
    label2color_device<<<GET_1D_BLOCKS(size,threadPerBlock),threadPerBlock>>>
    (size,label, color, output);
    cudaDeviceSynchronize();
}

