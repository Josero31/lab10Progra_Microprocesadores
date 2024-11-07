#include <iostream>
#include <cuda.h>
#include <chrono>

__global__ void sumaVectores(float *x, float *y, float *z, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        z[i] = x[i] + y[i];
    }
}

int main() {
    int n = 1000000; // Tamaño de los vectores
    size_t size = n * sizeof(float);
    
    // Asignación de memoria en el host
    float *h_x = (float *)malloc(size);
    float *h_y = (float *)malloc(size);
    float *h_z = (float *)malloc(size);

    // Inicialización de vectores
    for (int i = 0; i < n; ++i) {
        h_x[i] = i * 0.5f;
        h_y[i] = i * 0.2f;
    }

    // Asignación de memoria en la GPU
    float *d_x, *d_y, *d_z;
    cudaMalloc(&d_x, size);
    cudaMalloc(&d_y, size);
    cudaMalloc(&d_z, size);

    // Transferencia de datos al dispositivo
    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice);

    // Definir el número de bloques e hilos
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Medir el tiempo de ejecución en la GPU
    auto start_gpu = std::chrono::high_resolution_clock::now();
    sumaVectores<<<numBlocks, blockSize>>>(d_x, d_y, d_z, n);
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();

    // Transferencia de resultado al host
    cudaMemcpy(h_z, d_z, size, cudaMemcpyDeviceToHost);

    // Medir el tiempo en CPU
    auto start_cpu = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n; ++i) {
        h_z[i] = h_x[i] + h_y[i];
    }
    auto end_cpu = std::chrono::high_resolution_clock::now();

    // Liberar memoria en la GPU
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);

    // Liberar memoria en el host
    free(h_x);
    free(h_y);
    free(h_z);

    // Imprimir resultados de tiempo
    std::chrono::duration<float, std::milli> gpu_duration = end_gpu - start_gpu;
    std::chrono::duration<float, std::milli> cpu_duration = end_cpu - start_cpu;
    std::cout << "Tiempo de ejecución en GPU: " << gpu_duration.count() << " ms\n";
    std::cout << "Tiempo de ejecución en CPU: " << cpu_duration.count() << " ms\n";

    return 0;
}
