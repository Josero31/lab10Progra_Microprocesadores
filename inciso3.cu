#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define CHANNELS 3
#define BINS 256

__global__ void calculate_histogram(unsigned char *image, int width, int height, int *hist_r, int *hist_g, int *hist_b) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = (y * width + x) * CHANNELS;

    if (x < width && y < height) {
        atomicAdd(&hist_r[image[idx]], 1);
        atomicAdd(&hist_g[image[idx + 1]], 1);
        atomicAdd(&hist_b[image[idx + 2]], 1);
    }
}

int main() {
    // Suponiendo que la imagen es de 512x512 píxeles
    int width = 512;
    int height = 512;
    int imageSize = width * height * CHANNELS;

    // Asignar memoria en el host
    std::vector<unsigned char> h_image(imageSize);
    std::vector<int> h_hist_r(BINS, 0);
    std::vector<int> h_hist_g(BINS, 0);
    std::vector<int> h_hist_b(BINS, 0);

    // Inicializar la imagen con datos aleatorios (para el ejemplo)
    for (int i = 0; i < imageSize; ++i) {
        h_image[i] = rand() % 256;
    }

    // Asignar memoria en el dispositivo
    unsigned char *d_image;
    int *d_hist_r, *d_hist_g, *d_hist_b;
    cudaMalloc((void**)&d_image, imageSize * sizeof(unsigned char));
    cudaMalloc((void**)&d_hist_r, BINS * sizeof(int));
    cudaMalloc((void**)&d_hist_g, BINS * sizeof(int));
    cudaMalloc((void**)&d_hist_b, BINS * sizeof(int));

    // Copiar datos a la GPU
    cudaMemcpy(d_image, h_image.data(), imageSize * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemset(d_hist_r, 0, BINS * sizeof(int));
    cudaMemset(d_hist_g, 0, BINS * sizeof(int));
    cudaMemset(d_hist_b, 0, BINS * sizeof(int));

    // Lanzar kernel CUDA
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    calculate_histogram<<<gridSize, blockSize>>>(d_image, width, height, d_hist_r, d_hist_g, d_hist_b);

    // Copiar resultados al host
    cudaMemcpy(h_hist_r.data(), d_hist_r, BINS * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_hist_g.data(), d_hist_g, BINS * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_hist_b.data(), d_hist_b, BINS * sizeof(int), cudaMemcpyDeviceToHost);

    // Liberar memoria
    cudaFree(d_image);
    cudaFree(d_hist_r);
    cudaFree(d_hist_g);
    cudaFree(d_hist_b);

    // Mostrar resultados
    std::cout << "Histograma R: ";
    for (int i = 0; i < BINS; ++i) {
        std::cout << h_hist_r[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Histograma G: ";
    for (int i = 0; i < BINS; ++i) {
        std::cout << h_hist_g[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Histograma B: ";
    for (int i = 0; i < BINS; ++i) {
        std::cout << h_hist_b[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
