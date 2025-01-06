// main_parallel_cuda.cu

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <filesystem>

#include "MeanShift.h"
#include "Point.h"
#include "Utils.h"

namespace fs = std::filesystem;

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if(err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n", \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Device function to calculate Euclidean distance
__device__ double device_euclidean_distance(const Point a, const Point b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

// CUDA kernel for mean shift computation
__global__ void mean_shift_kernel(Point* d_points, Point* d_new_points, int num_points, double bandwidth,
                                  double epsilon, char* d_converged, int kernel_type) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    Point point = d_points[idx];
    double sum_x = 0.0, sum_y = 0.0;
    double sum_weight = 0.0;

    for (int i = 0; i < num_points; ++i) {
        double dist = device_euclidean_distance(point, d_points[i]);
        double weight = (kernel_type == FLAT) ? ((dist < bandwidth) ? 1.0 : 0.0)
                                              : exp(- (dist * dist) / (2 * bandwidth * bandwidth));

        if (weight > 0) {
            sum_x += d_points[i].x * weight;
            sum_y += d_points[i].y * weight;
            sum_weight += weight;
        }
    }

    if (sum_weight > 0) {
        d_new_points[idx].x = sum_x / sum_weight;
        d_new_points[idx].y = sum_y / sum_weight;
        double shift_distance = device_euclidean_distance(point, d_new_points[idx]);
        d_converged[idx] = (shift_distance < epsilon);
    } else {
        d_new_points[idx] = point;
        d_converged[idx] = 1;
    }
}

// MeanShift class CUDA run method
void MeanShift::run_cuda(std::vector<Point>& points, const std::string& kernel_name,
                         std::vector<IterationTime>& times, int& cluster_count, int threads_per_block) {
    size_t num_points = points.size();
    size_t point_size = sizeof(Point) * num_points;

    Point* d_points;
    Point* d_new_points;
    char* d_converged;
    CUDA_CHECK(cudaMalloc(&d_points, point_size));
    CUDA_CHECK(cudaMalloc(&d_new_points, point_size));
    CUDA_CHECK(cudaMalloc(&d_converged, sizeof(char) * num_points));

    CUDA_CHECK(cudaMemcpy(d_points, points.data(), point_size, cudaMemcpyHostToDevice));

    int num_blocks = (num_points + threads_per_block - 1) / threads_per_block;
    std::cout << "Threads per block: " << threads_per_block << "\n";

    bool all_converged = false;
    int current_iteration = 0;
    const int max_iterations = 100;
    std::string mode = "cuda_threads_" + std::to_string(threads_per_block);

    save_iteration_to_csv(points, kernel_name, current_iteration, num_points, mode);

    while (!all_converged && current_iteration < max_iterations) {
        all_converged = true;
        auto iter_start = std::chrono::high_resolution_clock::now();

        mean_shift_kernel<<<num_blocks, threads_per_block>>>(d_points, d_new_points, num_points,
                                                             bandwidth, epsilon, d_converged, kernel_type);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<char> converged(num_points);
        CUDA_CHECK(cudaMemcpy(converged.data(), d_converged, sizeof(char) * num_points, cudaMemcpyDeviceToHost));

        int num_converged = 0;
        for (size_t i = 0; i < num_points; ++i) {
            if (converged[i]) {
                num_converged++;
            } else {
                all_converged = false;
            }
        }

        auto iter_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> iter_duration = iter_end - iter_start;

        current_iteration++;
        times.push_back(IterationTime{ current_iteration, iter_duration.count() });

        std::vector<Point> new_points(num_points);
        CUDA_CHECK(cudaMemcpy(new_points.data(), d_new_points, point_size, cudaMemcpyDeviceToHost));

        if (verbose_iterations) {
            std::cout << "\nIteration " << current_iteration << " completed in " << iter_duration.count() << " ms.\n";
            std::cout << "Converged points: " << num_converged << "/" << num_points << "\n";
            std::cout << "Saved to generated_points/iterations/cuda/" << kernel_name << "/" << num_points
                      << "/" << mode << "/mean_shift_result_" << kernel_name << "_cuda_" << num_points
                      << "_threads_" << threads_per_block << "_iter" << current_iteration << ".csv\n";
        }

        save_iteration_to_csv(new_points, kernel_name, current_iteration, num_points, mode);
        CUDA_CHECK(cudaMemcpy(d_points, d_new_points, point_size, cudaMemcpyDeviceToDevice));
    }

    CUDA_CHECK(cudaMemcpy(points.data(), d_points, point_size, cudaMemcpyDeviceToHost));
    cluster_count = count_clusters(points, bandwidth);

    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_new_points));
    CUDA_CHECK(cudaFree(d_converged));
}

// Main function
int main() {
    std::string data_dir = "generated_points";
    std::vector<size_t> sizes = { 10000, 25000, 50000, 100000 };
    std::vector<int> thread_options = { 64, 128, 256, 512, 1024 };

    for (const auto& size : sizes) {
        std::string filename = data_dir + "/points_" + std::to_string(size) + ".csv";
        std::vector<Point> points_original = read_points_from_csv(filename);

        if (points_original.empty()) {
            std::cout << "Error: Unable to read points from " << filename << "\n";
            continue;
        }

        for (const auto& kernel_type : { FLAT, GAUSSIAN }) {
            std::string kernel_name = (kernel_type == FLAT) ? "flat" : "gaussian";
            double bandwidth = (kernel_type == FLAT) ? 20.0 : 1.0;
            double epsilon = (kernel_type == FLAT) ? 1.0 : 0.2;

            for (int threads_per_block : thread_options) {
                std::vector<Point> points = points_original;
                std::cout << "Running with " << threads_per_block << " threads per block\n";

                std::string output_filename = "cuda_" + kernel_name + "_" + std::to_string(size) +
                                              "_threads_" + std::to_string(threads_per_block) + "_result.txt";
                std::ofstream output_file(output_filename);
                if (!output_file.is_open()) {
                    std::cout << "Error: Unable to open " << output_filename << "\n";
                    continue;
                }

                std::string mode = "cuda_threads_" + std::to_string(threads_per_block);
                write_output(std::cout, output_file, "Configuration: CUDA, Points: " + std::to_string(size) +
                                                     ", Kernel: " + kernel_name +
                                                     ", Threads/block: " + std::to_string(threads_per_block) + "\n");

                MeanShift mean_shift(bandwidth, epsilon, kernel_type, true, true, false);
                std::vector<IterationTime> times;
                int cluster_count;

                mean_shift.run_cuda(points, kernel_name, times, cluster_count, threads_per_block);

                double total_time = 0.0;
                for (const auto& iter_time : times) {
                    write_output(std::cout, output_file, "Iteration " + std::to_string(iter_time.iteration) +
                                                         ": " + std::to_string(iter_time.iteration_time_ms) + " ms\n");
                    total_time += iter_time.iteration_time_ms;
                }

                write_output(std::cout, output_file, "Total time: " + std::to_string(total_time) + " ms\n");
                write_output(std::cout, output_file, "Number of clusters: " + std::to_string(cluster_count) + "\n\n");

                output_file.close();
            }
        }
    }

    return 0;
}
